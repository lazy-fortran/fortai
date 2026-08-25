#define _GNU_SOURCE

/*
 * FortAI's resident socket transport.
 *
 * This file intentionally contains no JSON, chat template, generation, or
 * response policy. Those operations are implemented by fortai_native_http.f90
 * and use fortai_string::string_t. C is retained for POSIX socket byte I/O,
 * signal handling, and the CUDA/runtime ABI.
 */
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern int fortai_native_http_handle(const char *, int, const char *, int, char *, int, int *,
    int *, char *, int);

enum {
    FORTAI_HTTP_MAX_REQUEST = 16 * 1024 * 1024,
    FORTAI_HTTP_INITIAL_RESPONSE = 64 * 1024,
    FORTAI_HTTP_MAX_RESPONSE = 64 * 1024 * 1024
};

typedef struct {
    const char *model_path;
    int cuda;
} fortai_transport;

typedef struct {
    const fortai_transport *server;
    int *fds;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    size_t active;
    int stopping;
    pthread_mutex_t mutex;
    pthread_cond_t available;
    pthread_cond_t space;
    pthread_cond_t drained;
} fortai_request_queue;

static volatile sig_atomic_t fortai_stop;
static pthread_mutex_t fortai_dispatch_mutex = PTHREAD_MUTEX_INITIALIZER;

static void fortai_signal_handler(int signal_number) {
    (void)signal_number;
    fortai_stop = 1;
}

static int fortai_install_signal_handlers(void) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = fortai_signal_handler;
    if (sigemptyset(&action.sa_mask) != 0) return -1;
    /* Do not restart accept(): the flag must wake the transport so it can
     * close the listener and let the Fortran owner release the model. */
    action.sa_flags = 0;
    if (sigaction(SIGINT, &action, NULL) != 0) return -1;
    if (sigaction(SIGTERM, &action, NULL) != 0) return -1;
    return 0;
}

static void *fortai_realloc(void *pointer, size_t size) {
    void *result = realloc(pointer, size);
    if (result == NULL && size != 0)
        fprintf(stderr, "fortai-server: allocation failed (%zu bytes)\n", size);
    return result;
}

static int fortai_send_all(int fd, const void *data, size_t length) {
    const char *cursor = (const char *)data;
    while (length > 0) {
        ssize_t sent = send(fd, cursor, length, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (sent == 0) return -1;
        cursor += sent;
        length -= (size_t)sent;
    }
    return 0;
}

static int fortai_http_read_request(int fd, char **request_out, size_t *length_out) {
    size_t capacity = 65536;
    size_t length = 0;
    size_t header_end = 0;
    size_t content_length = 0;
    char *request = (char *)malloc(capacity + 1);
    if (request == NULL) return -1;
    while (length < FORTAI_HTTP_MAX_REQUEST) {
        if (capacity == length) {
            size_t grown_capacity = capacity * 2;
            if (grown_capacity > FORTAI_HTTP_MAX_REQUEST) grown_capacity = FORTAI_HTTP_MAX_REQUEST;
            if (grown_capacity == capacity) break;
            char *grown = (char *)fortai_realloc(request, grown_capacity + 1);
            if (grown == NULL) break;
            request = grown;
            capacity = grown_capacity;
        }
        ssize_t received = recv(fd, request + length, capacity - length, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) break;
        length += (size_t)received;
        request[length] = '\0';
        char *separator = strstr(request, "\r\n\r\n");
        if (separator == NULL) continue;
        header_end = (size_t)(separator - request) + 4;
        char *line = request;
        while (line < separator) {
            char *next = strstr(line, "\r\n");
            if (next == NULL || next > separator) break;
            if (strncasecmp(line, "Content-Length:", 15) == 0)
                content_length = (size_t)strtoull(line + 15, NULL, 10);
            line = next + 2;
        }
        if (content_length > FORTAI_HTTP_MAX_REQUEST ||
            header_end > FORTAI_HTTP_MAX_REQUEST - content_length) break;
        if (length >= header_end + content_length) {
            request[header_end + content_length] = '\0';
            *request_out = request;
            *length_out = header_end + content_length;
            return 0;
        }
    }
    free(request);
    return -1;
}

static const char *fortai_http_reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 500: return "Internal Server Error";
    default: return "FortAI Response";
    }
}

static int fortai_http_status(int fd, int status, const char *mime, const char *body, size_t length) {
    char header[512];
    int header_length = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\nX-FortAI-Backend: fortai\r\n\r\n",
        status, fortai_http_reason(status), mime, length);
    if (header_length <= 0 || (size_t)header_length >= sizeof(header)) return -1;
    if (fortai_send_all(fd, header, (size_t)header_length) != 0) return -1;
    return fortai_send_all(fd, body, length);
}

static int fortai_dispatch_request(const fortai_transport *server, int fd, const char *request, int length) {
    size_t capacity = FORTAI_HTTP_INITIAL_RESPONSE;
    char *response = (char *)malloc(capacity);
    char content_type[128];
    int response_length = 0;
    int status = 500;
    int result;
    if (response == NULL) return -1;
    while (1) {
        result = fortai_native_http_handle(request, length, server->model_path,
            server->cuda, response, (int)capacity, &response_length, &status,
            content_type, (int)sizeof(content_type));
        if (result >= 0) break;
        int required = -result;
        if (required <= 0 || required > FORTAI_HTTP_MAX_RESPONSE) {
            free(response);
            return -1;
        }
        char *grown = (char *)fortai_realloc(response, (size_t)required + 1);
        if (grown == NULL) {
            free(response);
            return -1;
        }
        response = grown;
        capacity = (size_t)required + 1;
    }
    if (result != 0) {
        free(response);
        return -1;
    }
    result = fortai_http_status(fd, status, content_type, response, (size_t)response_length);
    free(response);
    return result;
}

static int fortai_parse_worker_count(void) {
    const char *value = getenv("FORTAI_THREADS_HTTP");
    if (value == NULL || value[0] == '\0' || strcasecmp(value, "0") == 0)
        value = getenv("FORTAI_PARALLEL");
    if (value == NULL || value[0] == '\0') return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1) return 1;
    if (parsed > 256) parsed = 256;
    return (int)parsed;
}

static int fortai_parse_parallel_slots(void) {
    const char *value = getenv("FORTAI_PARALLEL");
    if (value == NULL || value[0] == '\0') return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1) return 1;
    if (parsed > 256) parsed = 256;
    return (int)parsed;
}

static void fortai_close_client(const fortai_transport *server, int client);

static int fortai_process_accept_loop(const fortai_transport *server, int listener) {
    while (!fortai_stop) {
        int client = accept(listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        fortai_close_client(server, client);
    }
    return 0;
}

static int fortai_run_cpu_process_slots(const fortai_transport *server, int listener, int slots) {
    pid_t *children = NULL;
    int child_count = 0;
    int result = 0;

    if (slots <= 1) return -2;
    children = (pid_t *)calloc((size_t)(slots - 1), sizeof(*children));
    if (children == NULL) return -2;
    for (int i = 1; i < slots; ++i) {
        pid_t child = fork();
        if (child < 0) {
            result = -2;
            break;
        }
        if (child == 0) {
            (void)fortai_process_accept_loop(server, listener);
            close(listener);
            _exit(0);
        }
        children[child_count++] = child;
    }
    if (result == 0) result = fortai_process_accept_loop(server, listener);
    fortai_stop = 1;
    for (int i = 0; i < child_count; ++i) {
        (void)kill(children[i], SIGTERM);
    }
    for (int i = 0; i < child_count; ++i) {
        int status = 0;
        while (waitpid(children[i], &status, 0) < 0 && errno == EINTR) {
        }
    }
    free(children);
    return result;
}

static void fortai_close_client(const fortai_transport *server, int client) {
    char *request = NULL;
    size_t request_length = 0;
    if (fortai_http_read_request(client, &request, &request_length) == 0) {
        /* CPU process slots have private model state after fork.  In the
         * threaded transport (and for CUDA, whose contexts are not fork-safe)
         * serialize native calls until device-owned sequence slots exist. */
        pthread_mutex_lock(&fortai_dispatch_mutex);
        (void)fortai_dispatch_request(server, client, request, (int)request_length);
        pthread_mutex_unlock(&fortai_dispatch_mutex);
        free(request);
    }
    shutdown(client, SHUT_RDWR);
    close(client);
}

static int fortai_queue_push(fortai_request_queue *queue, int fd) {
    pthread_mutex_lock(&queue->mutex);
    while (queue->count == queue->capacity && !queue->stopping && !fortai_stop)
        pthread_cond_wait(&queue->space, &queue->mutex);
    if (queue->stopping || fortai_stop) {
        pthread_mutex_unlock(&queue->mutex);
        return -1;
    }
    queue->fds[queue->tail] = fd;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    pthread_cond_signal(&queue->available);
    pthread_mutex_unlock(&queue->mutex);
    return 0;
}

static void *fortai_http_worker(void *argument) {
    fortai_request_queue *queue = (fortai_request_queue *)argument;
    while (1) {
        pthread_mutex_lock(&queue->mutex);
        while (queue->count == 0 && !queue->stopping)
            pthread_cond_wait(&queue->available, &queue->mutex);
        if (queue->count == 0 && queue->stopping) {
            pthread_mutex_unlock(&queue->mutex);
            return NULL;
        }
        int client = queue->fds[queue->head];
        queue->head = (queue->head + 1) % queue->capacity;
        queue->count--;
        queue->active++;
        pthread_cond_signal(&queue->space);
        pthread_mutex_unlock(&queue->mutex);

        fortai_close_client(queue->server, client);

        pthread_mutex_lock(&queue->mutex);
        queue->active--;
        if (queue->count == 0 && queue->active == 0) pthread_cond_broadcast(&queue->drained);
        pthread_mutex_unlock(&queue->mutex);
    }
}

static int fortai_open_listener(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    if (host == NULL || strcmp(host, "0.0.0.0") == 0 || strcmp(host, "*") == 0) {
        address.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        struct hostent *resolved = gethostbyname(host);
        if (resolved == NULL || resolved->h_addrtype != AF_INET) {
            close(fd);
            return -1;
        }
        memcpy(&address.sin_addr, resolved->h_addr, sizeof(address.sin_addr));
    }
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0 || listen(fd, 16) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int fortai_server_set_environment(const char *name, const char *value) {
    if (name == NULL || value == NULL) return -1;
    return setenv(name, value, 1);
}

int fortai_server_online_cpus(void) {
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 && count < INT32_MAX ? (int)count : 1;
}

int fortai_http_transport_run(const char *host, int port, const char *model, int cuda) {
    fortai_transport server = {model, cuda};
    fortai_request_queue queue;
    pthread_t *workers = NULL;
    int worker_count = fortai_parse_worker_count();
    int started_workers = 0;
    int mutex_ready = 0;
    int available_ready = 0;
    int space_ready = 0;
    int drained_ready = 0;
    const int parallel_slots = fortai_parse_parallel_slots();
    signal(SIGPIPE, SIG_IGN);
    fortai_stop = 0;
    if (fortai_install_signal_handlers() != 0) return -1;
    const int listener = fortai_open_listener(host, port);
    if (listener < 0) return -1;

    /* CUDA contexts and streams are not fork-safe.  Native CPU model state is
     * fork-safe after load: immutable GGUF pages remain shared and each child
     * gets private recurrent/KV/work pages on its first write. */
    if (cuda == 0 && parallel_slots > 1) {
        if (setenv("FORTAI_SLOT_MODE", "process", 1) != 0) {
            close(listener);
            return -1;
        }
        fprintf(stderr, "FORTAI_SERVER_SLOT_MODE=process slots=%d\n", parallel_slots);
        const int process_result = fortai_run_cpu_process_slots(&server, listener, parallel_slots);
        close(listener);
        if (process_result != -2) return process_result;
        fortai_stop = 0;
        (void)setenv("FORTAI_SLOT_MODE", "serialized", 1);
        fprintf(stderr, "fortai-server: CPU process slots unavailable; using one serialized model\n");
    } else {
        (void)setenv("FORTAI_SLOT_MODE", "serialized", 1);
    }

    memset(&queue, 0, sizeof(queue));
    queue.server = &server;
    queue.capacity = (size_t)worker_count * 4;
    if (queue.capacity < 16) queue.capacity = 16;
    queue.fds = (int *)calloc(queue.capacity, sizeof(*queue.fds));
    workers = (pthread_t *)calloc((size_t)worker_count, sizeof(*workers));
    if (queue.fds == NULL || workers == NULL) {
        free(queue.fds);
        free(workers);
        close(listener);
        return -1;
    }
    if (pthread_mutex_init(&queue.mutex, NULL) != 0) goto queue_init_failed;
    mutex_ready = 1;
    if (pthread_cond_init(&queue.available, NULL) != 0) goto queue_init_failed;
    available_ready = 1;
    if (pthread_cond_init(&queue.space, NULL) != 0) goto queue_init_failed;
    space_ready = 1;
    if (pthread_cond_init(&queue.drained, NULL) != 0) goto queue_init_failed;
    drained_ready = 1;
    for (int i = 0; i < worker_count; ++i) {
        if (pthread_create(&workers[started_workers], NULL, fortai_http_worker, &queue) != 0) break;
        started_workers++;
    }
    if (started_workers == 0) {
        pthread_mutex_destroy(&queue.mutex);
        pthread_cond_destroy(&queue.available);
        pthread_cond_destroy(&queue.space);
        pthread_cond_destroy(&queue.drained);
        free(queue.fds);
        free(workers);
        close(listener);
        return -1;
    }
    while (!fortai_stop) {
        int client = accept(listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (fortai_queue_push(&queue, client) != 0) {
            shutdown(client, SHUT_RDWR);
            close(client);
            break;
        }
    }
    close(listener);

    pthread_mutex_lock(&queue.mutex);
    queue.stopping = 1;
    pthread_cond_broadcast(&queue.available);
    pthread_cond_broadcast(&queue.space);
    pthread_mutex_unlock(&queue.mutex);
    for (int i = 0; i < started_workers; ++i) pthread_join(workers[i], NULL);
    pthread_mutex_destroy(&queue.mutex);
    pthread_cond_destroy(&queue.available);
    pthread_cond_destroy(&queue.space);
    pthread_cond_destroy(&queue.drained);
    free(queue.fds);
    free(workers);
    return 0;

queue_init_failed:
    if (drained_ready) pthread_cond_destroy(&queue.drained);
    if (space_ready) pthread_cond_destroy(&queue.space);
    if (available_ready) pthread_cond_destroy(&queue.available);
    if (mutex_ready) pthread_mutex_destroy(&queue.mutex);
    free(queue.fds);
    free(workers);
    close(listener);
    return -1;
}
