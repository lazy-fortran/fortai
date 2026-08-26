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
#include <ctype.h>
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
    case 204: return "No Content";
    case 401: return "Unauthorized";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 500: return "Internal Server Error";
    default: return "FortAI Response";
    }
}

static int fortai_copy_header_value(const char *request, size_t request_length, const char *name,
        char *value, size_t capacity) {
    const char *cursor = request;
    const char *end = request + request_length;
    const size_t name_length = strlen(name);
    value[0] = '\0';
    while (cursor < end) {
        const char *line_end = strstr(cursor, "\r\n");
        if (line_end == NULL || line_end > end) line_end = end;
        if ((size_t)(line_end - cursor) > name_length &&
                strncasecmp(cursor, name, name_length) == 0 && cursor[name_length] == ':') {
            const char *first = cursor + name_length + 1;
            while (first < line_end && (*first == ' ' || *first == '\t')) first++;
            const char *last = line_end;
            while (last > first && (last[-1] == ' ' || last[-1] == '\t')) last--;
            size_t copied = (size_t)(last - first);
            if (copied >= capacity) copied = capacity - 1;
            if (memchr(first, '\r', copied) != NULL || memchr(first, '\n', copied) != NULL) return 0;
            memcpy(value, first, copied);
            value[copied] = '\0';
            return (int)copied;
        }
        if (line_end == end) break;
        cursor = line_end + 2;
    }
    return 0;
}

static int fortai_cors_origin_allowed(const char *configured, const char *origin) {
    if (origin == NULL || origin[0] == '\0') return 0;
    if (strcmp(configured, "*") == 0) return 1;
    if (strcasecmp(configured, "localhost") == 0)
        return strstr(origin, "localhost") != NULL || strstr(origin, "127.0.0.1") != NULL ||
            strstr(origin, "[::1]") != NULL;
    const char *cursor = configured;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') cursor++;
        const char *last = cursor;
        while (*last != '\0' && *last != ',') last++;
        while (last > cursor && (last[-1] == ' ' || last[-1] == '\t')) last--;
        if ((size_t)(last - cursor) == strlen(origin) && strncasecmp(cursor, origin, (size_t)(last - cursor)) == 0)
            return 1;
        cursor = *last == ',' ? last + 1 : last;
    }
    return 0;
}

static int fortai_key_equals(const char *configured, const char *candidate) {
    const char *cursor = configured;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') cursor++;
        const char *last = cursor;
        while (*last != '\0' && *last != ',') last++;
        while (last > cursor && (last[-1] == ' ' || last[-1] == '\t' || last[-1] == '\r' || last[-1] == '\n')) last--;
        if ((size_t)(last - cursor) == strlen(candidate) &&
                strncmp(cursor, candidate, (size_t)(last - cursor)) == 0) return 1;
        cursor = *last == ',' ? last + 1 : last;
    }
    return 0;
}

static int fortai_api_key_file_matches(const char *path, const char *candidate) {
    FILE *file = fopen(path, "r");
    char line[1024];
    if (file == NULL) return 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        char *first = line;
        while (*first != '\0' && isspace((unsigned char)*first)) first++;
        if (*first == '#' || *first == '\0') continue;
        char *last = first + strlen(first);
        while (last > first && isspace((unsigned char)last[-1])) last--;
        *last = '\0';
        if (strcmp(first, candidate) == 0) {
            fclose(file);
            return 1;
        }
    }
    fclose(file);
    return 0;
}

static int fortai_public_path(const char *request, size_t request_length);

static int fortai_authorized(const char *request, size_t request_length) {
    /* llama.cpp leaves health, model discovery, and UI assets public even
     * when an API key is configured.  Keep that contract for probes and the
     * embedded web UI; generation and administration stay protected. */
    if (fortai_public_path(request, request_length)) return 1;
    const char *configured = getenv("LLAMA_API_KEY");
    if (configured == NULL || configured[0] == '\0') configured = getenv("FORTAI_API_KEY");
    if (configured == NULL || configured[0] == '\0') configured = getenv("LLAMACPP_API_KEY");
    const char *key_file = getenv("FORTAI_API_KEY_FILE");
    if (key_file == NULL || key_file[0] == '\0') key_file = getenv("LLAMA_ARG_API_KEY_FILE");
    if (key_file == NULL || key_file[0] == '\0') key_file = getenv("LLAMACPP_API_KEY_FILE");
    if ((configured == NULL || configured[0] == '\0') && (key_file == NULL || key_file[0] == '\0')) return 1;
    char authorization[2048];
    if (fortai_copy_header_value(request, request_length, "Authorization", authorization, sizeof(authorization)) <= 0) {
        if (fortai_copy_header_value(request, request_length, "X-Api-Key", authorization, sizeof(authorization)) <= 0)
            return 0;
    }
    const char *candidate = authorization;
    if (strncasecmp(candidate, "Bearer ", 7) == 0) candidate += 7;
    while (*candidate == ' ' || *candidate == '\t') candidate++;
    if (*candidate == '\0' || strchr(candidate, '\r') != NULL || strchr(candidate, '\n') != NULL) return 0;
    if (configured != NULL && configured[0] != '\0' && fortai_key_equals(configured, candidate)) return 1;
    return key_file != NULL && key_file[0] != '\0' && fortai_api_key_file_matches(key_file, candidate);
}

static const char *fortai_safe_env(const char *name, const char *fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0' || strchr(value, '\r') != NULL || strchr(value, '\n') != NULL)
        return fallback;
    return value;
}

static int fortai_copy_request_path(const char *request, size_t request_length,
        char *path, size_t capacity) {
    if (request == NULL || path == NULL || capacity == 0) return 0;
    path[0] = '\0';
    const char *first = memchr(request, ' ', request_length);
    if (first == NULL) return 0;
    first++;
    const char *last = memchr(first, ' ', (size_t)(request + request_length - first));
    if (last == NULL || last <= first) return 0;
    const char *query = memchr(first, '?', (size_t)(last - first));
    if (query != NULL) last = query;
    size_t copied = (size_t)(last - first);
    if (copied >= capacity) copied = capacity - 1;
    memcpy(path, first, copied);
    path[copied] = '\0';
    return (int)copied;
}

static int fortai_public_path(const char *request, size_t request_length) {
    char path[2048];
    char normalized_prefix[2048];
    if (fortai_copy_request_path(request, request_length, path, sizeof(path)) <= 0) return 0;
    const char *prefix = getenv("FORTAI_API_PREFIX");
    if (prefix != NULL && prefix[0] != '\0' && strcmp(prefix, "/") != 0) {
        size_t prefix_length = strlen(prefix);
        if (prefix_length >= sizeof(path)) return 0;
        while (prefix_length > 1 && prefix[prefix_length - 1] == '/') prefix_length--;
        if (prefix[0] == '/') {
            memcpy(normalized_prefix, prefix, prefix_length);
            normalized_prefix[prefix_length] = '\0';
        } else {
            if (prefix_length + 1 >= sizeof(normalized_prefix)) return 0;
            normalized_prefix[0] = '/';
            memcpy(normalized_prefix + 1, prefix, prefix_length);
            normalized_prefix[prefix_length + 1] = '\0';
            prefix_length++;
        }
        if (strncmp(path, normalized_prefix, prefix_length) != 0 ||
                (path[prefix_length] != '\0' && path[prefix_length] != '/')) return 0;
        memmove(path, path + prefix_length, strlen(path + prefix_length) + 1);
        if (path[0] == '\0') snprintf(path, sizeof(path), "/");
    }
    if (strcmp(path, "/") == 0 || strcmp(path, "/ui") == 0 || strcmp(path, "/index.html") == 0 ||
            strcmp(path, "/health") == 0 || strcmp(path, "/v1/health") == 0 ||
            strcmp(path, "/models") == 0 || strcmp(path, "/v1/models") == 0) return 1;
    if (strncmp(request, "GET ", 4) != 0 ||
            fortai_safe_env("FORTAI_STATIC_PATH", "")[0] == '\0') return 0;
    const char *dot = strrchr(path, '.');
    if (dot == NULL) return 0;
    return strcasecmp(dot, ".html") == 0 || strcasecmp(dot, ".css") == 0 ||
        strcasecmp(dot, ".js") == 0 || strcasecmp(dot, ".json") == 0 ||
        strcasecmp(dot, ".svg") == 0 || strcasecmp(dot, ".png") == 0 ||
        strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0 ||
        strcasecmp(dot, ".ico") == 0 || strcasecmp(dot, ".woff") == 0 ||
        strcasecmp(dot, ".woff2") == 0 || strcasecmp(dot, ".ttf") == 0;
}

static int fortai_http_status(int fd, int status, const char *mime, const char *body, size_t length,
        const char *origin) {
    char header[2048];
    const char *configured_origins = fortai_safe_env("FORTAI_CORS_ORIGINS", "*");
    const char *methods = fortai_safe_env("FORTAI_CORS_METHODS", "GET, POST, DELETE, OPTIONS");
    const char *allowed_headers = fortai_safe_env("FORTAI_CORS_HEADERS", "*");
    const char *credentials_value = fortai_safe_env("FORTAI_CORS_CREDENTIALS", "true");
    const int credentials = strcasecmp(credentials_value, "0") != 0 &&
        strcasecmp(credentials_value, "false") != 0 && strcasecmp(credentials_value, "off") != 0;
    char allow_origin[512];
    if (credentials && fortai_cors_origin_allowed(configured_origins, origin)) {
        snprintf(allow_origin, sizeof(allow_origin), "%s", origin);
    } else {
        snprintf(allow_origin, sizeof(allow_origin), "%s", configured_origins);
    }
    int header_length = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\nX-FortAI-Backend: fortai\r\n"
        "Access-Control-Allow-Origin: %s\r\nAccess-Control-Allow-Methods: %s\r\n"
        "Access-Control-Allow-Headers: %s\r\nAccess-Control-Allow-Credentials: %s\r\n"
        "%s\r\n",
        status, fortai_http_reason(status), mime, length, allow_origin, methods, allowed_headers,
        credentials ? "true" : "false", (origin != NULL && origin[0] != '\0') ? "Vary: Origin" : "");
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
    char origin[512];
    if (response == NULL) return -1;
    fortai_copy_header_value(request, (size_t)length, "Origin", origin, sizeof(origin));
    if ((size_t)length >= 8 && strncasecmp(request, "OPTIONS ", 8) == 0) {
        result = fortai_http_status(fd, 204, "text/plain; charset=utf-8", "", 0, origin);
        free(response);
        return result;
    }
    if (!fortai_authorized(request, (size_t)length)) {
        const char unauthorized[] = "{\"error\":{\"message\":\"unauthorized\",\"type\":\"authentication_error\"}}\n";
        result = fortai_http_status(fd, 401, "application/json", unauthorized, sizeof(unauthorized) - 1, origin);
        free(response);
        return result;
    }
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
    result = fortai_http_status(fd, status, content_type, response, (size_t)response_length, origin);
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
#ifdef SO_REUSEPORT
    const char *reuse_port = getenv("FORTAI_REUSE_PORT");
    if (reuse_port != NULL && (strcmp(reuse_port, "1") == 0 ||
            strcasecmp(reuse_port, "true") == 0 || strcasecmp(reuse_port, "on") == 0)) {
        setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
    }
#endif
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
