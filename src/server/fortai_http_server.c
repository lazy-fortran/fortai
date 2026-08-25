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
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
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

static volatile sig_atomic_t fortai_stop;

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
    signal(SIGPIPE, SIG_IGN);
    fortai_stop = 0;
    if (fortai_install_signal_handlers() != 0) return -1;
    const int listener = fortai_open_listener(host, port);
    if (listener < 0) return -1;
    while (!fortai_stop) {
        int client = accept(listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }
        char *request = NULL;
        size_t request_length = 0;
        if (fortai_http_read_request(client, &request, &request_length) == 0) {
            (void)fortai_dispatch_request(&server, client, request, (int)request_length);
            free(request);
        }
        shutdown(client, SHUT_RDWR);
        close(client);
    }
    close(listener);
    return 0;
}
