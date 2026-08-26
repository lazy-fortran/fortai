#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>

/*
 * The Fortran GGUF reader keeps the index and metadata in native objects, but
 * exposes tensor payloads as read-only byte views.  Mapping the file once is
 * the same ownership model used by llama.cpp: pages are faulted in only when
 * a CPU oracle or a CUDA upload actually touches them, and there is no second
 * heap copy of the model.
 */
void *fortai_gguf_mmap_file(const char *path, size_t *size_out)
{
    int fd;
    struct stat st;
    void *mapping;

    if (path == NULL || size_out == NULL) return NULL;
    *size_out = 0;
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return NULL;
    }
    mapping = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return NULL;
    *size_out = (size_t)st.st_size;
    return mapping;
}

void *fortai_gguf_mmap_slice(void *mapping, size_t offset, size_t size,
                             size_t mapped_size)
{
    if (mapping == NULL || size == 0 || offset > mapped_size || size > mapped_size - offset) {
        return NULL;
    }
    return (void *)((uint8_t *)mapping + offset);
}

void fortai_gguf_munmap_file(void *mapping, size_t size)
{
    if (mapping != NULL && size > 0) (void)munmap(mapping, size);
}

int fortai_gguf_mmap_evict(void *mapping, size_t offset, size_t size,
                           size_t mapped_size)
{
    long page_size;
    uintptr_t begin;
    uintptr_t end;
    uintptr_t mapping_end;

    if (mapping == NULL || size == 0 || offset > mapped_size || size > mapped_size - offset) {
        return EINVAL;
    }
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return EINVAL;
    begin = (uintptr_t)mapping + offset;
    end = begin + size;
    mapping_end = (uintptr_t)mapping + mapped_size;
    begin &= ~((uintptr_t)page_size - 1u);
    if (end > mapping_end) end = mapping_end;
    end = (end + (uintptr_t)page_size - 1u) & ~((uintptr_t)page_size - 1u);
    if (end > mapping_end) end = mapping_end;
    if (end <= begin) return 0;
    return madvise((void *)begin, (size_t)(end - begin), MADV_DONTNEED) == 0 ? 0 : errno;
}
