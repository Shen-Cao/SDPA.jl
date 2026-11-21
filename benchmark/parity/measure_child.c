/*
 * A small Linux supervisor so ru_maxrss is not floored by the Python
 * orchestrator's heap inherited by its pre-exec fork. The Python runner
 * owns the external deadline and kills this whole session on timeout.
 */
#define _DEFAULT_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fputs("usage: measure_child measurements.json executable [args...]\n", stderr);
        return 2;
    }
    struct timespec start, end;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        perror("clock_gettime");
        return 2;
    }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 2;
    }
    if (pid == 0) {
        execv(argv[2], argv + 2);
        perror(argv[2]);
        _exit(127);
    }
    struct rusage usage;
    int status;
    pid_t waited;
    do {
        waited = wait4(pid, &status, 0, &usage);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0 || clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        perror("wait4/clock_gettime");
        return 2;
    }
    double elapsed = (double)(end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
    FILE *report = fopen(argv[1], "w");
    if (!report) {
        perror(argv[1]);
        return 2;
    }
    int wrote = fprintf(report,
        "{\"wall_seconds\":%.9f,\"peak_rss_bytes\":%lld,\"user_cpu_seconds\":%.6f,"
        "\"system_cpu_seconds\":%.6f,\"exit_code\":%d}\n",
        elapsed, 1024LL * usage.ru_maxrss,
        usage.ru_utime.tv_sec + 1e-6 * usage.ru_utime.tv_usec,
        usage.ru_stime.tv_sec + 1e-6 * usage.ru_stime.tv_usec, exit_code);
    if (fclose(report) != 0 || wrote < 0) {
        perror("writing measurement");
        return 2;
    }
    return 0;
}
