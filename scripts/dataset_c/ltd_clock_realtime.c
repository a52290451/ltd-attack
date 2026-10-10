#define _GNU_SOURCE
#include <time.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>

/* PILOT ONLY. Changes perceived realtime inside this process;
   monotonic timers and host/kernel PCAP timestamps are unchanged. */
static long clock_offset_seconds(void) {
    const char *raw = getenv("LTD_CLOCK_OFFSET_SECONDS");
    if (!raw || !*raw) return 0;
    int sign = 1;
    if (*raw == '-') { sign = -1; ++raw; }
    else if (*raw == '+') ++raw;
    if (*raw == '\0') return 0;
    long n = 0;
    for (const char *p = raw; *p; ++p) {
        if (*p < '0' || *p > '9') return 0;
        n = 10*n + (*p - '0');
        if (n > 7200) return 0;
    }
    return sign*n;
}
static int is_realtime_clock(clockid_t id) {
    if (id == CLOCK_REALTIME) return 1;
#ifdef CLOCK_REALTIME_COARSE
    if (id == CLOCK_REALTIME_COARSE) return 1;
#endif
#ifdef CLOCK_REALTIME_ALARM
    if (id == CLOCK_REALTIME_ALARM) return 1;
#endif
    return 0;
}
int clock_gettime(clockid_t id, struct timespec *tp) {
    long rc = syscall(SYS_clock_gettime, id, tp);
    if (rc == 0 && is_realtime_clock(id)) tp->tv_sec += clock_offset_seconds();
    return (int)rc;
}
int gettimeofday(struct timeval *tv, void *tz) {
    long rc = syscall(SYS_gettimeofday, tv, tz);
    if (rc == 0) tv->tv_sec += clock_offset_seconds();
    return (int)rc;
}
time_t time(time_t *ptr) {
    struct timespec ts;
    if (syscall(SYS_clock_gettime, CLOCK_REALTIME, &ts) != 0) return (time_t)-1;
    time_t value = (time_t)(ts.tv_sec + clock_offset_seconds());
    if (ptr) *ptr = value;
    return value;
}
