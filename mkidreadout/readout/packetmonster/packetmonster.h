#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdint.h>
#include <sys/time.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <math.h>
#include <byteswap.h>
#include <sys/mman.h>
#include <sched.h>
#include "mkidshm.h"

#define _POSIX_C_SOURCE 200809L
#define BUFLEN 1500
#define DEFAULT_PORT 50000
#define SHAREDBUF 536870912
#define TSOFFS 1514764800
#define STRBUF 80
#define SHM_NAME_LEN 80
#define ENERGY_BIN_PT 16384 //2^14
#define H_TIMES_C 1239.842 // units: eV*nm
#define READER_THREAD 0
#define BIN_WRITER_THREAD 1
#define SHM_IMAGE_WRITER_THREAD 2
#define CIRC_BUFF_WRITER_THREAD 3

struct datapacket {
    unsigned int baseline:17;
    unsigned int wvl:18;
    unsigned int timestamp:9;
    unsigned int ycoord:10;
    unsigned int xcoord:10;
}__attribute__((packed));;

struct hdrpacket {
    unsigned long timestamp:36;
    unsigned int frame:12;
    unsigned int roach:8;
    unsigned int start:8;
}__attribute__((packed));;

typedef struct {
    uint64_t unread;
    char data[SHAREDBUF];
}readoutstream_t;

typedef struct{
    int port;
    int nRoachStreams;
    readoutstream_t **roachStreamList;
    sem_t **imageSems;

    sem_t *quitSem;

    int cpu; //if cpu=-1 then don't maximize priority

} ReaderParams;

typedef struct{
    readoutstream_t *roachStream;
    char *ramdisk;

    sem_t *quitSem;

    int cpu; //if cpu=-1 then don't maximize priority

} BinWriterParams;

typedef struct{
    readoutstream_t *roachStream;
    int nRoach;
    int nSharedImages;
    char **sharedImageNames;
    char *wvlShmName; //if NULL don't use wavecal

    sem_t *quitSem;

    int cpu; //if cpu=-1 then don't maximize priority
    
} ShmImageWriterParams;

typedef struct{
    readoutstream_t *roachStream;
    char *bufferName;
    char *wvlShmName; //if NULL don't use wavecal

    sem_t *quitSem;

    int cpu; //if cpu=-1 then don't maximize priority

} CircBuffWriterParams;

typedef struct{
    pthread_attr_t attr;
    pthread_t thread;

} ThreadParams;

int maximizePriority(int cpu);

void *shmImageWriter(void *prms);
void *binWriter(void *prms);
void *reader(void *prms);
void *circBuffWriter(void *prms);

int startReaderThread(ReaderParams *rparams, ThreadParams *tparams);
int startBinWriterThread(BinWriterParams *rparams, ThreadParams *tparams);
int startShmImageWriterThread(ShmImageWriterParams *rparams, ThreadParams *tparams);
void quitAllThreads(sem_t *quitSem, int nThreads);

