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
#include <assert.h>
#include "mkidshm.h"

#define _POSIX_C_SOURCE 200809L
#define BUFLEN 1504
#define MAX_PACKSIZE 12928
#define DEFAULT_PORT 50000
#define SHAREDBUF 536870912
#define RINGBUF_SIZE 536870912
#define RAD_TO_DEG 57.2957795131
#define BINWRITER_MINSIZE 808
#define TSOFFS 1546300800 //Jan 1 2019 UTC
#define STRBUF 80
#define SHM_NAME_LEN 80
#define ENERGY_BIN_PT 16384 //2^14
#define PHASE_BIN_PT 32768.0 //2^14
#define H_TIMES_C 1239.842 // units: eV*nm
#define READER_THREAD 0
#define BIN_WRITER_THREAD 1
#define SHM_IMAGE_WRITER_THREAD 2
#define EVENT_BUFF_WRITER_THREAD 3

//#define _TIMING_TEST //turn on when you want a ts discrepancy file
//#define _DEBUG_OUTPUT //turns on (fairly obtrusive) debugging output
//#define _DEBUG_READER_OUTPUT

#define handle_error_en(en, msg) \
        do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

typedef float wvlcoeff_t;

typedef struct {
    int baseline:17;
    int phase:18;
    unsigned int timestamp:9;
    unsigned int ycoord:10;
    unsigned int xcoord:10;
}__attribute__((packed)) PHOTON_WORD;;

typedef struct {
    unsigned long timestamp:36;
    unsigned int frame:12;
    unsigned int roach:8;
    unsigned int start:8;
}__attribute__((packed)) STREAM_HEADER;;

typedef struct {
    uint64_t unread;
    char data[SHAREDBUF];
} READOUT_STREAM;

typedef struct{
    uint8_t data[RINGBUF_SIZE];
    uint64_t writeInd;
    uint64_t nCycles;

} RINGBUFFER;

typedef struct{
    char solutionFile[STRBUF];
    int writing;
    uint32_t nCols;
    uint32_t nRows;
    // Each pixel has 3 coefficients, with address given by 
    // &a = 3*(nCols*y + x); &b = &a + 1; &c = &a + 2
    wvlcoeff_t *data;

} WAVECAL_BUFFER;

typedef struct{
    int port;
    RINGBUFFER *packBuf;
    char quitSemName[STRBUF];
    char ringBufResetSemName[STRBUF];

    int cpu; //if cpu=-1 then don't maximize priority

} READER_PARAMS;

typedef struct{
    int writing;
    char writerPath[STRBUF];
    RINGBUFFER *packBuf;
    char quitSemName[STRBUF];
    char ringBufResetSemName[STRBUF];

    int cpu; //if cpu=-1 then don't maximize priority

} BIN_WRITER_PARAMS;

typedef struct{
    RINGBUFFER *packBuf;
    int nRoach;
    int nSharedImages;
    char **sharedImageNames;
    WAVECAL_BUFFER *wavecal; //if NULL don't use wavecal

    char quitSemName[STRBUF];
    char ringBufResetSemName[STRBUF];

    int cpu; //if cpu=-1 then don't maximize priority
    
} SHM_IMAGE_WRITER_PARAMS;

typedef struct{
    RINGBUFFER *packBuf;
    char bufferName[STRBUF];
    WAVECAL_BUFFER *wavecal; //if NULL don't use wavecal

    char quitSemName[STRBUF];
    char ringBufResetSemName[STRBUF];

    int nRows;
    int nCols;

    int cpu; //if cpu=-1 then don't maximize priority

} EVENT_BUFF_WRITER_PARAMS;

typedef struct{
    pthread_attr_t attr;
    pthread_t thread;

} THREAD_PARAMS;

int maximizePriority(int cpu);

void *shmImageWriter(void *prms);
void *binWriter(void *prms);
void *reader(void *prms);
void *eventBuffWriter(void *prms);

void addPacketToImage(MKID_IMAGE *sharedImage, char *photonWord, 
        unsigned int l, WAVECAL_BUFFER *wavecal);
void addPacketToEventBuffer(MKID_EVENT_BUFFER *buffer, char *photonWord, 
        unsigned int l, uint64_t headerTS, WAVECAL_BUFFER *wavecal, int nRows, int nCosl);

int startReaderThread(READER_PARAMS *rparams, THREAD_PARAMS *tparams);
int startBinWriterThread(BIN_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
int startShmImageWriterThread(SHM_IMAGE_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
int startEventBuffWriterThread(EVENT_BUFF_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams);
void quitAllThreads(const char *quitSemName, int nThreads);
void resetSem(const char *quitSemName);
float getWavelength(PHOTON_WORD *photon, WAVECAL_BUFFER *wavecal);
void getRingBufState(RINGBUFFER* packBuf, sem_t *ringBufResetSem, uint64_t *nCycles, uint64_t *writeInd);
void diep(char *s);
