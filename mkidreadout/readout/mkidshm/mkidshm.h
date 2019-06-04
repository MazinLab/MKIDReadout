#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <fcntl.h>
#include <semaphore.h>
#include <string.h>
#include <errno.h>
#include <time.h>

// Compile shared object with: gcc -shared -o libmkidshm.so -fPIC mkidshm.c -lrt -lpthread

#ifndef MKIDSHM_H
#define MKIDSHM_H

//#define _SHM_DEBUG //turns on debug output

#ifdef __cplusplus
extern "C" {

#endif

#define N_DONE_SEMS 10
#define MKIDSHM_VERSION 4
#define TIMEDWAIT_FUDGE 5000 //half ms
#define STRBUFLEN 80
#define WVLIDLEN 150
typedef int image_t; //can mess around with changing this w/o many subsitutions
typedef float coeff_t;
typedef float wvl_t;

typedef struct{
    //metadata
    uint32_t version;

    uint32_t nCols;
    uint32_t nRows;
    uint32_t useWvl; //ignore wavelength information if 0
    uint32_t nWvlBins;
    uint32_t useEdgeBins; //if 1 add bins for photons out of range
    uint32_t wvlStart;
    uint32_t wvlStop;
    uint32_t valid; //if 0 image is invalid b/c params changed during integration
    uint64_t startTime; //start timestamp of current integration (same as firmware time)
    uint64_t integrationTime; //integration time in half-ms
    uint32_t takingImage;
    char name[STRBUFLEN];
    char imageBufferName[STRBUFLEN]; //form: /imgbuffername (in /dev/shm)
    char takeImageSemName[STRBUFLEN];
    char doneImageSemName[STRBUFLEN];
    char wavecalID[WVLIDLEN];

} MKID_IMAGE_METADATA;


typedef struct{
    MKID_IMAGE_METADATA *md; //pointer to shared memory buffer

    // For nCounts in pixel (x, y) and wavelength bin i:
    //  image[i*nCols*nRows + y*nCols + x]
    image_t *image; //pointer to shared memory buffer

    sem_t *takeImageSem; //post to start integration
    sem_t **doneImageSemList; //post when integration is done

} MKID_IMAGE;


typedef struct{
    // coordinates
    uint8_t x;
    uint8_t y;

    uint64_t time; //arrival time (could also shorten and make relative)
    wvl_t wvl; //wavelength

} MKID_PHOTON_EVENT;

typedef struct{
    uint32_t version;

    uint32_t size; //Size of circular buffer
    int endInd; //index of last write
    int writing; //1 if currently writing event
    int nCycles; //increment on each complete cycle of buffer
    int useWvl; //1 if using wavecal (otherwise use phase)

    char name[STRBUFLEN]; //form: /imgbuffername (in /dev/shm)
    char eventBufferName[STRBUFLEN]; //form: /imgbuffername (in /dev/shm)
    char newPhotonSemName[STRBUFLEN];
    char wavecalID[WVLIDLEN];

} MKID_EVENT_BUFFER_METADATA;

typedef struct{
    MKID_EVENT_BUFFER_METADATA *md;
    MKID_PHOTON_EVENT *buffer;
    sem_t **newPhotonSemList;

} MKID_EVENT_BUFFER;


int MKIDShmImage_open(MKID_IMAGE *imageStruct, const char *imgName);
int MKIDShmImage_close(MKID_IMAGE *imageStruct);
int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, const char *imgName, MKID_IMAGE *outputImage);
int MKIDShmImage_populateMD(MKID_IMAGE_METADATA *imageMetadata, const char *name, int nCols, int nRows, int useWvl, int nWvlBins, int useEdgeBins, int wvlStart, int wvlStop);
void MKIDShmImage_startIntegration(MKID_IMAGE *image, uint64_t startTime, uint64_t integrationTime);
void MKIDShmImage_wait(MKID_IMAGE *image, int semInd);

//time is in half-ms, cancels integration if stopImage is 1
int MKIDShmImage_timedwait(MKID_IMAGE *image, int semInd, int time, int stopImage);
int MKIDShmImage_checkIfDone(MKID_IMAGE *image, int semInd);
void MKIDShmImage_postDoneSem(MKID_IMAGE *image, int semInd);
void MKIDShmImage_copy(MKID_IMAGE *image, image_t *ouputBuffer);
void MKIDShmImage_setWvlRange(MKID_IMAGE *image, int wvlStart, int wvlStop);
void MKIDShmImage_resetSems(MKID_IMAGE *image);
//void MKIDShmImage_setInvalid(MKID_IMAGE *image);
//void MKIDShmImage_setValid(MKID_IMAGE *image);

int MKIDShmEventBuffer_open(MKID_EVENT_BUFFER *bufferStruct, const char *bufferName);
int MKIDShmEventBuffer_create(MKID_EVENT_BUFFER_METADATA *bufferMetadata, const char *bufferName, MKID_EVENT_BUFFER *outputBuffer);
int MKIDShmEventBuffer_populateMD(MKID_EVENT_BUFFER_METADATA *metadata, const char *name, int size, int useWvl);
void MKIDShmEventBuffer_postDoneSem(MKID_EVENT_BUFFER *buffer, int semInd);
void MKIDShmEventBuffer_resetSems(MKID_EVENT_BUFFER *buffer);
int MKIDShmEventBuffer_addEvent(MKID_EVENT_BUFFER *buffer, MKID_PHOTON_EVENT *photon);
void MKIDShmEventBuffer_reset(MKID_EVENT_BUFFER *eventBuffer);
//int MKIDShmEventBuffer_addEvent(MKID_EVENT_BUFFER *buffer, int x, int y, uint64_t time, coeff_t wvl);


void *openShmFile(const char *shmName, size_t size, int create);

#ifdef __cplusplus
}

#endif

#endif
