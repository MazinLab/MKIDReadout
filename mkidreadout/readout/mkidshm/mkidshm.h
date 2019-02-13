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

// Compile shared object with: gcc -shared -o libmkidshm.so -fPIC mkidshm.c -lrt -lpthread

#ifndef MKIDSHM_H
#define MKIDSHM_H
typedef int image_t; //can mess around with changing this w/o many subsitutions

typedef struct{
    //metadata
    uint32_t nXPix;
    uint32_t nYPix;
    uint32_t useWvl; //ignore wavelength information if 0
    uint32_t nWvlBins;
    uint32_t wvlStart;
    uint32_t wvlStop;
    uint64_t startTime; //start timestamp of current integration (same as firmware time)
    uint64_t integrationTime; //integration time in half-ms
    char imageBufferName[80]; //form: /imgbuffername (in /dev/shm)
    char takeImageSemName[80];
    char doneImageSemName[80];

} MKID_IMAGE_METADATA;


typedef struct{
    MKID_IMAGE_METADATA *md; //pointer to shared memory buffer

    // For nCounts in pixel (x, y) and wavelength bin i:
    //  image[i*nXPix*nYPix + y*nXPix + x]
    image_t *image; //pointer to shared memory buffer

    sem_t *takeImageSem; //post to start integration
    sem_t *doneImageSem; //post when integration is done

} MKID_IMAGE;


typedef struct{
    // coordinates
    uint8_t x;
    uint8_t y;

    uint64_t time; //arrival time (could also shorten and make relative)
    float wvl; //wavelength

} MKID_PHOTON_EVENT;

typedef struct{
    uint32_t bufferSize; //Size of circular buffer
    uint32_t startInd; //index of earliest valid photon
    uint32_t curInd; //index of last write
    int writing; //1 if currently writing event
    int nCycles; //increment on each complete cycle of buffer

} MKID_EVENT_BUFFER_METADATA;

typedef struct{
    MKID_EVENT_BUFFER_METADATA *md;
    MKID_PHOTON_EVENT *eventBuffer;

} MKID_EVENT_BUFFER;

int openMKIDShmImage(MKID_IMAGE *imageStruct, char *imgName);
int closeMKIDShmImage(MKID_IMAGE *imageStruct);
int createMKIDShmImage(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage);
int populateImageMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nXPix, int nYPix, int useWvl, int nWvlBins, int wvlStart, int wvlStop);
void startIntegration(MKID_IMAGE *image, uint64_t startTime);
void waitForImage(MKID_IMAGE *image);
int checkDoneImage(MKID_IMAGE *image);

void *openShmFile(char *shmName, size_t size, int create);
#endif
