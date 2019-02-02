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

// Compile shared object with: gcc -shared -o libmkidshm.so -fPIC shmimage.c -lrt -lpthread

#ifndef SHMIMAGE_H
#define SHMIMAGE_H
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
    double *image; //pointer to shared memory buffer

    sem_t *takeImageSem; //post to start integration
    sem_t *doneImageSem; //post when integration is done

} MKID_IMAGE;

int openMKIDShmImage(MKID_IMAGE *imageStruct, char *imgName);
int closeMKIDShmImage(MKID_IMAGE *imageStruct);
int createMKIDShmImage(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage);
#endif
