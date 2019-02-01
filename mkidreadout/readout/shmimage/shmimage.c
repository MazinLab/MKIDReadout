#include "shmimage.h"

int createMKIDShmImage(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage){
    int mdfd, imgfd;
    MKID_IMAGE_METADATA *mdPtr;
    double *imgPtr;
    char imageName[80];

    snprintf(imageName, 80, "%s", imgName);

    mdfd = shm_open(imageName, O_RDWR|O_CREAT, S_IWUSR|S_IRUSR|S_IWGRP|S_IRGRP);
    if(mdfd == -1){
        perror("Error opening shm metadata");
        return -1;

    }

    
    if(ftruncate(mdfd, sizeof(MKID_IMAGE_METADATA))==-1){
        perror("Error truncating shm metadata FD");
        return -1;

    }

    mdPtr = (MKID_IMAGE_METADATA*)mmap(NULL, sizeof(MKID_IMAGE_METADATA), PROT_READ | PROT_WRITE, MAP_SHARED, mdfd, 0);
    if(mdPtr == MAP_FAILED){
        perror("Error mapping shm metadata");
        return -1;

    }

    memcpy(mdPtr, imageMetadata, sizeof(MKID_IMAGE_METADATA)); //copy contents of imageMetadata into shared memory buffer
    outputImage->md = mdPtr;

    // CREATE IMAGE DATA BUFFER
    int imageSize = (mdPtr->nXPix)*(mdPtr->nYPix)*(mdPtr->nWvlBins);

    imgfd = shm_open(mdPtr->imageBufferName, O_RDWR|O_CREAT, S_IWUSR|S_IRUSR|S_IWGRP|S_IRGRP);
    if(mdfd == -1){
        perror("Error opening shm metadata");
        return -1;

    }
 
    if(ftruncate(imgfd, sizeof(double)*imageSize)==-1){
        perror("Error truncating shm metadata FD");
        return -1;

    }

    imgPtr = (double*)mmap(NULL, sizeof(double)*imageSize, PROT_READ | PROT_WRITE, MAP_SHARED, imgfd, 0);
    if(imgPtr == MAP_FAILED){
        perror("Error mapping shm metadata");
        return -1;

    }
    outputImage->image = imgPtr;

    // OPEN SEMAPHORES
    outputImage->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    outputImage->doneImageSem = sem_open(mdPtr->doneImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    if((outputImage->takeImageSem==SEM_FAILED)||(outputImage->doneImageSem==SEM_FAILED)) 
        printf("Semaphore creation failed %s\n", strerror(errno));

    close(imgfd);
    close(mdfd);
    return 0;
    

}
    

int openMKIDShmImage(MKID_IMAGE *imageStruct, char *imgName){
    // OPEN IMAGE METADATA BUFFER
    int mdfd, imgfd;
    MKID_IMAGE_METADATA *mdPtr;
    double *imgPtr;
    char imageName[80];

    snprintf(imageName, 80, "%s", imgName);

    mdfd = shm_open(imageName, O_RDWR, S_IWUSR);
    if(mdfd == -1){
        perror("Error opening shm metadata");
        return -1;

    }

    
    if(ftruncate(mdfd, sizeof(MKID_IMAGE_METADATA))==-1){
        perror("Error truncating shm metadata FD");
        return -1;

    }

    mdPtr = (MKID_IMAGE_METADATA*)mmap(NULL, sizeof(MKID_IMAGE_METADATA), PROT_READ | PROT_WRITE, MAP_SHARED, mdfd, 0);
    if(mdPtr == MAP_FAILED){
        perror("Error mapping shm metadata");
        return -1;

    }

    imageStruct->md = mdPtr;

    // OPEN IMAGE DATA BUFFER
    int imageSize = (mdPtr->nXPix)*(mdPtr->nYPix)*(mdPtr->nWvlBins);

    imgfd = shm_open(imageStruct->md->imageBufferName, O_RDWR, S_IWUSR);
    if(mdfd == -1){
        perror("Error opening shm metadata");
        return -1;

    }
 
    if(ftruncate(imgfd, sizeof(double)*imageSize)==-1){
        perror("Error truncating shm metadata FD");
        return -1;

    }

    imgPtr = (double*)mmap(NULL, sizeof(double)*imageSize, PROT_READ | PROT_WRITE, MAP_SHARED, imgfd, 0);
    if(imgPtr == MAP_FAILED){
        perror("Error mapping shm metadata");
        return -1;

    }
    
    imageStruct->image = imgPtr;

    // OPEN SEMAPHORES
    imageStruct->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    imageStruct->doneImageSem = sem_open(mdPtr->doneImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    if((imageStruct->takeImageSem==SEM_FAILED)||(imageStruct->doneImageSem==SEM_FAILED)) 
        printf("Semaphore creation failed %s\n", strerror(errno));

    close(imgfd);
    close(mdfd);
    return 0;

}

int closeMKIDShmImage(MKID_IMAGE *imageStruct){
    sem_close(imageStruct->takeImageSem);
    sem_close(imageStruct->doneImageSem);
    munmap(imageStruct->image, sizeof(double)*(imageStruct->md->nXPix)*(imageStruct->md->nYPix)*(imageStruct->md->nWvlBins));
    munmap(imageStruct->md, sizeof(MKID_IMAGE_METADATA));

}
