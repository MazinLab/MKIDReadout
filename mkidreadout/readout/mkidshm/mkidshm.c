#include "mkidshm.h"

void *openShmFile(const char *shmName, size_t size, int create){
    char name[80];
    int fd;
    void *shmPtr;
    int flag;

    if(create==1)
        flag = O_RDWR|O_CREAT|O_EXCL;
    else
        flag = O_RDWR;

    snprintf(name, 80, "%s", shmName);

    fd = shm_open(name, flag, S_IWUSR|S_IRUSR|S_IWGRP|S_IRGRP);
    if(fd == -1){
        perror("Error opening shm metadata");
        return NULL;

    }

    
    if(ftruncate(fd, size)==-1){
        perror("Error truncating shm metadata FD");
        close(fd);
        return NULL;

    }

    shmPtr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(shmPtr == MAP_FAILED){
        perror("Error mapping shm metadata");
        close(fd);
        return NULL;

    }


    return shmPtr;

}
    

int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, const char *imgName, MKID_IMAGE *outputImage){
    MKID_IMAGE_METADATA *mdPtr;
    char doneSemName[80];
    image_t *imgPtr;
    int i;

    mdPtr = (MKID_IMAGE_METADATA*)openShmFile(imgName, sizeof(MKID_IMAGE_METADATA), 1);

    if(mdPtr==NULL)
        return -1;

    memcpy(mdPtr, imageMetadata, sizeof(MKID_IMAGE_METADATA)); //copy contents of imageMetadata into shared memory buffer
    outputImage->md = mdPtr;

    // CREATE IMAGE DATA BUFFER
    int imageSize = (mdPtr->nCols)*(mdPtr->nRows)*(mdPtr->nWvlBins);

    imgPtr = (image_t*)openShmFile(mdPtr->imageBufferName, sizeof(image_t)*imageSize, 1);
    if(imgPtr==NULL)
        return -1;

    outputImage->image = imgPtr;

    // OPEN SEMAPHORES
    outputImage->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    if(outputImage->takeImageSem==SEM_FAILED)
        printf("Semaphore creation failed %s\n", strerror(errno));

    outputImage->doneImageSemList = (sem_t**)malloc(N_DONE_SEMS*sizeof(sem_t*));
    for(i=0; i<N_DONE_SEMS; i++){ 
        snprintf(doneSemName, 80, "%s%d", mdPtr->doneImageSemName, i);
        outputImage->doneImageSemList[i] = sem_open(doneSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
        if(outputImage->doneImageSemList[i] == SEM_FAILED)
            printf("Done img semaphore creation failed %s\n", strerror(errno));

    }

    return 0;
    

}
    

int MKIDShmImage_open(MKID_IMAGE *imageStruct, const char *imgName){
    MKID_IMAGE_METADATA *mdPtr;
    image_t *imgPtr;
    char doneSemName[80];
    int i;

    // OPEN METADATA BUFFER
    mdPtr = (MKID_IMAGE_METADATA*)openShmFile(imgName, sizeof(MKID_IMAGE_METADATA), 0);
    if(mdPtr == NULL)
        return -1;

    imageStruct->md = mdPtr;

    // OPEN IMAGE BUFFER 
    int imageSize = (mdPtr->nCols)*(mdPtr->nRows)*(mdPtr->nWvlBins);
    imgPtr = (image_t*)openShmFile(imageStruct->md->imageBufferName, imageSize*sizeof(image_t), 0);
    if(imgPtr == NULL)
        return -1;
 
    imageStruct->image = imgPtr;

    // OPEN SEMAPHORES
    imageStruct->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    imageStruct->doneImageSemList = (sem_t**)malloc(N_DONE_SEMS*sizeof(sem_t*));
    for(i=0; i<N_DONE_SEMS; i++){ 
        snprintf(doneSemName, 80, "%s%d", mdPtr->doneImageSemName, i);
        imageStruct->doneImageSemList[i] = sem_open(doneSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
        if(imageStruct->doneImageSemList[i] == SEM_FAILED)
            printf("Done img semaphore creation failed %s\n", strerror(errno));

    }

    return 0;

}

int MKIDShmImage_close(MKID_IMAGE *imageStruct){
    int i;

    sem_close(imageStruct->takeImageSem);

    for(i=0; i<N_DONE_SEMS; i++)
        sem_close(imageStruct->doneImageSemList[i]);
    free(imageStruct->doneImageSemList);

    munmap(imageStruct->image, sizeof(image_t)*(imageStruct->md->nCols)*(imageStruct->md->nRows)*(imageStruct->md->nWvlBins));
    munmap(imageStruct->md, sizeof(MKID_IMAGE_METADATA));
    return 0;

}

int MKIDShmImage_populateMD(MKID_IMAGE_METADATA *imageMetadata, const char *name, int nCols, int nRows, int useWvl, int nWvlBins, int wvlStart, int wvlStop){
    imageMetadata->nCols = nCols;
    imageMetadata->nRows = nRows;
    imageMetadata->useWvl = useWvl;
    imageMetadata->nWvlBins = nWvlBins;
    imageMetadata->wvlStart = wvlStart;
    imageMetadata->wvlStop = wvlStop;
    imageMetadata->startTime = 0;
    imageMetadata->integrationTime = 0;
    snprintf(imageMetadata->imageBufferName, 80, "%s.buf", name);
    snprintf(imageMetadata->takeImageSemName, 80, "%s.takeImg", name);
    snprintf(imageMetadata->doneImageSemName, 80, "%s.doneImg", name);
    return 0;

}

void MKIDShmImage_startIntegration(MKID_IMAGE *image, uint64_t startTime, uint64_t integrationTime){
    image->md->startTime = startTime;
    image->md->integrationTime = integrationTime;
    
    sem_post(image->takeImageSem);
    
    
}

void MKIDShmImage_setWvlRange(MKID_IMAGE *image, int wvlStart, int wvlStop){
    image->md->wvlStart = wvlStart;
    image->md->wvlStop = wvlStop;

}

void MKIDShmImage_postDoneSem(MKID_IMAGE *image, int semInd){
    int i;
    if(semInd==-1)
        for(i=0; i<N_DONE_SEMS; i++)
            sem_post(image->doneImageSemList[i]);
    else
        sem_post(image->doneImageSemList[semInd]);

}

//Blocking
void MKIDShmImage_wait(MKID_IMAGE *image, int semInd){
    sem_wait(image->doneImageSemList[semInd]);}

//Non-blocking
int MKIDShmImage_checkIfDone(MKID_IMAGE *image, int semInd){
    return sem_trywait(image->doneImageSemList[semInd]);}

void MKIDShmImage_copy(MKID_IMAGE *image, image_t *outputBuffer){
    memcpy(outputBuffer, image->image, sizeof(image_t) * image->md->nCols * image->md->nRows * image->md->nWvlBins);}

