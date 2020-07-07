#include "mkidshm.h"

void *openShmFile(const char *shmName, size_t size, int create){
    char name[STRBUFLEN];
    char error[200];
    int fd;
    void *shmPtr;
    int flag;

    if(create==1)
        flag = O_RDWR|O_CREAT|O_EXCL;
    else
        flag = O_RDWR;

    snprintf(name, STRBUFLEN, "%s", shmName);

    fd = shm_open(name, flag, S_IWUSR|S_IRUSR|S_IWGRP|S_IRGRP|S_IWOTH|S_IROTH);
    if(fd == -1){
        snprintf(error, 200, "Error opening %s", name);
        perror(error);
        return NULL;

    }

    
    if(ftruncate(fd, size)==-1){
        snprintf(error, 200, "Error truncating %s", name);
        perror(error);
        close(fd);
        return NULL;

    }

    shmPtr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(shmPtr == MAP_FAILED){
        snprintf(error, 200, "Error mapping %s", name);
        perror(error);
        close(fd);
        return NULL;

    }


    close(fd);
    return shmPtr;

}
    

int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, const char *imgName, MKID_IMAGE *outputImage){
    MKID_IMAGE_METADATA *mdPtr;
    char doneSemName[STRBUFLEN + 11];
    image_t *imgPtr;
    int i;
    int depth;

    mdPtr = (MKID_IMAGE_METADATA*)openShmFile(imgName, sizeof(MKID_IMAGE_METADATA), 1);

    if(mdPtr==NULL)
        return -1;

    memcpy(mdPtr, imageMetadata, sizeof(MKID_IMAGE_METADATA)); //copy contents of imageMetadata into shared memory buffer
    outputImage->md = mdPtr;

    // CREATE IMAGE DATA BUFFER
    if(mdPtr->useEdgeBins==1)
        depth = mdPtr->nWvlBins + 2;
    else
        depth = mdPtr->nWvlBins;

    int imageSize = (mdPtr->nCols)*(mdPtr->nRows)*depth;

    imgPtr = (image_t*)openShmFile(mdPtr->imageBufferName, sizeof(image_t)*imageSize, 1);
    if(imgPtr==NULL)
        return -1;

    outputImage->image = imgPtr;

    // OPEN SEMAPHORES
    outputImage->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH, 0);
    if(outputImage->takeImageSem==SEM_FAILED)
        printf("Semaphore creation failed %s\n", strerror(errno));

    outputImage->doneImageSemList = (sem_t**)malloc(N_DONE_SEMS*sizeof(sem_t*));
    for(i=0; i<N_DONE_SEMS; i++){ 
        snprintf(doneSemName, STRBUFLEN+11, "%s%d", mdPtr->doneImageSemName, i);
        outputImage->doneImageSemList[i] = sem_open(doneSemName, O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP, 0);
        if(outputImage->doneImageSemList[i] == SEM_FAILED)
            printf("Done img semaphore creation failed %s\n", strerror(errno));

    }

    return 0;
    

}
    

int MKIDShmImage_open(MKID_IMAGE *imageStruct, const char *imgName){
    MKID_IMAGE_METADATA *mdPtr;
    image_t *imgPtr;
    char doneSemName[STRBUFLEN + 11];
    int i;
    int depth;

    // OPEN METADATA BUFFER
    mdPtr = (MKID_IMAGE_METADATA*)openShmFile(imgName, sizeof(MKID_IMAGE_METADATA), 0);
    if(mdPtr == NULL)
        return -1;


    if(mdPtr->version != MKIDSHM_VERSION){
        printf("ERROR: Version mismatch between libmkidshm and shared memory file");
        return -1;

    }

    imageStruct->md = mdPtr;

    // OPEN IMAGE BUFFER 
    if(mdPtr->useEdgeBins==1)
        depth = mdPtr->nWvlBins + 2;
    else
        depth = mdPtr->nWvlBins;

    int imageSize = (mdPtr->nCols)*(mdPtr->nRows)*depth;
    imgPtr = (image_t*)openShmFile(imageStruct->md->imageBufferName, imageSize*sizeof(image_t), 0);
    if(imgPtr == NULL)
        return -1;
 
    imageStruct->image = imgPtr;

    // OPEN SEMAPHORES
    imageStruct->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    imageStruct->doneImageSemList = (sem_t**)malloc(N_DONE_SEMS*sizeof(sem_t*));
    for(i=0; i<N_DONE_SEMS; i++){ 
        snprintf(doneSemName, STRBUFLEN+11, "%s%d", mdPtr->doneImageSemName, i);
        imageStruct->doneImageSemList[i] = sem_open(doneSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
        if(imageStruct->doneImageSemList[i] == SEM_FAILED)
            printf("Done img semaphore creation failed %s\n", strerror(errno));

    }

    return 0;

}

int MKIDShmImage_close(MKID_IMAGE *imageStruct){
    int i;
    int depth;

    sem_close(imageStruct->takeImageSem);

    for(i=0; i<N_DONE_SEMS; i++)
        sem_close(imageStruct->doneImageSemList[i]);
    free(imageStruct->doneImageSemList);

    if(imageStruct->md->useEdgeBins==1)
        depth = imageStruct->md->nWvlBins + 2;
    else
        depth = imageStruct->md->nWvlBins;

    int imageSize = (imageStruct->md->nCols)*(imageStruct->md->nRows)*depth;

    munmap(imageStruct->image, sizeof(image_t)*imageSize);
    munmap(imageStruct->md, sizeof(MKID_IMAGE_METADATA));
    return 0;

}

int MKIDShmImage_populateMD(MKID_IMAGE_METADATA *imageMetadata, const char *name, int nCols, int nRows, int useWvl, int nWvlBins, int useEdgeBins, int wvlStart, int wvlStop){
    imageMetadata->version = MKIDSHM_VERSION;
    imageMetadata->nCols = nCols;
    imageMetadata->nRows = nRows;
    imageMetadata->useWvl = useWvl;
    imageMetadata->nWvlBins = nWvlBins;
    imageMetadata->useEdgeBins = useEdgeBins;
    imageMetadata->wvlStart = wvlStart;
    imageMetadata->wvlStop = wvlStop;
    imageMetadata->startTime = 0;
    imageMetadata->integrationTime = 0;
    imageMetadata->takingImage = 0;
    imageMetadata->valid = 1;
    snprintf(imageMetadata->name, STRBUFLEN, "%s", name);
    snprintf(imageMetadata->wavecalID, WVLIDLEN, "%s", "none");
    snprintf(imageMetadata->imageBufferName, STRBUFLEN, "%s.buf", name);
    snprintf(imageMetadata->takeImageSemName, STRBUFLEN, "%s.takeImg", name);
    snprintf(imageMetadata->doneImageSemName, STRBUFLEN, "%s.doneImg", name);
    return 0;

}

void MKIDShmImage_startIntegration(MKID_IMAGE *image, uint64_t startTime, uint64_t integrationTime){
    image->md->startTime = startTime;
    image->md->integrationTime = integrationTime;
    image->md->valid = 1;
    
    sem_post(image->takeImageSem);
    
    
}

void MKIDShmImage_setWvlRange(MKID_IMAGE *image, int wvlStart, int wvlStop){
    image->md->wvlStart = wvlStart;
    image->md->wvlStop = wvlStop;
    image->md->valid = 0;

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

int MKIDShmImage_timedwait(MKID_IMAGE *image, int semInd, int time, int stopImage){
    struct timespec tspec;
    int retval;
    #ifdef _SHM_DEBUG
    char error[200];
    #endif 

    time += TIMEDWAIT_FUDGE;
    clock_gettime(CLOCK_REALTIME, &tspec);
    tspec.tv_sec += time/2000;
    tspec.tv_nsec += (time%2000)*500000;
    tspec.tv_sec += tspec.tv_nsec/1000000000;
    tspec.tv_nsec = tspec.tv_nsec%1000000000;

    retval = sem_timedwait(image->doneImageSemList[semInd], &tspec);

    #ifdef _SHM_DEBUG
    if(retval == -1){
        snprintf(error, 200, "Timedwait error %ld", tspec.tv_nsec);
        perror(error);

    }
    #endif

    if((retval == -1) && (stopImage)){
        image->md->takingImage = 0;
        sem_trywait(image->takeImageSem);

    }

    return retval;


}

//Non-blocking
int MKIDShmImage_checkIfDone(MKID_IMAGE *image, int semInd){
    return sem_trywait(image->doneImageSemList[semInd]);}

void MKIDShmImage_copy(MKID_IMAGE *image, image_t *outputBuffer){
    int depth;

    if(image->md->useEdgeBins==1)
        depth = image->md->nWvlBins + 2;
    else
        depth = image->md->nWvlBins;
    int imageSize = (image->md->nCols)*(image->md->nRows)*depth;

    memcpy(outputBuffer, image->image, sizeof(image_t) * imageSize);

}

void MKIDShmImage_resetSems(MKID_IMAGE *image){
    while(sem_trywait(image->takeImageSem)==0);

    int i;
    for(i=0; i<N_DONE_SEMS; i++)
        while(sem_trywait(image->doneImageSemList[i])==0);

}

int MKIDShmEventBuffer_open(MKID_EVENT_BUFFER *bufferStruct, const char *bufferName){
    MKID_EVENT_BUFFER_METADATA *mdPtr;
    MKID_PHOTON_EVENT *bufferPtr;
    char newPhotonSemName[STRBUFLEN + 11];
    int i;
    int depth;

    // OPEN METADATA BUFFER
    mdPtr = (MKID_EVENT_BUFFER_METADATA*)openShmFile(bufferName, sizeof(MKID_EVENT_BUFFER_METADATA), 0);
    if(mdPtr == NULL)
        return -1;


    if(mdPtr->version != MKIDSHM_VERSION){
        printf("ERROR: Version mismatch between libmkidshm and shared memory file");
        return -1;

    }

    bufferStruct->md = mdPtr;

    // OPEN IMAGE BUFFER 
    bufferPtr = (MKID_PHOTON_EVENT*)openShmFile(bufferStruct->md->eventBufferName, bufferStruct->md->size*sizeof(MKID_PHOTON_EVENT), 0);
    if(bufferPtr == NULL)
        return -1;
 
    bufferStruct->buffer = bufferPtr;

    // OPEN SEMAPHORES
    bufferStruct->newPhotonSemList = (sem_t**)malloc(N_DONE_SEMS*sizeof(sem_t*));
    for(i=0; i<N_DONE_SEMS; i++){ 
        snprintf(newPhotonSemName, STRBUFLEN+11, "%s%d", mdPtr->newPhotonSemName, i);
        bufferStruct->newPhotonSemList[i] = sem_open(newPhotonSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
        if(bufferStruct->newPhotonSemList[i] == SEM_FAILED)
            printf("New photon semaphore creation failed %s\n", strerror(errno));

    }

    return 0;

}

int MKIDShmEventBuffer_create(MKID_EVENT_BUFFER_METADATA *bufferMetadata, const char *bufferName, MKID_EVENT_BUFFER *outputBuffer){
    MKID_EVENT_BUFFER_METADATA *mdPtr;
    char newPhotonSemName[STRBUFLEN + 11];
    MKID_PHOTON_EVENT *bufferPtr;
    int i;

    mdPtr = (MKID_EVENT_BUFFER_METADATA*)openShmFile(bufferName, sizeof(MKID_EVENT_BUFFER_METADATA), 1);

    if(mdPtr==NULL)
        return -1;

    memcpy(mdPtr, bufferMetadata, sizeof(MKID_EVENT_BUFFER_METADATA)); //copy contents of imageMetadata into shared memory buffer
    outputBuffer->md = mdPtr;

    // CREATE IMAGE DATA BUFFER
    bufferPtr = (MKID_PHOTON_EVENT*)openShmFile(mdPtr->eventBufferName, sizeof(MKID_PHOTON_EVENT)*mdPtr->size, 1);
    if(bufferPtr==NULL)
        return -1;

    outputBuffer->buffer = bufferPtr;

    // OPEN SEMAPHORES
    outputBuffer->newPhotonSemList = (sem_t**)malloc(N_DONE_SEMS*sizeof(sem_t*));
    for(i=0; i<N_DONE_SEMS; i++){ 
        snprintf(newPhotonSemName, STRBUFLEN+11, "%s%d", mdPtr->newPhotonSemName, i);
        outputBuffer->newPhotonSemList[i] = sem_open(newPhotonSemName, O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP, 0);
        if(outputBuffer->newPhotonSemList[i] == SEM_FAILED){
            printf("New photon semaphore creation failed %s\n", strerror(errno));
            return -1;

        }

    }

    return 0;
    

}

int MKIDShmEventBuffer_populateMD(MKID_EVENT_BUFFER_METADATA *metadata, const char *name, int size, int useWvl){
    metadata->version = MKIDSHM_VERSION;
    metadata->useWvl = useWvl;
    metadata->size = size;
    metadata->writing = 0;
    metadata->nCycles = 0;
    metadata->endInd = -1;
    snprintf(metadata->name, STRBUFLEN, "%s", name);
    snprintf(metadata->wavecalID, WVLIDLEN, "%s", "none");
    snprintf(metadata->eventBufferName, STRBUFLEN, "%s.buf", name);
    snprintf(metadata->newPhotonSemName, STRBUFLEN, "%s.newPhot", name);
    return 0;

}

int MKIDShmEventBuffer_addEvent(MKID_EVENT_BUFFER *buffer, MKID_PHOTON_EVENT *photon){
    buffer->md->writing = 1;
    int writeInd = buffer->md->endInd + 1;
    if(writeInd == buffer->md->size){ //we've reached the end of the buffer
        writeInd = 0;
        buffer->md->nCycles += 1;

    }

    buffer->buffer[writeInd] = *photon; 
    buffer->md->endInd = writeInd;

    buffer->md->writing = 0;
    MKIDShmEventBuffer_postDoneSem(buffer, -1);

    return 0;

}

void MKIDShmEventBuffer_postDoneSem(MKID_EVENT_BUFFER *buffer, int semInd){
    int i;
    if(semInd==-1)
        for(i=0; i<N_DONE_SEMS; i++)
            sem_post(buffer->newPhotonSemList[i]);
    else
        sem_post(buffer->newPhotonSemList[semInd]);

}

void MKIDShmEventBuffer_reset(MKID_EVENT_BUFFER *eventBuffer){
    int i;
    eventBuffer->md->endInd = -1;
    eventBuffer->md->nCycles = 0;

    for(i=0; i<N_DONE_SEMS; i++)
        while(sem_trywait(eventBuffer->newPhotonSemList[i])==0);

}

