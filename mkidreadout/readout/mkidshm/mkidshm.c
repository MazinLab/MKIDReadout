#include "mkidshm.h"

void *openShmFile(const char *shmName, size_t size, int create){
    char name[80];
    char error[200];
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


    return shmPtr;

}
    

int MKIDShmImage_create(MKID_IMAGE_METADATA *imageMetadata, const char *imgName, MKID_IMAGE *outputImage){
    MKID_IMAGE_METADATA *mdPtr;
    char doneSemName[80];
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
        snprintf(doneSemName, 80, "%s%d", mdPtr->doneImageSemName, i);
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
    snprintf(imageMetadata->name, 80, "%s", name);
    snprintf(imageMetadata->wavecalID, 150, "%s", "none");
    snprintf(imageMetadata->imageBufferName, 80, "%s.buf", name);
    snprintf(imageMetadata->takeImageSemName, 80, "%s.takeImg", name);
    snprintf(imageMetadata->doneImageSemName, 80, "%s.doneImg", name);
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
    //char error[200];

    time += TIMEDWAIT_FUDGE;
    clock_gettime(CLOCK_REALTIME, &tspec);
    tspec.tv_sec += time/2000;
    tspec.tv_nsec += (time%2000)*500000;
    tspec.tv_sec += tspec.tv_nsec/1000000000;
    tspec.tv_nsec = tspec.tv_nsec%1000000000;

    retval = sem_timedwait(image->doneImageSemList[semInd], &tspec);

    //if(retval == -1){
    //    snprintf(error, 200, "Timedwait error %ld", tspec.tv_nsec);
    //    perror(error);

    //}

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

