#include "mkidshm.h"

void *openShmFile(char *shmName, size_t size, int create){
    char name[80];
    int fd;
    void *shmPtr;
    int flag;

    if(create==1)
        flag = O_RDWR|O_CREAT;
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
    

int createMKIDShmImage(MKID_IMAGE_METADATA *imageMetadata, char *imgName, MKID_IMAGE *outputImage){
    MKID_IMAGE_METADATA *mdPtr;
    image_t *imgPtr;

    mdPtr = (MKID_IMAGE_METADATA*)openShmFile(imgName, sizeof(MKID_IMAGE_METADATA), 1);

    if(mdPtr==NULL)
        return -1;

    memcpy(mdPtr, imageMetadata, sizeof(MKID_IMAGE_METADATA)); //copy contents of imageMetadata into shared memory buffer
    outputImage->md = mdPtr;

    // CREATE IMAGE DATA BUFFER
    int imageSize = (mdPtr->nXPix)*(mdPtr->nYPix)*(mdPtr->nWvlBins);

    imgPtr = (image_t*)openShmFile(mdPtr->imageBufferName, sizeof(image_t)*imageSize, 1);
    if(imgPtr==NULL)
        return -1;

    outputImage->image = imgPtr;

    // OPEN SEMAPHORES
    outputImage->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    outputImage->doneImageSem = sem_open(mdPtr->doneImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    if((outputImage->takeImageSem==SEM_FAILED)||(outputImage->doneImageSem==SEM_FAILED)) 
        printf("Semaphore creation failed %s\n", strerror(errno));

    return 0;
    

}
    

int openMKIDShmImage(MKID_IMAGE *imageStruct, char *imgName){
    MKID_IMAGE_METADATA *mdPtr;
    image_t *imgPtr;

    // OPEN METADATA BUFFER
    mdPtr = (MKID_IMAGE_METADATA*)openShmFile(imgName, sizeof(MKID_IMAGE_METADATA), 0);
    if(mdPtr == NULL)
        return -1;

    imageStruct->md = mdPtr;

    // OPEN IMAGE BUFFER 
    int imageSize = (mdPtr->nXPix)*(mdPtr->nYPix)*(mdPtr->nWvlBins);
    imgPtr = (image_t*)openShmFile(imageStruct->md->imageBufferName, imageSize*sizeof(image_t), 0);
    if(imgPtr == NULL)
        return -1;
 
    imageStruct->image = imgPtr;

    // OPEN SEMAPHORES
    imageStruct->takeImageSem = sem_open(mdPtr->takeImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    imageStruct->doneImageSem = sem_open(mdPtr->doneImageSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    if((imageStruct->takeImageSem==SEM_FAILED)||(imageStruct->doneImageSem==SEM_FAILED)) 
        printf("Semaphore creation failed %s\n", strerror(errno));

    return 0;

}

int closeMKIDShmImage(MKID_IMAGE *imageStruct){
    sem_close(imageStruct->takeImageSem);
    sem_close(imageStruct->doneImageSem);
    munmap(imageStruct->image, sizeof(image_t)*(imageStruct->md->nXPix)*(imageStruct->md->nYPix)*(imageStruct->md->nWvlBins));
    munmap(imageStruct->md, sizeof(MKID_IMAGE_METADATA));
    return 0;

}

int populateImageMD(MKID_IMAGE_METADATA *imageMetadata, char *name, int nXPix, int nYPix, int useWvl, int nWvlBins, int wvlStart, int wvlStop){
    imageMetadata->nXPix = nXPix;
    imageMetadata->nYPix = nYPix;
    imageMetadata->useWvl = useWvl;
    imageMetadata->nWvlBins = nWvlBins;
    imageMetadata->wvlStart = wvlStart;
    imageMetadata->wvlStop = wvlStop;
    imageMetadata->startTime = 0;
    imageMetadata->integrationTime = 0;
    snprintf(imageMetadata->imageBufferName, 80, "%s.buf", name);
    snprintf(imageMetadata->takeImageSemName, 80, "%s.takeImg", name);
    snprintf(imageMetadata->doneImageSemName, 80, "%s.doneImg", name);

}

void startIntegration(MKID_IMAGE *image, uint64_t startTime){
    sem_post(image->takeImageSem);}

//Blocking
void waitForImage(MKID_IMAGE *image){
    sem_wait(image->takeImageSem);}

//Non-blocking
int checkDoneImage(MKID_IMAGE *image){
    return sem_trywait(image->doneImageSem);}

