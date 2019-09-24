#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <semaphore.h>

#include <mkidshm.h>

// gcc eventListParser.c -o eventListParser -I../../readout/mkidshm -L../../readout/mkidshm -Wl,-rpath=../../readout/mkidshm/ -lmkidshm -lpthread -lrt

int main(){
    MKID_EVENT_BUFFER eventBuffer;
    MKID_PHOTON_EVENT photon;
    int photonInd = 0;
    int firstIter = 1;
    int nCycles;

    MKIDShmEventBuffer_open(&eventBuffer, "/EventBufferTest0");
    printf("endInd: %d\n", eventBuffer.md->endInd);
    printf("nCycles: %d\n", eventBuffer.md->nCycles);
    MKIDShmEventBuffer_reset(&eventBuffer);
    //FILE *eventFile = fopen("eventList.txt", "w");

    while(1){
        sem_wait(eventBuffer.newPhotonSemList[0]);
        if(firstIter){
            photonInd = eventBuffer.md->endInd;
            nCycles = eventBuffer.md->nCycles;
            printf("first event\n");
            printf("endInd: %d\n", photonInd);
            printf("nCycles: %d\n", nCycles);
            firstIter = 0;

        }
            
        photonInd = eventBuffer.md->endInd;
        //if(photonInd % 1 ==0){
        //    fprintf(eventFile, ".");
        //    fflush(stdout);

        //}

        //photonInd += 1;
        //photonInd = photonInd%eventBuffer.md->size;

        //if(photonInd % 100 == 0){
        //    fprintf(eventFile, "\nPhoton #%d\n", photonInd);
        //    photon = eventBuffer.buffer[photonInd];
        //    fprintf(eventFile, "    x: %d\n", photon.x);
        //    fprintf(eventFile, "    y: %d\n", photon.y);
        //    fprintf(eventFile, "   ts: %lu\n", photon.time);
        //    fprintf(eventFile, "  wvl: %f\n\n", photon.wvl);

        //}


    }

    //fclose(eventFile);


}
