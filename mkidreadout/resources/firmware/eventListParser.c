#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <semaphore.h>

#include <mkidshm.h>

// gcc eventListParser.c -o eventListParser -I../../readout/mkidshm -L../../readout/mkidshm -Wl,-rpath=../../readout/mkidshm/ -lmkidshm -lpthread -lrt

int main(){
    MKID_EVENT_BUFFER eventBuffer;
    MKID_PHOTON_EVENT photon;
    int photonInd=0;

    MKIDShmEventBuffer_open(&eventBuffer, "/EventBufferTest0");

    while(1){
        sem_wait(eventBuffer.newPhotonSemList[0]);
        //photonInd = eventBuffer.md->endInd;
        photonInd += 1;
        photonInd = photonInd%eventBuffer.md->bufferSize;
        printf("Photon #%d\n", photonInd);
        photon = eventBuffer.eventBuffer[photonInd];
        printf("    x: %d\n", photon.x);
        printf("    y: %d\n", photon.y);
        printf("   ts: %lu\n", photon.time);
        printf("  wvl: %f\n\n", photon.wvl);


    }


}
