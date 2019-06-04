#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <semaphore.h>

#include <mkidshm.h>

// gcc eventListParser.c -o eventListParser -lmkidshm -lpthread -lrt

int main(){
    MKID_EVENT_BUFFER eventBuffer;
    MKID_PHOTON_EVENT photon;

    MKIDShmEventBuffer_open(&eventBuffer, "/EventBufferTest0");

    while(1){
        sem_wait(eventBuffer.newPhotonSemList[0]);
        printf("Photon #%d\n", eventBuffer.md->endInd);
        photon = eventBuffer.eventBuffer[eventBuffer.md->endInd];
        printf("    x: %d\n", photon.x);
        printf("    y: %d\n", photon.y);
        printf("   ts: %lu\n", photon.time);
        printf("  wvl: %f\n\n", photon.wvl);


    }


}
