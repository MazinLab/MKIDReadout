// BinCheck.c 
// check the .bin files written out by PacketMaster2
//    Ben Mazin, 7/2/16

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdint.h>
#include <sys/time.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <math.h>

#define _POSIX_C_SOURCE 200809L

#define BUFLEN 1500
#define PORT 50000
#define XPIX 80
#define YPIX 125
#define NROACH 10

// compile with gcc -o BinCheck BinCheck.c -I. -lm -lrt

struct datapacket {
    unsigned int xcoord:10;
    unsigned int ycoord:10;
    unsigned int timestamp:9;
    unsigned int wvl:18;
    unsigned int baseline:17;
}__attribute__((packed));;

struct hdrpacket {
    unsigned int start:8;
    unsigned int roach:8;
    unsigned int frame:12;
    unsigned long timestamp:36;
}__attribute__((packed));;

int main(int argc, char *argv[])
{	
    FILE *rp;
    uint64_t d1,curroach,curframe,curtime,timestamp,pnum,hnum;
    struct hdrpacket *hdr;
    struct datapacket *data;
    uint64_t frame[NROACH],nphot[NROACH];
    double arrivaltime[NROACH];
    long i;
    double photontime, curphotontime;    

    // Make sure that the output filename argument has been provided
	if (argc != 2) {
		fprintf(stderr, "Please specify input .bin file to check!\n");
		return 1;
	}
	
	printf("Loading %s\n",argv[1]);
    rp = fopen(argv[1],"rb");

    pnum = 0;
    curroach = 0;
    curframe = 0;
    timestamp = 0;
    hnum = 0;
    memset(frame,0,sizeof(frame[0])*NROACH);
    memset(nphot,0,sizeof(nphot[0])*NROACH);
    memset(arrivaltime,0.0,sizeof(arrivaltime[0])*NROACH);
        
    while( !feof(rp) ) {
        fread(&d1,sizeof(d1),1,rp);
        hdr = (struct hdrpacket *) &d1;
        
        // see if the packet we read in was a header packet
        if (hdr->start == 0b11111111) {         // found new packet header!
            curroach = hdr->roach;
            curframe = hdr->frame;
            timestamp = hdr->timestamp;
            photontime = ((double)timestamp)/2000.0 + 1451606400.0;
            
            // check if frame arrived in sequence
            if( frame[curroach] != curframe ) {
                if( frame[curroach] != 0 ) printf("Hdr packets %d: Nphot %d: Roach %d: Expected Frame %d, Received Frame %d\n",hnum,pnum,curroach,frame[curroach],curframe); fflush(stdout); // don't print error until we know what frame we are at
                frame[curroach] = curframe+1;
            }
            else {
                frame[curroach] = (frame[curroach]+1)%4096;
            }
                       
            hnum++; 
        }
        else {                                  // must be a data packet!
            // if we haven't seen a header packet yet ignore this photon 
            if( hnum == 0 ) continue;            
            // if it is a fake photon skip it            
            if (hdr->start == 0b01111111 && hdr->roach == 0b11111111 ) continue;
            
            data = (struct datapacket *) &d1;
            curphotontime = photontime + (1e-6*((double)data->timestamp));
            //printf(" - (%d,%d,%d) Timestamp = %f, Last Photon = %f\n",hnum,pnum,data->timestamp,curphotontime,arrivaltime[curroach]);           
            
            if( curphotontime <= arrivaltime[curroach] ) {
                printf("Photon out of time order! (%d,%d) Timestamp = %f, Last Photon = %f\n",hnum,pnum,curphotontime,arrivaltime[curroach]);
            }
            arrivaltime[curroach] = curphotontime;
            
            //printf("Timestamp = %f, Last Packet = %f\n",photontime,curphotontime);
            
            // construct photon arrival time and see if it is out of order
            
            
            nphot[curroach]++;
            pnum++;            
            
            if(nphot[0] > 20) break;
        }
        
    }
    
    printf("Received %d header packets.\n",hnum);
    printf("Received %d photon data packets.\n",pnum);
    for(i=0;i<NROACH;i++) printf("ROACH %d -> %d photons\n",i,nphot[i]);
    
        
    fclose(rp);

}




