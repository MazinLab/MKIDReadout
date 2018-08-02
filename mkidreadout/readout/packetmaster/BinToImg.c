
// read in a .bin file and run cuber on it
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
#include <byteswap.h>

#define _POSIX_C_SOURCE 200809L

#define BUFLEN 1500
#define PORT 50000
#define XPIX 80
#define YPIX 125
#define NROACH 10
//#define LOGPATH "/mnt/data0/logs/"

// compile with gcc -o BinToImg BinToImg.c -I. -lm -lrt

/*
struct datapacket {
    unsigned int xcoord:10;
    unsigned int ycoord:10;
    unsigned int timestamp:9;
    unsigned int wvl:18;
    unsigned int baseline:17;
}__attribute__((packed));;
*/

struct datapacket {
    unsigned int baseline:17;
    unsigned int wvl:18;
    unsigned int timestamp:9;
    unsigned int ycoord:10;
    unsigned int xcoord:10;
}__attribute__((packed));;

/*
struct hdrpacket {
    unsigned int start:8;
    unsigned int roach:8;
    unsigned int frame:12;
    unsigned long timestamp:36;
}__attribute__((packed));;
*/

struct hdrpacket {
    unsigned long timestamp:36;
    unsigned int frame:12;
    unsigned int roach:8;
    unsigned int start:8;
}__attribute__((packed));;

void diep(char *s)
{
  printf("errono: %d",errno);
  perror(s);
  exit(1);
}

int need_to_stop() //Checks for a stop file and returns true if found, else returns 0
{
    char stopfilename[] = "stop.bin";
    FILE* stopfile;
    stopfile = fopen(stopfilename,"r");
    if (stopfile == 0) //Don't stop
    {
        errno = 0;
        return 0;
    }
    else //Stop file exists, stop
    {
        printf("found stop file. Exiting\n");
        return 1;
    }
}

void ParsePacket( uint16_t image[XPIX][YPIX], char *packet, unsigned int l, uint64_t frame[NROACH])
{
    unsigned int i;
    struct hdrpacket *hdr;
    struct datapacket *data;
    uint64_t starttime;
    uint16_t curframe;
    char curroach;
    uint64_t swp,swp1;

    // pull out header information from the first packet
    swp = *((uint64_t *) (&packet[0]));
    swp1 = __bswap_64(swp);
    hdr = (struct hdrpacket *) (&swp1);             

    starttime = hdr->timestamp;
    curframe = hdr->frame;
    curroach = hdr->roach;
        
    // check for missed frames and print an error if we got the wrong frame number
    if( frame[curroach] != curframe) {
        printf("Roach %d: Expected Frame %d, Received Frame %d\n",curroach,frame[curroach],curframe); fflush(stdout);
        frame[curroach] = (frame[curroach]+1)%4096;
    }
    else {
        frame[curroach] = (frame[curroach]+1)%4096;
    }

    for(i=1;i<l/8;i++) {
       
       swp = *((uint64_t *) (&packet[i*8]));
       swp1 = __bswap_64(swp);
       data = (struct datapacket *) (&swp1);
       image[(data->xcoord)%XPIX][(data->ycoord)%YPIX]++;
       
       // debug
       //if( (data->xcoord)%XPIX > 50 ) {
       //printf("x,y = %d,%d \t %lx\n",(data->xcoord),(data->ycoord), *((uint64_t *) (&packet[i*8])) );           
       //}
    }

    //printf("%d %d %d %d %d %d %d %d %d %d - roach %d frame %d\n",frame[0],frame[1],frame[2],frame[3],frame[4],frame[5],frame[6],frame[7],frame[8],frame[9],curroach,curframe); fflush(stdout);

}

void Cuber()
{
    int br,i,j,cwr;
    char data[1024];
    char olddata[1048576];
    char packet[808*2];
    time_t s,olds;  // Seconds
    struct timespec spec;
    uint16_t image[XPIX][YPIX];
    FILE *wp;
    char outfile[160];
    unsigned int oldbr = 0;     // number of bytes of unparsed data sitting in olddata
    uint64_t frame[NROACH];
    uint64_t pcount = 0;
    struct hdrpacket *hdr;
    struct hdrpacket hdrv;
    char temp[8];
    uint64_t swp,swp1;
    char cmd[120];    
    
    printf("Fear the wrath of CUBER!\n");
    printf(" Cuber: My PID is %d\n", getpid());
    printf(" Cuber: My parent's PID is %d\n", getppid()); fflush(stdout);
    
    while( (cwr = open("/mnt/ramdisk/CuberPipe.pip", O_RDONLY | O_NDELAY)) == -1);
    printf("cwr = %d\n",cwr);
    
    //cwr = open("/mnt/ramdisk/CuberPipe.pip", O_RDONLY );    
    
    //printf("struct sizes = %d, %d\n",sizeof(struct hdrpacket),sizeof(struct datapacket));

    memset(image, 0, sizeof(image[0][0]) * XPIX * YPIX);    // zero out array
    memset(olddata, 0, sizeof(olddata[0])*2048);    // zero out array
    memset(data, 0, sizeof(data[0]) * 1024);    // zero out array
    memset(packet, 0, sizeof(packet[0]) * 808 * 2);    // zero out array
    memset(frame,0,sizeof(frame[0])*NROACH);

    clock_gettime(CLOCK_REALTIME, &spec);   
    olds  = spec.tv_sec;

    while (access( "/mnt/ramdisk/QUIT", F_OK ) == -1)
    {
       // if it is a new second, zero the image array and start over
       clock_gettime(CLOCK_REALTIME, &spec);   
       s  = spec.tv_sec;
       if( s > olds ) {                 
          // we are in a new second, so write out image array and then zero out the array
          //printf("CUBER: Finised second %d.",olds);  fflush(stdout);
          
          sprintf(outfile,"/mnt/ramdisk/%d.img",olds);
          wp = fopen(outfile,"wb");
          fwrite(image, sizeof(image[0][0]), XPIX * YPIX, wp);
          fclose(wp);

          olds = s;
          memset(image, 0, sizeof(image[0][0]) * XPIX * YPIX);    // zero out array
          printf("CUBER: Parse rate = %d pkts/sec.  Data in buffer = %d\n",pcount,oldbr); fflush(stdout);
          pcount=0;
          
          // spawn Bin2PNG to make png file
          sprintf(cmd,"/mnt/data0/PacketMaster2/Bin2PNG %s /mnt/ramdisk/%d.png &",outfile,olds);
          system(cmd);
       }
       
       // not a new second, so read in new data and parse it          
	   br = read(cwr, data, 1024*sizeof(char));
	    
       if( br != -1) {
          //if( br > 0 ) printf("br = %d | oldbr = %d\n",br,oldbr);fflush(stdout);
                 
          // we may be in the middle of a packet so put together a full packet from old data and new data
          // and only parse when the packet is complete

          // append current data to existing data if old data is present
          // NOTE !!!  olddata must start with a packet header
          if( oldbr > 0 ) {
             memmove(&olddata[oldbr],data,br);
             oldbr+=br;
          } 
          else {
             memcpy(olddata,data,br);
             oldbr=br;          
          }
       }   
       
       // if there is data waiting, process it
       if( oldbr > 0 ) {       
          // search the available data for a packet boundary
          for( i=1; i<oldbr/8; i++) {
             
             swp = *((uint64_t *) (&olddata[i*8]));
             swp1 = __bswap_64(swp);
             hdr = (struct hdrpacket *) (&swp1);             
             
             /*temp[0] = olddata[i*8 + 7];
             temp[1] = olddata[i*8 + 6];
             temp[2] = olddata[i*8 + 5];                                       
             temp[3] = olddata[i*8 + 4];             
             temp[4] = olddata[i*8 + 3];
             temp[5] = olddata[i*8 + 2];
             temp[6] = olddata[i*8 + 1];                                       
             temp[7] = olddata[i*8 ];  */
             
             //hdrv = __bswap_64(*((struct hdrpacket *) &olddata[i*8]));
             //hdr = (struct hdrpacket *) temp;
                          
             //printf("%d-%d\t",i,hdr->start); fflush(stdout);
             if (hdr->start == 0b11111111) {        // found new packet header!
                printf("Found new packet header at %d.  roach=%d, frame=%d\n",i,hdr->roach,hdr->frame);
       
                if( i*8 > 104*8 ) { 
                   printf("Error - packet too long: %d\n",i);
                   //printf("br = %d : oldbr = %d : i = %d\n",br,oldbr,i);
                   //for(j=0;j<oldbr/8;j++) printf("%d : 0x%X\n",j,(uint64_t) olddata[j*8]);                
                   fflush(stdout);
                }
                
                //printf("Packet ends at %d\n",i*8);

                // fill packet for parsing
                memmove(packet,olddata,i*8);
                   
                // copy any extra data to olddata
                memmove(olddata,&olddata[i*8],oldbr-i*8);
                oldbr = oldbr-i*8;
                   
                // parse it!
                pcount++;
                ParsePacket(image,packet,i*8,frame);
                break;  // abort loop and start over!                
             } 
             else if (hdr->start == 0b01111111 && hdr->roach == 0b11111111 )  {  // short packet detected
                // fill packet for parsing
                printf("Partial Packet %d\n",i);fflush(stdout);
                
                memmove(packet,olddata,i*8);
                   
                // copy any extra data to olddata, removing fake photon
                memmove(olddata,&olddata[(i+1)*8],oldbr-(i+1)*8);
                oldbr = oldbr-(i+1)*8;
                   
                // parse it!
                pcount++;
                ParsePacket(image,packet,i*8,frame);
                break;  // abort loop and start over!                
                
             }           
             
             // need to check for short/EOF packet, which is one zero and 63 ones
             /*
             if( olddata[i*8] == 0b01111111 && olddata[i*8+1] == 0b11111111 ) {
                // detected short packet!  Copy and parse!
                print("Detected short packet!\n");
                memmove(packet,olddata,i*8);
                   
                // copy any extra data to olddata, but don't copy EOF packet!
                memmove(olddata,&olddata[i*8+8],br-oldbr-8);
                oldbr = br - oldbr - 8;
                ParsePacket(image,packet,br-oldbr,frame);             
             } 
             */           
          }
          
                   
       }                
    }

    printf("CUBER: Closing\n");
    close(cwr);
    return;
}

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
    uint64_t swp, swp1,count=0;   

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
        count++;

        swp1 = __bswap_64(d1);
        hdr = (struct hdrpacket *) (&swp1);   
        
        if (hdr->start == 0b11111111) {        // found new packet header!
                //printf("Found new packet header at %d. time = %d\tframe=%d\t 0x%lx\n",count,hdr->timestamp,hdr->roach,hdr->frame,*((uint64_t *) &hdr));
                //printf("%lx\n",*((uint64_t *) (&swp1)));
                                
                // read photons
                //fread(&d1,sizeof(d1),1,rp);
         
                //swp1 = d1;
                swp1 = __bswap_64(d1);
                data = (struct datapacket *) (&swp1);   
                printf("x,y = %d,%d \t %lx\n",(data->xcoord),(data->ycoord), *((uint64_t *) (&swp1)) );            
        }            
        
        
    }
    
    printf("Received %d header packets.\n",hnum);
    printf("Received %d photon data packets.\n",pnum);
    for(i=0;i<NROACH;i++) printf("ROACH %d -> %d photons\n",i,nphot[i]);
    
        
    fclose(rp);
    
}
