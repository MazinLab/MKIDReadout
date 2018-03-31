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
#include <sys/mman.h>

#define _POSIX_C_SOURCE 200809L

#define BUFLEN 1500
#define PORT 50000
#define XPIX 140
#define YPIX 145
#define NROACH 20
#define SHAREDBUF 536870912

#define TSOFFS 1514764800 

//#define LOGPATH "/mnt/data0/logs/"

// compile with gcc -o PacketMaster5 PacketMaster5.c -I. -lm -lrt

struct datapacket {
    unsigned int baseline:17;
    unsigned int wvl:18;
    unsigned int timestamp:9;
    unsigned int ycoord:10;
    unsigned int xcoord:10;
}__attribute__((packed));;

struct hdrpacket {
    unsigned long timestamp:36;
    unsigned int frame:12;
    unsigned int roach:8;
    unsigned int start:8;
}__attribute__((packed));;

struct readoutstream {
    uint64_t unread;
    char busy;               // set to 1 to disallow reads, 0 to allow reads
    char data[SHAREDBUF];
};

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

// open a shared memory space named buf
struct readoutstream *OpenShared(char buf[40])
{
    int fd;    
    struct readoutstream *rptr;
    
    // Create shared memory for photon data
    fd = shm_open(buf, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) { 
        perror("shm_open");  /* something went wrong */
        exit(1);             
    }
    
    if (ftruncate(fd, sizeof(struct readoutstream)) == -1) { 
        perror("ftruncate");  /* something went wrong */
        exit(1);               
    }
  
    rptr = mmap(NULL, sizeof(struct readoutstream), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (rptr == MAP_FAILED) { 
        perror("mmap");  /* something went wrong */
        exit(1);            
    }
    
    close(fd);
    return(rptr);    
}

void ParsePacket( uint16_t image[XPIX][YPIX], char *packet, unsigned int l, uint64_t frame[NROACH])
{
    uint64_t i;
    //struct hdrpacket *hdr;
    struct datapacket *data;
    //uint64_t starttime;
    //uint16_t curframe;
    //char curroach;
    uint64_t swp,swp1;

    // pull out header information from the first packet
//    swp = *((uint64_t *) (&packet[0]));
//    swp1 = __bswap_64(swp);
//    hdr = (struct hdrpacket *) (&swp1);             

//    starttime = hdr->timestamp;
//    curframe = hdr->frame;
//    curroach = hdr->roach;
        
    // check for missed frames and print an error if we got the wrong frame number
/*
    if( frame[curroach] != curframe) {
        printf("Roach %d: Expected Frame %d, Received Frame %d\n",curroach,frame[curroach],curframe); fflush(stdout);
        frame[curroach] = (frame[curroach]+1)%4096;
    }
    else {
        frame[curroach] = (frame[curroach]+1)%4096;
    }
*/

    for(i=1;i<l/8;i++) {
       
       swp = *((uint64_t *) (&packet[i*8]));
       swp1 = __bswap_64(swp);
       data = (struct datapacket *) (&swp1);
       //image[(data->xcoord)%XPIX][(data->ycoord)%YPIX]++;
       
       if( data->xcoord >= XPIX || data->ycoord >= YPIX ) continue;
       image[data->xcoord][data->ycoord]++;
      
    }

}

void Cuber()
{
    int64_t br,i,j;
    char data[1024];
    char *olddata;
    char packet[808*16];
    time_t s,olds;  // Seconds
    struct timespec spec;
    uint16_t image[XPIX][YPIX];
    FILE *wp;
    char outfile[160];
    uint64_t oldbr = 0;     // number of bytes of unparsed data sitting in olddata
    uint64_t frame[NROACH];
    uint64_t pcount = 0;
    struct hdrpacket *hdr;
    struct hdrpacket hdrv;
    char temp[8];
    uint64_t swp,swp1,newstart;
    char cmd[120];   
    char errflag = 0; 
    struct readoutstream *rptr;
    uint64_t pstart;
    
    //For timing test
    struct timeval tv;
    unsigned long long sysTs;
    uint64_t roachTs;
    

    printf("Fear the wrath of CUBER!\n");
    printf(" Cuber: My PID is %d\n", getpid());
    printf(" Cuber: My parent's PID is %d\n", getppid()); fflush(stdout);
    
    // open shared memory block 2 for photon data
    rptr = OpenShared("/roachstream2");    
    olddata = (char *) malloc(sizeof(char)*SHAREDBUF);
    
    memset(image, 0, sizeof(image[0][0]) * XPIX * YPIX);    // zero out array
    memset(olddata, 0, sizeof(olddata[0])*2048);    // zero out array
    memset(data, 0, sizeof(data[0]) * 1024);    // zero out array
    memset(packet, 0, sizeof(packet[0]) * 808 * 2);    // zero out array
    memset(frame,0,sizeof(frame[0])*NROACH);

    clock_gettime(CLOCK_REALTIME, &spec);   
    olds  = spec.tv_sec;

    FILE *timeFile = fopen("timetestPk5.txt", "w");

    while (access( "/home/ramdisk/QUIT", F_OK ) == -1)
    {
       // if it is a new second, zero the image array and start over
       clock_gettime(CLOCK_REALTIME, &spec);   
       s  = spec.tv_sec;
       if( s > olds ) {                 
          // we are in a new second, so write out image array and then zero out the array
          sprintf(outfile,"/home/ramdisk/%d.img",olds);
          wp = fopen(outfile,"wb");
          fwrite(image, sizeof(image[0][0]), XPIX * YPIX, wp);
          fclose(wp);

          olds = s;
          memset(image, 0, sizeof(image[0][0]) * XPIX * YPIX);    // zero out array
          printf("CUBER: Parse rate = %d pkts/sec. Data in buffer = %d\n",pcount,oldbr); fflush(stdout);
          pcount=0;
       }
       
       // not a new second, so read in new data and parse it    
       while( rptr->busy == 1 ) continue; 
       rptr->busy=1;     
       br = rptr->unread;
       if( br%8 != 0 ) printf("Misalign in Cuber - %d\n",br); 
	   	    
       if( br > 0) {               
          // we may be in the middle of a packet so put together a full packet from old data and new data
          // and only parse when the packet is complete

          // append current data to existing data if old data is present
          // NOTE !!!  olddata must start with a packet header
          if( oldbr > 0 ) {
             memmove(&olddata[oldbr],rptr->data,br);
             oldbr+=br;
             rptr->unread = 0;
             if (oldbr > SHAREDBUF-2000) {
                printf("oldbr = %d!  Dumping data!\n",oldbr); fflush(stdout);
                br = 0;
                oldbr=0;          
             }
          } 
          else {
             memmove(olddata,rptr->data,br);
             oldbr=br;          
             rptr->unread = 0;     
          }
       }
       rptr->busy=0;   

       // sanity check that the first packet in oldbr is a header packet
       /*
       if(oldbr > 0) {
           swp = *((uint64_t *) (&olddata[0]));
           swp1 = __bswap_64(swp);
           hdr = (struct hdrpacket *) (&swp1);
           if (hdr->start != 0b11111111) {
               printf("Oldbr (%d) start packet not a header! roach=%d frame=%d ts=%ld\n",oldbr,hdr->roach,hdr->frame,hdr->timestamp);  fflush(stdout);
           } 
       } */          

       // if there is data waiting, process it
       pstart = 0;
       if( oldbr >= 808*10 ) {       
          // search the available data for a packet boundary
          //printf("Start Parse\n"); fflush(stdout);
          for( i=1; i<oldbr/8; i++) {
             
             swp = *((uint64_t *) (&olddata[i*8]));
             swp1 = __bswap_64(swp);
             hdr = (struct hdrpacket *) (&swp1);             
                                      
             if (hdr->start == 0b11111111) {        // found new packet header!
                // fill packet and parse
                // printf("Found Header at %d\n",i*8); fflush(stdout);
                memmove(packet,&olddata[pstart],i*8 - pstart);
                pcount++;                
                roachTs = (uint64_t)hdr->timestamp;
                gettimeofday(&tv, NULL);
                sysTs = (unsigned long long)(tv.tv_sec)*1000 + (unsigned long long)(tv.tv_usec)/1000 - (unsigned long long)TSOFFS*1000;
                sysTs = sysTs*2;
                fprintf(timeFile, "%llu %llu\n", roachTs, sysTs);
                ParsePacket(image,packet,i*8 - pstart,frame); 
		pstart = i*8;   // move start location for next packet	                      
             }
          }

	  // if there is data remaining save it for next run through
          //printf("Copying excess %d, %d\n",oldbr,pstart); fflush(stdout);
          memmove(olddata,&olddata[pstart],oldbr-pstart);
          oldbr = oldbr-pstart;          
       }                           
    }

    printf("CUBER: Closing\n");
    free(olddata);
    return;
}

void Writer()
{
    long            ms; // Milliseconds
    time_t          s,olds;  // Seconds
    struct timespec spec;
    long dat, outcount;
    int mode=0;
    FILE *wp, *rp;
    char data[1024];
    char path[80];
    char fname[120];
    int br;
    struct readoutstream *rptr;

    printf("Rev up the RAID array,WRITER is active!\n");
    printf(" Writer: My PID is %d\n", getpid());
    printf(" Writer: My parent's PID is %d\n", getppid());

    // open shared memory block 1 for photon data
    rptr = OpenShared("/roachstream1");
    
    //  Write looks for a file on /home/ramdisk named "START" which contains the write path.  
    //  If this file is present, enter writing mode
    //  and delete the file.  Keep writing until the file "STOP" file appears.
    //  Shut down if a "QUIT" file appears 

    // mode = 0 :  Not doing anything, just keep pipe clear and junk and data that comes down it
    // mode = 1 :  "START" file detected enter write mode
    // mode = 2 :  continous writing mode, watch for "STOP" or "QUIT" files
    // mode = 3 :  QUIT file detected, exit

    while (mode != 3) {

       // keep the shared mem clean!       
       if( mode == 0 ) {
          while( rptr->busy == 1 ) continue;
	      rptr->busy = 1;
	      rptr->unread = 0;
	      rptr->busy = 0;
       }

       if( mode == 0 && access( "/home/ramdisk/START", F_OK ) != -1 ) {
          // start file exists, go to mode 1
           mode = 1;
           printf("Mode 0->1\n");
       } 

       if( mode == 1 ) {
          // read path from start, generate filename, and open file pointer for writing
          rp = fopen("/home/ramdisk/START","r");
          fscanf(rp,"%s",path);
          fclose(rp);
          remove("/home/ramdisk/START");

          clock_gettime(CLOCK_REALTIME, &spec);   
          s  = spec.tv_sec;
          olds = s;
          sprintf(fname,"%s%d.bin",path,s);
          printf("Writing to %s\n",fname);
          wp = fopen(fname,"wb");
          mode = 2;
          outcount = 0;
          printf("Mode 1->2\n");
       }

       if( mode == 2 ) {
          if ( access( "/home/ramdisk/STOP", F_OK ) != -1 ) {
             // stop file exists, finish up and go to mode 0
	         fclose(wp);
             remove("/home/ramdisk/STOP");
             mode = 0;
             printf("Mode 2->0\n");
          } else {
             // continuous write mode, store data from the pipe to disk

	         // start a new file every 1 seconds
             clock_gettime(CLOCK_REALTIME, &spec);   
             s  = spec.tv_sec;

             if( s - olds >= 1 ) {
                 fclose(wp);
                 sprintf(fname,"%s%d.bin",path,s);
                 printf("WRITER: Writing to %s, rate = %ld MBytes/sec\n",fname,outcount/1000000);
                 wp = fopen(fname,"wb");
                 olds = s;
                 outcount = 0;               
             }

	         // write all data in shared memory to disk
	         if( rptr->unread > 0 ) {
	            while( rptr->busy == 1 ) continue;
	            rptr->busy = 1;
	            fwrite( rptr->data, 1, rptr->unread, wp);    // could probably speed this up by copying data to new array and doing the fwrite after setting busy to 0
	            outcount += rptr->unread; 
	            rptr->unread = 0;
	            rptr->busy = 0;
	         }
          }
       }

       // check for quit flag and then bug out if received! 
       if( access( "/home/ramdisk/QUIT", F_OK ) != -1 ) {
	      fclose(wp);
          remove("/home/ramdisk/START");
          remove("/home/ramdisk/STOP");
          remove("/home/ramdisk/QUIT");
          mode = 3;
          printf("Mode 3\n");
       }

    }

/*
    clock_gettime(CLOCK_REALTIME, &spec);

    s  = spec.tv_sec;
    ms = lround(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds

    printf("Current time: %"PRIdMAX".%03ld seconds since the Epoch\n", (intmax_t)s, ms);

    while(exitflag == 0) {
       read(pfds[0], &dat, sizeof(long));
       printf("%ld\n",dat);	
    }
*/
    printf("WRITER: Closing\n");
    return;
}


void Reader()
{
  //set up a socket connection
  struct sockaddr_in si_me, si_other;
  int s, i, slen=sizeof(si_other);
  unsigned char buf[BUFLEN];
  ssize_t nBytesReceived = 0;
  ssize_t nTotalBytes = 0;
  int n1,n2;
  struct readoutstream *rptr1, *rptr2;

  printf("READER: Connecting to Socket!\n"); fflush(stdout);

  // Create shared memory for photon data
  rptr1 = OpenShared("/roachstream1");
  rptr2 = OpenShared("/roachstream2");

  if ((s=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP))==-1)
    diep("socket");
  printf("READER: socket created\n");
  fflush(stdout);

  memset((char *) &si_me, 0, sizeof(si_me));
  si_me.sin_family = AF_INET;
  si_me.sin_port = htons(PORT);
  si_me.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(s, (const struct sockaddr *)(&si_me), sizeof(si_me))==-1)
      diep("bind");
  printf("READER: socket bind\n");
  fflush(stdout);

  //Set receive buffer size, the default is too small.  
  //If the system will not allow this size buffer, you will need
  //to use sysctl to change the max buffer size
  int retval = 0;
  int bufferSize = 33554432;
  retval = setsockopt(s, SOL_SOCKET, SO_RCVBUF, &bufferSize, sizeof(bufferSize));
  if (retval == -1)
    diep("set receive buffer size");

  //Set recv to timeout after 3 secs
  const struct timeval sock_timeout={.tv_sec=3, .tv_usec=0};
  retval = setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (char*)&sock_timeout, sizeof(sock_timeout));
  if (retval == -1)
    diep("set receive buffer size");

  uint64_t nFrames = 0;
  
  // clean out socket before we start
  //printf("READER: clearing buffer.\n"); fflush(stdout);
  //while ( recv(s, buf, BUFLEN, 0) > 0 );
  //printf("READER: buffer clear!\n"); fflush(stdout);

  while (access( "/home/ramdisk/QUIT", F_OK ) == -1)
  {
    /*
    if (nFrames % 100 == 0)
    {
        printf("Frame %d\n",nFrames);  fflush(stdout);
    }
    */
    nBytesReceived = recv(s, buf, BUFLEN, 0);
    //printf("read from socket %d %d!\n",nFrames, nBytesReceived); fflush(stdout);
    if (nBytesReceived == -1)
    {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
      {// recv timed out, clear the error and check again
        errno = 0;
        continue;
      }
      else
        diep("recvfrom()");
    }
    
    if (nBytesReceived == 0 ) continue;
    
    nTotalBytes += nBytesReceived;
    //printf("Received packet from %s:%d\nData: %s\n\n", 
    //inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port), buf);
    //printf("Received %d bytes. Data: ",nBytesReceived);

    if( nBytesReceived % 8 != 0 ) {
       printf("Misalign in reader %d\n",nBytesReceived); fflush(stdout);
    }

    ++nFrames; 

    //printf("read from socket %d %d!\n",nFrames, nBytesReceived);
    
    // write the socket data to shared memory
    while( rptr1->busy == 1 ) continue;
    rptr1->busy = 1;
    if( rptr1->unread >= (SHAREDBUF - BUFLEN) ) {
       perror("Data overflow 1 in Reader.\n");   
    } 
    else {
       memmove( &(rptr1->data[rptr1->unread]),buf,nBytesReceived);
       rptr1->unread += nBytesReceived;
    }      
    rptr1->busy = 0;

    while( rptr2->busy == 1 ) continue;
    rptr2->busy = 1;
    if( rptr2->unread >= (SHAREDBUF - BUFLEN) ) {
       perror("Data overflow 2 in Reader.\n");   
    } 
    else {
       memmove( &(rptr2->data[rptr2->unread]),buf,nBytesReceived);
       rptr2->unread += nBytesReceived;
    }      
    rptr2->busy = 0;

  }

  //fclose(dump_file);
  printf("received %ld frames, %ld bytes\n",nFrames,nTotalBytes);
  close(s);

  return;

}

// copied from http://material.karlov.mff.cuni.cz/people/hajek/Magon/lojza/merak.c
double timespec_subtract (struct timespec *x, struct timespec *y) {
  /* Perform the carry for the later subtraction by updating Y. */
  if (x->tv_nsec < y->tv_nsec)
    {
      int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
      y->tv_nsec -= 1000000000 * nsec;
      y->tv_sec += nsec;
    }

  if (1000000000 < x->tv_nsec - y->tv_nsec)
    {
      int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000;
      y->tv_nsec += 1000000000 * nsec;
      y->tv_sec -= nsec;
    }

  /* Compute the time remaining to wait.
     `tv_nsec' is certainly positive. */
//  *diff=1.*(double)(x->tv_sec - y->tv_sec) + 1e-9*(double)(x->tv_nsec - y->tv_nsec);
  //diff->tv_sec = x->tv_sec - y->tv_sec;
  //diff->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is positive. */
 // return y->tv_sec < x->tv_sec;
  return 1.*(double)(x->tv_sec - y->tv_sec) + 1e-9*(double)(x->tv_nsec - y->tv_nsec);
}

int main(void)
{
    pid_t pid;
    int rv;
    char buf[30];
    struct readoutstream *rptr1, *rptr2;

    signal(SIGCHLD, SIG_IGN);  /* now I don't have to wait()! */

    // Create shared memory for photon data
    rptr1 = OpenShared("/roachstream1");
    rptr2 = OpenShared("/roachstream2");
    
    // make sure we don't think we have data in the shared location
    rptr1->unread = 0;
    rptr1->busy = 0;
    rptr2->unread = 0;
    rptr2->busy = 0;        
        
    // Delete pre-existing control files
    remove("/home/ramdisk/START");
    remove("/home/ramdisk/STOP");
    remove("/home/ramdisk/QUIT");
    
    //printf("struct sizes = %d, %d\n",sizeof(struct hdrpacket),sizeof(struct datapacket));

    switch(pid = fork()) {
    case -1:
        perror("fork");  /* something went wrong */
        exit(1);         /* parent exits */

    case 0:
	Writer();
        exit(0);

    default:
	printf("You have invoked PacketMaster3.  This is the socket reader process.\n");
        printf("My PID is %d\n", getpid());
        printf("Writer's PID is %d\n", pid);

	// spawn Cuber
	if (!fork()) {
	        //printf("MASTER: Spawning Cuber\n"); fflush(stdout);
        	Cuber();
        	//printf("MASTER: Cuber died!\n"); fflush(stdout);
        	exit(0);
    	} 
        
	Reader();
	//TestReader();

        wait(NULL);
        printf("Reader: En Taro Adun!\n");
    }
    
    // close shared memory
    shm_unlink("/roachstream1");   
    shm_unlink("/roachstream2");   
}
