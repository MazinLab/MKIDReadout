#define _GNU_SOURCE

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
#include <sched.h>

#define _POSIX_C_SOURCE 200809L
#define BUFLEN 1500
#define PORT 50000
#define XPIX 80
#define YPIX 125
#define NROACH 10
#define SHAREDBUF 536870912
#define TSOFFS 1483228800


#define handle_error_en(en, msg) \
        do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

// global semaphores for locking shared memory.  sem0 = rptr1, sem1 = rptr2 
static sem_t sem[2];

//#define LOGPATH "/mnt/data0/logs/"

// compile with gcc -Wall -Wextra -o PacketMaster6 PacketMaster6.c -I. -lm -lrt -lpthread -O3

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
    char data[SHAREDBUF];
};

void diep(char *s)
{
  printf("errono: %d",errno);
  perror(s);
  exit(1);
}

// maximize thread priority and set to desired CPU
int MaximizePriority(int cpu)
{
    int ret,s;        
    struct sched_param params;  // struct sched_param is used to store the scheduling priority
    
    pthread_t this_thread = pthread_self();
    
    cpu_set_t cpuset;

    // Set affinity mask to include cpu passed to thread
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    s = pthread_setaffinity_np(this_thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
        handle_error_en(s, "pthread_setaffinity_np");

    // Check the actual affinity mask assigned to the thread 
    s = pthread_getaffinity_np(this_thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
        handle_error_en(s, "pthread_getaffinity_np");

    printf("Set returned by pthread_getaffinity_np() contained:\n");
    if (CPU_ISSET(cpu, &cpuset)) printf("    CPU %d\n", cpu);    
    
    // We'll set the priority to the maximum.
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    
    ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    if (ret != 0) {
        // Print the error
        printf("Error setting thread realtime priority - %d,%d\n",params.sched_priority,ret);
        return(ret);     
    }
    
    return(ret);
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
    int64_t br,i,ret;
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
    uint64_t swp,swp1;
    struct readoutstream *rptr;
    uint64_t pstart;
    struct timeval tv;
    unsigned long long sysTs;
    uint64_t roachTs;

    ret = MaximizePriority(6);
    printf("Fear the wrath of CUBER!\n");
    
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
    
    //FILE *timeFile = fopen("timetestPk6.txt", "w");

    while (access( "/mnt/ramdisk/QUIT", F_OK ) == -1)
    {
       // if it is a new second, zero the image array and start over
       clock_gettime(CLOCK_REALTIME, &spec);   
       s  = spec.tv_sec;
       if( s > olds ) {                 
          // we are in a new second, so write out image array and then zero out the array
          sprintf(outfile,"/mnt/ramdisk/%d.img",olds);
          wp = fopen(outfile,"wb");
          fwrite(image, sizeof(image[0][0]), XPIX * YPIX, wp);
          fclose(wp);

          olds = s;
          memset(image, 0, sizeof(image[0][0]) * XPIX * YPIX);    // zero out array
          printf("CUBER: Parse rate = %d pkts/sec. Data in buffer = %d\n",pcount,oldbr); fflush(stdout);
          pcount=0;
       }
       
       // not a new second, so read in new data and parse it
       sem_wait(&sem[1]);        
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
       sem_post(&sem[1]);  

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
                roachTs = (uint64_t)hdr->timestamp;
                gettimeofday(&tv, NULL);
                sysTs = (unsigned long long)(tv.tv_sec)*1000 + (unsigned long long)(tv.tv_usec)/1000 - (unsigned long long)TSOFFS*1000;
                sysTs = sysTs*2;
                //fprintf(timeFile, "%llu %llu\n", roachTs, sysTs);

                memmove(packet,&olddata[pstart],i*8 - pstart);
                pcount++;                
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
    //long            ms; // Milliseconds
    time_t          s,olds;  // Seconds
    struct timespec spec;
    long outcount;
    int ret, mode=0;
    FILE *wp, *rp;
    //char data[1024];
    char path[80];
    char fname[120];
    struct readoutstream *rptr;
    
    ret = MaximizePriority(4);

    printf("Rev up the RAID array, WRITER is active!\n");

    // open shared memory block 1 for photon data
    rptr = OpenShared("/roachstream1");
    
    //  Write looks for a file on /mnt/ramdisk named "START" which contains the write path.  
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
          sem_wait(&sem[0]);
	      rptr->unread = 0;
	      sem_post(&sem[0]);
       }

       if( mode == 0 && access( "/mnt/ramdisk/START", F_OK ) != -1 ) {
          // start file exists, go to mode 1
           mode = 1;
           printf("Mode 0->1\n");
       } 

       if( mode == 1 ) {
          // read path from start, generate filename, and open file pointer for writing
          rp = fopen("/mnt/ramdisk/START","r");
          fscanf(rp,"%s",path);
          fclose(rp);
          remove("/mnt/ramdisk/START");

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
          if ( access( "/mnt/ramdisk/STOP", F_OK ) != -1 ) {
             // stop file exists, finish up and go to mode 0
	         fclose(wp);
             remove("/mnt/ramdisk/STOP");
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
	         sem_wait(&sem[0]);
	         if( rptr->unread > 0 ) {
	            fwrite( rptr->data, 1, rptr->unread, wp);    // could probably speed this up by copying data to new array and doing the fwrite after setting busy to 0
	            outcount += rptr->unread; 
	            rptr->unread = 0;
	         }
	         sem_post(&sem[0]);
          }
       }

       // check for quit flag and then bug out if received! 
       if( access( "/mnt/ramdisk/QUIT", F_OK ) != -1 ) {
	      fclose(wp);
          remove("/mnt/ramdisk/START");
          remove("/mnt/ramdisk/STOP");
          remove("/mnt/ramdisk/QUIT");
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
  int s,ret;
  unsigned char buf[BUFLEN];
  ssize_t nBytesReceived = 0;
  ssize_t nTotalBytes = 0;
  struct readoutstream *rptr1, *rptr2;
  
  ret = MaximizePriority(2);

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

  while (access( "/mnt/ramdisk/QUIT", F_OK ) == -1)
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
    sem_wait(&sem[0]);       
    if( rptr1->unread >= (SHAREDBUF - BUFLEN) ) {
       perror("Data overflow 1 in Reader.\n");   
    } 
    else {
       memmove( &(rptr1->data[rptr1->unread]),buf,nBytesReceived);
       rptr1->unread += nBytesReceived;
    }      
    sem_post(&sem[0]);
    
    sem_wait(&sem[1]);
    if( rptr2->unread >= (SHAREDBUF - BUFLEN) ) {
       perror("Data overflow 2 in Reader.\n");   
    } 
    else {
       memmove( &(rptr2->data[rptr2->unread]),buf,nBytesReceived);
       rptr2->unread += nBytesReceived;
    }      
    sem_post(&sem[1]);

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
    
    pthread_t threads[3];
    pthread_attr_t attr;
    void *status;
    
    int rc,t;
    char buf[30];
    struct readoutstream *rptr1, *rptr2;
    
    // Delete pre-existing control files
    remove("/mnt/ramdisk/START");
    remove("/mnt/ramdisk/STOP");
    remove("/mnt/ramdisk/QUIT");
    
    // Initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Set up semaphores
    sem_init(&sem[0], 0, 1);
    sem_init(&sem[1], 0, 1);

    // Create shared memory for photon data
    rptr1 = OpenShared("/roachstream1");
    rptr2 = OpenShared("/roachstream2");
    
    t=0;
    rc = pthread_create(&threads[0], &attr, Reader, (void *) &t);
    if (rc){
        printf("ERROR creating Reader(); return code from pthread_create() is %d\n", rc);
        exit(-1);
    }
    
    rc = pthread_create(&threads[1], &attr, Writer, (void *) &t);
    if (rc){
        printf("ERROR creating Writer(); return code from pthread_create() is %d\n", rc);
        exit(-1);
    }
    
    rc = pthread_create(&threads[2], &attr, Cuber, (void *) &t);
    if (rc){
        printf("ERROR creating Cuber(); return code from pthread_create() is %d\n", rc);
        exit(-1);
    }
    
    pthread_attr_destroy(&attr);
    rc = pthread_join(threads[1], &status); // wait until we detect quit condition in Writer()
    if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
    }
                       
    // close shared memory
    printf("Closing shared memory");
    sem_wait(&sem[0]);  // stop messing with memory 
    sem_wait(&sem[1]);      
    
    printf("Killing Cuber and Reader");
    pthread_cancel(threads[0]);  // kill Reader
    pthread_cancel(threads[2]);  // kill Cuber
    
    shm_unlink("/roachstream1");   
    shm_unlink("/roachstream2");   
    sem_close(&sem[0]);
    sem_close(&sem[1]);
    
    pthread_exit(NULL);  // close up shop
}
