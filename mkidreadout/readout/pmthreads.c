#include "pmthreads.h"

int MaximizePriority(int cpu){
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
    params.sched_priority = sched_get_priority_max(SCHED_FIFO) - 20;
    printf("Setting priority to %d\n", params.sched_priority);
    
    ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    if (ret != 0) {
        // Print the error
        printf("Error setting thread realtime priority - %d,%d\n",params.sched_priority,ret);
        return(ret);     
    }
    
    return(ret);
}

int startReaderThread(READER_PARAMS *rparams, THREAD_PARAMS *tparams){
    int rc; 
    pthread_attr_init(&(tparams->attr));
    rc = pthread_create(&(tparams->thread), &(tparams->attr), reader, rparams);
    if (rc){
        printf("ERROR creating reader(); return code from pthread_create() is %d\n", rc);
        //exit(-1);
    } 

    return rc;

}

int startBinWriterThread(BIN_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams){
    int rc; 
    pthread_attr_init(&(tparams->attr));
    rc = pthread_create(&(tparams->thread), &(tparams->attr), binWriter, rparams);
    if (rc){
        printf("ERROR creating binWriter(); return code from pthread_create() is %d\n", rc);
        //exit(-1);
    } 

    return rc;

}

int startShmImageWriterThread(SHM_IMAGE_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams){
    int rc; 
    pthread_attr_init(&(tparams->attr));
    rc = pthread_create(&(tparams->thread), &(tparams->attr), shmImageWriter, rparams);
    if (rc){
        printf("ERROR creating shmImageWriter(); return code from pthread_create() is %d\n", rc);
        //exit(-1);
    } 

    return rc;

}

int startEventBuffWriterThread(EVENT_BUFF_WRITER_PARAMS *rparams, THREAD_PARAMS *tparams){
    int rc; 
    pthread_attr_init(&(tparams->attr));
    rc = pthread_create(&(tparams->thread), &(tparams->attr), eventBuffWriter, rparams);
    if (rc){
        printf("ERROR creating eventBuffWriter(); return code from pthread_create() is %d\n", rc);
        //exit(-1);
    } 

    return rc;

}

void *shmImageWriter(void *prms)
{
    int64_t i,j,ret,imgIdx;
    char packet[MAX_PACKSIZE];
    struct timespec startSpec;
    struct timespec stopSpec;
    struct timeval tv;
    unsigned long long sysTs;
    long nsElapsed;
    uint64_t pcount = 0;
    STREAM_HEADER *hdr;
    uint64_t swp,swp1;
    uint64_t pStartInd;
    uint64_t pStartCycle;
    RINGBUFFER *packBuf;
    uint64_t bufReadInd = 0;
    uint64_t lastCycle = 0;
    uint64_t nWriteCycles;
    uint64_t bufWriteInd;
    int nUnread;
    int packSize;

    uint64_t curTs;
    uint64_t prevTs;
    uint16_t *boardNums;
    uint16_t curRoachInd;
    uint16_t prevRoachInd;
    uint32_t *doneIntegrating; //Array of bitmasks (one for each image, bits are roaches)
    uint32_t doneIntMask; //constant - each place value corresponds to a roach board
    SHM_IMAGE_WRITER_PARAMS *params;
    MKID_IMAGE *sharedImages;
    sem_t *quitSem;
    sem_t *ringBufResetSem;

    params = (SHM_IMAGE_WRITER_PARAMS*)prms; //cast param struct

    if(params->cpu != -1)
        ret = MaximizePriority(params->cpu);
    printf("SharedImageWriter online.\n");

    packBuf = params->packBuf;

    doneIntMask = (1<<(params->nRoach))-1;
    printf("DONE INT MASK: %x\n", doneIntMask);
    printf("NROACH: %x\n", params->nRoach);

    quitSem = sem_open(params->quitSemName, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);
    ringBufResetSem = sem_open(params->ringBufResetSemName, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);
        
    
    memset(packet, 0, sizeof(packet[0]) * 808 * 2);    // zero out array
    boardNums = calloc(params->nRoach, sizeof(uint16_t));

    doneIntegrating = calloc(params->nSharedImages, sizeof(uint32_t));
    sharedImages = (MKID_IMAGE*)malloc(params->nSharedImages*sizeof(MKID_IMAGE));

    for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++){
        MKIDShmImage_open(sharedImages+imgIdx, params->sharedImageNames[imgIdx]);
        printf("opening shared image %s\n", params->sharedImageNames[imgIdx]);
        memset(sharedImages[imgIdx].image, 0, sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nCols * sharedImages[imgIdx].md->nRows); 
        printf("zeroing block w/ size %lu\n" ,sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nCols * sharedImages[imgIdx].md->nRows);

        MKIDShmImage_resetSems(sharedImages+imgIdx);


    }

    prevTs = 0;

    #ifdef _TIMING_TEST
    FILE *timeFile = fopen("timetest.txt", "w");
    #endif

    printf("SharedImageWriter done initializing\n");
    curRoachInd = 0;
    prevRoachInd = 0;
    pStartInd = 0;
    pStartCycle = 0;
    bufReadInd = 0; // first pack is header

    while (sem_trywait(quitSem) == -1)
    {
        getRingBufState(packBuf, ringBufResetSem, &nWriteCycles, &bufWriteInd);
        nUnread = (RINGBUF_SIZE)*(nWriteCycles - lastCycle) + (int)bufWriteInd - bufReadInd;

        if((nUnread + 8) < 0){
            printf("SharedImageWriter: nUnread < 0, unspecified glitch in ring buffer. Have fun! nUnread: %d writeInd: %lu nCycles: %lu\n", nUnread, bufWriteInd, nWriteCycles);
            printf("    bufReadInd: %lu lastCycle %lu\n", bufReadInd, lastCycle);

        }
        else if(nUnread > RINGBUF_SIZE){
            printf("SharedImageWriter: Missed %d bytes\n", nUnread - RINGBUF_SIZE);
            nUnread = RINGBUF_SIZE;
            bufReadInd = (bufWriteInd + 8)%RINGBUF_SIZE;
            if(bufReadInd % 8 > 0){
                printf("Misalign in shmImageWriter\n");
                bufReadInd -= bufReadInd%8;

            }

            lastCycle = nWriteCycles - 1;

        }

        if(nUnread % 8 >0)
            printf("Misalign in shmImageWriter\n");

        //if(nUnread > (RINGBUF_SIZE - bufReadInd))
        //    maxNToRead = RINGBUF_SIZE - bufReadInd;

        //else
        //    maxNToRead = nUnread;
        
        

        // if there is data waiting, process it
        if(nUnread > 0) {       
            // search the available data for a packet boundary
            for(i=0; i<nUnread; i+=8) { 
                for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++)
                {
                    //printf("looping through image %d\n", imgIdx); fflush(stdout);
                    //printf("Shared Image %d: %d\n", sharedImages[imgIdx]);
                    if(sem_trywait(sharedImages[imgIdx].takeImageSem)==0)
                    {
                        //printf("SharedImageWriter: taking image %s\n", params->sharedImageNames[imgIdx]);
                        #ifdef _DEBUG_OUTPUT
                        clock_gettime(CLOCK_REALTIME, &startSpec);
                        #endif
                        sharedImages[imgIdx].md->takingImage = 1;
                        doneIntegrating[imgIdx] = 0;   
                        strcpy(sharedImages[imgIdx].md->wavecalID, params->wavecal->solutionFile);
                        //sharedImages[imgIdx].md->valid = 1;
                        // zero out array:
                        memset(sharedImages[imgIdx].image, 0, sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nCols * sharedImages[imgIdx].md->nRows); 
                        if(sharedImages[imgIdx].md->startTime==0)
                            sharedImages[imgIdx].md->startTime = curTs;
                        #ifdef _DEBUG_OUTPUT
                        printf("SharedImageWriter: starting image at %lu, roach: %d\n", curTs, boardNums[curRoachInd]);
                        printf("                   startTime: %lu, int time: %lu\n", sharedImages[imgIdx].md->startTime, sharedImages[imgIdx].md->integrationTime);
                        #endif
                     
                    }

               }
               
               assert(bufReadInd % 8 == 0);

               swp = *((uint64_t *) (&packBuf->data[bufReadInd]));
               swp1 = __bswap_64(swp);
               hdr = (STREAM_HEADER *) (&swp1);             

               if (hdr->start == 0b11111111) {        // found new packet header!
                   // fill packet and parse
                   if(lastCycle == pStartCycle){
                       memmove(packet, &packBuf->data[pStartInd], bufReadInd - pStartInd);
                       packSize = bufReadInd - pStartInd;
                       pStartInd = bufReadInd;

                   }
                   else if(lastCycle == (pStartCycle + 1)){
                       if(bufReadInd > pStartInd)
                           printf("Shared mem overflow - skipped packet boundary\n");
                       assert(RINGBUF_SIZE - pStartInd <= MAX_PACKSIZE);
                       memmove(packet, &packBuf->data[pStartInd], RINGBUF_SIZE - pStartInd);
                       memmove(packet + (RINGBUF_SIZE - pStartInd), packBuf->data, bufReadInd);
                       packSize = RINGBUF_SIZE + (int)bufReadInd - pStartInd;
                       pStartInd = bufReadInd;
                       pStartCycle = lastCycle;

                   }

                   else
                       printf("Severe shared mem overflow! lastCycle: %lu, pStartCycle: %lu, nCycles %lu\n", lastCycle, pStartCycle, nWriteCycles);

                   prevRoachInd = curRoachInd;
                   curRoachInd = 0;

                   prevTs = curTs;
                   curTs = (uint64_t)hdr->timestamp;

                   #ifdef _TIMING_TEST
                   gettimeofday(&tv, NULL);
                   sysTs = (unsigned long long)(tv.tv_sec)*2000 + (unsigned long long)(tv.tv_usec)/500 - (unsigned long long)TSOFFS*2000;
                   #endif
                  
                   
                   //Figure out index corresponding to roach number (index of roachNum in boardNums)
                   //If this doesn't exist, assign it
                   for(j=0; j<params->nRoach; j++)
                   {
                       if(boardNums[j]==hdr->roach)
                       {
                           curRoachInd = j;
                           break;

                       }
                       if(boardNums[j]==0)
                       {
                           boardNums[j] = hdr->roach;
                           curRoachInd = j;
                           break;

                       }

                   }


                   #ifdef _TIMING_TEST
                   fprintf(timeFile, "%llu %llu %d\n", curTs, sysTs, boardNums[curRoachInd]);
                   #endif

                   //if(curTs < prevTs)
                   //    printf("Packet out of order. dt = %lu, curRoach = %d, prevRoach=%d \n", 
                   //          prevTs-curTs, boardNums[curRoachInd], boardNums[prevRoachInd]);
                   
                   for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++)
                   {
                       if(sharedImages[imgIdx].md->takingImage)
                       {
                           //printf("curRoachTs: %lld\n", curTs);
                           if((curTs>sharedImages[imgIdx].md->startTime)&&(curTs<=(sharedImages[imgIdx].md->startTime+sharedImages[imgIdx].md->integrationTime))){
                               addPacketToImage(sharedImages+imgIdx, packet, packSize, params->wavecal);
                               if((doneIntegrating[imgIdx] & (1<<curRoachInd)) == (1<<curRoachInd))
                                   printf("Packet out of order! roach: %d\n", boardNums[curRoachInd]);

                           }

                           else if(curTs>(sharedImages[imgIdx].md->startTime+sharedImages[imgIdx].md->integrationTime))
                           {
                               #ifdef _DEBUG_OUTPUT
                               if(!((doneIntegrating[imgIdx]>>curRoachInd)&1))
                                   printf("SharedImageWriter: Roach %d done Integrating\n", boardNums[curRoachInd]);
                               #endif
                               doneIntegrating[imgIdx] |= (1<<curRoachInd);

                           }

                           //printf("SharedImageWriter: curTs %lld\n", curTs);
                           pcount++;

                           if(doneIntegrating[imgIdx]==doneIntMask) //check to see if all boards are done integrating
                           {
                               sharedImages[imgIdx].md->takingImage = 0;
                               MKIDShmImage_postDoneSem(sharedImages + imgIdx, -1);
                               #ifdef _DEBUG_OUTPUT
                               clock_gettime(CLOCK_REALTIME, &stopSpec);
                               nsElapsed = 1000000000*(stopSpec.tv_sec - startSpec.tv_sec);
                               nsElapsed += (long)stopSpec.tv_nsec - startSpec.tv_nsec;
                               printf("SharedImageWriter: done image at %lu\n", curTs);
                               printf("SharedImageWriter: int time %lu\n", curTs-sharedImages[imgIdx].md->integrationTime);
                               printf("SharedImageWriter: real time %ld ms\n", (nsElapsed)/1000000);
                               //printf("SharedImageWriter: Parse rate = %lu pkts/img. Data in buffer = %lu\n",pcount,oldbr); fflush(stdout);
                               //printf("SharedImageWriter: forLoopIters %d\n", forLoopIters);
                               //printf("SharedImageWriter: whileLoopIters %d\n", whileLoopIters);
                               //printf("SharedImageWriter: oldbr: %lu\n\n", oldbr);
                               #endif
                               pcount = 0;

                           }
                       
                       }

                  }
	              //pStart = i*8;   // move start location for next packet	                      
               }

               bufReadInd += 8;
               if(bufReadInd >= RINGBUF_SIZE){
                   bufReadInd = 0;
                   lastCycle += 1;

               }
           }

        }                           
    }

    printf("SharedImageWriter: Freeing stuff\n");
    free(boardNums);
    for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++)
        MKIDShmImage_close(sharedImages+imgIdx);
    free(sharedImages);
    free(doneIntegrating);
    sem_close(quitSem);
    sem_close(ringBufResetSem);

    #ifdef _TIMING_TEST
    fclose(timeFile);
    #endif
    printf("SharedImageWriter: Closing\n");
    return NULL;
}

void *eventBuffWriter(void *prms)
{
    printf("EventBufferWriter: Closing\n");
    return NULL;

}

void* reader(void *prms){
    //set up a socket connection
    struct sockaddr_in si_me, si_other;
    int s, ret, i;
    ssize_t nBytesReceived = 0;
    ssize_t nTotalBytes = 0;
    size_t lastWriteSize;
    RINGBUFFER *packBuf;
    READER_PARAMS *params;
    sem_t *quitSem;
    sem_t *ringBufResetSem;
    char streamSemName[80];
    char overFlowBuf[BUFLEN];

    params = (READER_PARAMS*) prms;
    
    if(params->cpu != -1)
        ret = MaximizePriority(params->cpu);

    printf("READER: Connecting to Socket!\n"); fflush(stdout);

    //open semaphores
    quitSem = sem_open(params->quitSemName, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);
    ringBufResetSem = sem_open(params->ringBufResetSemName, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);

    packBuf = params->packBuf; //pointer to list of stream buffers, assume this is allocated
    packBuf->writeInd = 0;
    packBuf->nCycles = 0;
    sem_post(ringBufResetSem);

    if ((s=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP))==-1)
        diep("socket");
    printf("READER: socket created\n");
    fflush(stdout);

    memset((char *) &si_me, 0, sizeof(si_me));
    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(params->port);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(s, (const struct sockaddr *)(&si_me), sizeof(si_me))==-1)
        diep("bind");
    printf("READER: socket bind to port %d\n", params->port);
    fflush(stdout);

    //Set receive buffer size, the default is too small.  
    //If the system will not allow this size buffer, you will need
    //to use sysctl to change the max buffer size
    int retval = 0;
    int bufferSize = 536870912;
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
    #ifdef _DEBUG_READER_OUTPUT
    printf("DEBUG OUTPUT ON, listening for packets...\n");
    #endif

    while(sem_trywait(quitSem)==-1) //(access( "/home/ramdisk/QUIT", F_OK ) == -1)
    {
        #ifdef _DEBUG_READER_OUTPUT
        if (nFrames % 1000 == 0)
        {
            printf("Frame %d\n",nFrames);  fflush(stdout);
        }
        #endif

        if((RINGBUF_SIZE - packBuf->writeInd) <= BUFLEN){ //Need to use overflow buffer since we might cross ringbuffer boundary
            nBytesReceived = recv(s, overFlowBuf, BUFLEN, 0);
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

            else if (nBytesReceived == 0 ) continue;
            
            else if(nBytesReceived >= (RINGBUF_SIZE - packBuf->writeInd)){ //We've hit ringbuffer boundary
                memcpy(packBuf->data + packBuf->writeInd, overFlowBuf, RINGBUF_SIZE - packBuf->writeInd);
                lastWriteSize = RINGBUF_SIZE - packBuf->writeInd;

                sem_wait(ringBufResetSem);
                packBuf->writeInd = 0;
                packBuf->nCycles += 1;
                sem_post(ringBufResetSem);

                memcpy(packBuf->data + packBuf->writeInd, overFlowBuf + lastWriteSize, 
                        nBytesReceived - lastWriteSize); 
                packBuf->writeInd += nBytesReceived - lastWriteSize;
                #ifdef _DEBUG_READER_OUTPUT
                printf("Reader: nRingBufCycles: %lu\n", packBuf->nCycles);
                printf("    nBytesReceived: %ld\n", nBytesReceived);
                printf("    lastWriteSize: %ld\n", lastWriteSize);
                printf("    writeInd: %lu\n", packBuf->writeInd);
                #endif 

            }

            else{
                memcpy(packBuf->data + packBuf->writeInd, overFlowBuf, nBytesReceived);
                packBuf->writeInd += nBytesReceived;
                assert(packBuf->writeInd < RINGBUF_SIZE);

            }

        }



        else{
            nBytesReceived = recv(s, packBuf->data + packBuf->writeInd, BUFLEN, 0);
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
            else if (nBytesReceived == 0 ) continue;


            //else if((nBytesReceived + packBuf->writeInd) > RINGBUF_SIZE)
            //    printf("READER: RINGBUFFER MEMORY OVERFLOW\n");
            else if((nBytesReceived + packBuf->writeInd) == RINGBUF_SIZE){
                printf("READER: line 518 shouldn't happen\n");
                sem_wait(ringBufResetSem);
                packBuf->writeInd = 0;
                packBuf->nCycles += 1;
                sem_post(ringBufResetSem);

            }
            else
                packBuf->writeInd += nBytesReceived;

        }

        nTotalBytes += nBytesReceived;
        //printf("Received packet from %s:%d\nData: %s\n\n", 
        //        inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port), buf);
        //printf("Received %d bytes. Data: ",nBytesReceived);

        if( nBytesReceived % 8 != 0 ) {
            printf("Misalign in reader %ld\n",nBytesReceived); fflush(stdout);
        }

        ++nFrames; 


    }

    //fclose(dump_file);
    printf("received %ld frames, %ld bytes\n",nFrames,nTotalBytes);
    close(s);

    sem_close(quitSem);
    sem_close(ringBufResetSem);

    printf("Reader closing\n");

    return NULL;

}

void* binWriter(void *prms)
{
    //long            ms; // Milliseconds
    time_t s,olds;  // Seconds
    struct timespec spec;
    long outcount;
    int ret, mode=0;
    FILE *wp;
    //char data[1024];
    char fname[120];
    RINGBUFFER *packBuf;
    uint64_t bufReadInd = 0;
    uint64_t lastCycle = 0;
    uint64_t bufWriteInd;
    uint64_t nWriteCycles;
    int nUnread;
    BIN_WRITER_PARAMS *params;
    sem_t *quitSem;
    sem_t *ringBufResetSem;

    params = (BIN_WRITER_PARAMS*)prms; //cast param struct
    if(params->cpu!=-1)
        ret = MaximizePriority(params->cpu);

    packBuf = params->packBuf;

    wp = NULL;

    printf("Rev up the RAID array, WRITER is active!\n");

    quitSem = sem_open(params->quitSemName, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);
    ringBufResetSem = sem_open(params->ringBufResetSemName, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);
    
    // open shared memory block 1 for photon data
    
    //  Write looks for a file on /home/ramdisk named "START" which contains the write path.  
    //  If this file is present, enter writing mode
    //  and delete the file.  Keep writing until the file "STOP" file appears.
    //  Shut down if a "QUIT" file appears 

    // mode = 0 :  Not doing anything, just keep pipe clear and junk and data that comes down it
    // mode = 1 :  "START" file detected enter write mode
    // mode = 2 :  continous writing mode, watch for "STOP" or "QUIT" files
    // mode = 3 :  QUIT file detected, exit

    while (sem_trywait(quitSem) == -1){
       // keep the shared mem clean!       
       if( mode == 0 ) {
           getRingBufState(packBuf, ringBufResetSem, &nWriteCycles, &bufWriteInd);
           bufReadInd = bufWriteInd;
           lastCycle = nWriteCycles;

       }

       if(mode == 0 && params->writing == 1) {
          // start file exists, go to mode 1
           mode = 1;
           printf("Mode 0->1\n");
       } 

       if( mode == 1 ) {
          // read path from start, generate filename, and open file pointer for writing

          clock_gettime(CLOCK_REALTIME, &spec);   
          s  = spec.tv_sec;
          olds = s;
          sprintf(fname,"%s%ld.bin",params->writerPath,s);
          printf("Writing to %s\n",fname);
          wp = fopen(fname,"wb");
          mode = 2;
          outcount = 0;
          printf("Mode 1->2\n");
       }

       if( mode == 2 ) {
          if (params->writing == 0) {
             // stop file exists, finish up and go to mode 0
	         fclose(wp);
             wp = NULL;
             mode = 0;
             printf("Mode 2->0\n");
          } else {
             // continuous write mode, store data from the pipe to disk

	         // start a new file every 1 seconds
             clock_gettime(CLOCK_REALTIME, &spec);   
             s  = spec.tv_sec;

             if( s - olds >= 1 ) {
                 fclose(wp);
                 wp = NULL;
                 sprintf(fname,"%s%ld.bin",params->writerPath,s);
                 printf("WRITER: Writing to %s, rate = %ld MBytes/sec\n",fname,outcount/1000000);
                 wp = fopen(fname,"wb");
                 olds = s;
                 outcount = 0;               
             }

	         // write all data in shared memory to disk
             getRingBufState(packBuf, ringBufResetSem, &nWriteCycles, &bufWriteInd);
             nUnread = (RINGBUF_SIZE)*(nWriteCycles - lastCycle) + (int)bufWriteInd - bufReadInd;
             if(nUnread < 0)
                 printf("Writer: nUnread < 0, unspecified glitch in ring buffer. Have fun!\n");
             else if(nUnread > RINGBUF_SIZE){
                 printf("Writer: Missed %d bytes\n", nUnread - RINGBUF_SIZE);
                 nUnread = RINGBUF_SIZE;

                 bufReadInd = (bufWriteInd + 1)%RINGBUF_SIZE;
                 lastCycle = nWriteCycles - 1;

             }
             if(nUnread >= BINWRITER_MINSIZE){
                 if(nUnread >= (RINGBUF_SIZE - bufReadInd)){ 
	                fwrite(packBuf->data + bufReadInd, 1, RINGBUF_SIZE - bufReadInd, wp);    	         
	                fwrite(packBuf->data, 1, nUnread - (RINGBUF_SIZE - bufReadInd), wp);
                    lastCycle += 1;
                    outcount += RINGBUF_SIZE - bufReadInd;
                    bufReadInd = nUnread - (RINGBUF_SIZE - bufReadInd);

                 }

                else{
	                   fwrite(packBuf->data + bufReadInd, 1, nUnread, wp);    	         
                       bufReadInd += nUnread;
                       outcount += nUnread;

                    }

             }


                

          }
       }

    }

    if(wp!=NULL)
	  fclose(wp);
    sem_close(quitSem);
    sem_close(ringBufResetSem);

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
    printf("WRITER: Closing\n"); fflush(stdout);
    return NULL;
}

void addPacketToImage(MKID_IMAGE *sharedImage, char *photonWord, 
        unsigned int l, WAVECAL_BUFFER *wavecal)
{
    uint64_t i;
    PHOTON_WORD *data;
    uint64_t swp,swp1;
    float wvl, wvlBinSpacing;
    int wvlBinInd;

    for(i=1;i<l/8;i++) {
       
        swp = *((uint64_t *) (&photonWord[i*8]));
        swp1 = __bswap_64(swp);
        data = (PHOTON_WORD *) (&swp1);
        
        if( data->xcoord >= sharedImage->md->nCols || data->ycoord >= sharedImage->md->nRows ) 
            continue;

        if((sharedImage->md->useWvl)){
            if(wavecal == NULL){
                perror("ERROR: No wavecal buffer specified!");
                continue;

            }
            wvl = getWavelength(data, wavecal);

            if(sharedImage->md->useEdgeBins){
                if(wvl < sharedImage->md->wvlStart)
                    wvlBinInd = 0;
                else if(wvl >= sharedImage->md->wvlStop)
                    wvlBinInd = sharedImage->md->nWvlBins + 1;
                else{
                    wvlBinSpacing = (double)(sharedImage->md->wvlStop - sharedImage->md->wvlStart)/sharedImage->md->nWvlBins;
                    wvlBinInd = (int)(wvl - sharedImage->md->wvlStart)/wvlBinSpacing + 1;

                }
            }

            else{
                if((wvl < sharedImage->md->wvlStart) || (wvl >= sharedImage->md->wvlStop))
                    continue;
                else{
                    wvlBinSpacing = (double)(sharedImage->md->wvlStop - sharedImage->md->wvlStart)/sharedImage->md->nWvlBins;
                    wvlBinInd = (int)(wvl - sharedImage->md->wvlStart)/wvlBinSpacing;

                }

            }

            if(sharedImage->md->takingImage)
                sharedImage->image[(sharedImage->md->nCols)*(sharedImage->md->nRows)*wvlBinInd + (sharedImage->md->nCols)*(data->ycoord) + data->xcoord]++;

        }
        
        else
            if(sharedImage->md->takingImage)
                sharedImage->image[(sharedImage->md->nCols)*(data->ycoord) + data->xcoord]++;
      
    }

}

void addPacketToEventBuffer(MKID_EVENT_BUFFER *buffer, char *photonWord, 
        unsigned int l, uint64_t headerTS, WAVECAL_BUFFER *wavecal, int nRows, int nCols)
{
    uint64_t i;
    PHOTON_WORD *data;
    MKID_PHOTON_EVENT photon;
    uint64_t swp,swp1;
    float wvl, wvlBinSpacing;
    int wvlBinInd;

    for(i=1;i<l/8;i++) {
       
        swp = *((uint64_t *) (&photonWord[i*8]));
        swp1 = __bswap_64(swp);
        data = (PHOTON_WORD *) (&swp1);
        
        if( data->xcoord >= nCols || data->ycoord >= nRows ) 
            continue;

        if((buffer->md->useWvl)){
            if(wavecal == NULL){
                perror("ERROR: No wavecal buffer specified!");
                continue;

            }
            wvl = getWavelength(data, wavecal);

        }
            
        else
            wvl = (float)data->phase/PHASE_BIN_PT; //phase in radians

        photon.time = 500*((uint64_t)2000*TSOFFS + headerTS) + data->timestamp;
        photon.x = data->xcoord;
        photon.y = data->ycoord;
        photon.wvl = (wvl_t)wvl;

        MKIDShmEventBuffer_addEvent(buffer, &photon);

    }


}

float getWavelength(PHOTON_WORD *photon, WAVECAL_BUFFER *wavecal){
    float phase = (float)photon->phase*RAD_TO_DEG/PHASE_BIN_PT;
    int bufferInd = 3*(wavecal->nCols * photon->ycoord + photon->xcoord);
    float energy = phase*phase*wavecal->data[bufferInd] + phase*wavecal->data[bufferInd+1]
        + wavecal->data[bufferInd+2];
    //printf("%f %f | ", phase, energy);
    return H_TIMES_C/energy;

}

void getRingBufState(RINGBUFFER* packBuf, sem_t *ringBufResetSem, uint64_t *nCycles, uint64_t *writeInd){
    sem_wait(ringBufResetSem);
    *nCycles = packBuf->nCycles;
    *writeInd = packBuf->writeInd;
    sem_post(ringBufResetSem);
    
}

void resetSem(const char *semName){
    sem_t *sem;
    char name[80];
    snprintf(name, 80, "%s", semName);
    sem = sem_open(name, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP, 0);
    while(sem_trywait(sem) == 0);

}

void quitAllThreads(const char *quitSemName, int nThreads){
    int i;
    sem_t *quitSem;
    char name[80];
    snprintf(name, 80, "%s", quitSemName);
    quitSem = sem_open(name, 0);
    for(i=0; i<nThreads; i++)
        sem_post(quitSem);

}

void diep(char *s){
    printf("errono: %d",errno);
    perror(s);
    exit(1);
}
