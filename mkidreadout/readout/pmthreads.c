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
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    
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

void *shmImageWriter(void *prms)
{
    int64_t br,i,j,ret,imgIdx;
    char data[1024];
    char *olddata;
    char packet[808*16];
    struct timespec startSpec;
    struct timespec stopSpec;
    struct timeval tv;
    unsigned long long sysTs;
    uint64_t nsElapsed;
    uint64_t oldbr = 0;     // number of bytes of unparsed data sitting in olddata
    uint64_t pcount = 0;
    STREAM_HEADER *hdr;
    uint64_t swp,swp1;
    READOUT_STREAM *rptr;
    uint64_t pstart;

    uint64_t curTs;
    uint64_t prevTs;
    uint16_t *boardNums;
    uint16_t curRoachInd;
    uint32_t *doneIntegrating; //Array of bitmasks (one for each image, bits are roaches)
    uint32_t doneIntMask; //constant - each place value corresponds to a roach board
    SHM_IMAGE_WRITER_PARAMS *params;
    MKID_IMAGE *sharedImages;
    sem_t *quitSem;
    sem_t *streamSem;

    params = (SHM_IMAGE_WRITER_PARAMS*)prms; //cast param struct

    if(params->cpu != -1)
        ret = MaximizePriority(params->cpu);
    printf("SharedImageWriter online.\n");

    doneIntMask = (1<<(params->nRoach))-1;
    rptr = params->roachStream;

    quitSem = sem_open(params->quitSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    streamSem = sem_open(params->streamSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    sem_post(streamSem);
        
    olddata = (char *) malloc(sizeof(char)*SHAREDBUF);
    
    memset(olddata, 0, sizeof(olddata[0])*2048);    // zero out array
    memset(data, 0, sizeof(data[0]) * 1024);    // zero out array
    memset(packet, 0, sizeof(packet[0]) * 808 * 2);    // zero out array
    boardNums = calloc(params->nRoach, sizeof(uint16_t));

    doneIntegrating = calloc(params->nSharedImages, sizeof(uint32_t));
    sharedImages = (MKID_IMAGE*)malloc(params->nSharedImages*sizeof(MKID_IMAGE));

    for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++){
        MKIDShmImage_open(sharedImages+imgIdx, params->sharedImageNames[imgIdx]);
        printf("opening shared image %s\n", params->sharedImageNames[imgIdx]);
        memset(sharedImages[imgIdx].image, 0, sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nCols * sharedImages[imgIdx].md->nRows); 
        printf("zeroing block w/ size %lu\n" ,sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nCols * sharedImages[imgIdx].md->nRows);

    }

    prevTs = 0;

    printf("SharedImageWriter done initializing\n");

    while (sem_trywait(quitSem) == -1)
    {
       // read in new data and parse it
       sem_wait(streamSem);        
       br = rptr->unread;
       if( br%8 != 0 ) printf("Misalign in SharedImageWriter - %d\n",(int)br); 
	   	    
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
                printf("oldbr = %d!  Dumping data!\n",(int)oldbr); fflush(stdout);
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
       sem_post(streamSem);  


       // if there is data waiting, process it
       pstart = 0;
       if( oldbr >= 808 ) {       
          // search the available data for a packet boundary
          for( i=1; (uint64_t)i<oldbr/8; i++) { 
              for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++)
              {
                  //printf("looping through image %d\n", imgIdx); fflush(stdout);
                  //printf("Shared Image %d: %d\n", sharedImages[imgIdx]);
                  if(sem_trywait(sharedImages[imgIdx].takeImageSem)==0)
                  {
                      //printf("SharedImageWriter: taking image %s\n", params->sharedImageNames[imgIdx]);
                      sharedImages[imgIdx].md->takingImage = 1;
                      doneIntegrating[imgIdx] = 0;   
                      strcpy(sharedImages[imgIdx].md->wavecalID, params->wavecal->solutionFile);
                      //sharedImages[imgIdx].md->valid = 1;
                      // zero out array:
                      memset(sharedImages[imgIdx].image, 0, sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nCols * sharedImages[imgIdx].md->nRows); 
                      if(sharedImages[imgIdx].md->startTime==0)
                          sharedImages[imgIdx].md->startTime = curTs;
                   
                  }

             }
             
             swp = *((uint64_t *) (&olddata[i*8]));
             swp1 = __bswap_64(swp);
             hdr = (STREAM_HEADER *) (&swp1);             

             if (hdr->start == 0b11111111) {        // found new packet header!
                // fill packet and parse
                memmove(packet,&olddata[pstart],i*8 - pstart);
                curRoachInd = 0;

                prevTs = curTs;
                curTs = (uint64_t)hdr->timestamp;
               
                if(curTs < prevTs)
                    printf("Packet out of order check 0");
                //gettimeofday(&tv, NULL);
                //sysTs = (unsigned long long)(tv.tv_sec)*1000 + (unsigned long long)(tv.tv_usec)/1000 - (unsigned long long)TSOFFS*1000;
                //sysTs = sysTs*2;

                //fprintf(timeFile, "%llu %llu\n", curTs, sysTs);
                
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
                
                for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++)
                {
                   if(sharedImages[imgIdx].md->takingImage)
                   {
                       //printf("curRoachTs: %lld\n", curTs);
                       if((curTs>sharedImages[imgIdx].md->startTime)&&(curTs<=(sharedImages[imgIdx].md->startTime+sharedImages[imgIdx].md->integrationTime))){
                           addPacketToImage(sharedImages+imgIdx,packet,i*8 - pstart, params->wavecal);
                           if((doneIntegrating[imgIdx] & (1<<curRoachInd)) == (1<<curRoachInd))
                               printf("Packet out of order!\n");

                       }

                       else if(curTs>(sharedImages[imgIdx].md->startTime+sharedImages[imgIdx].md->integrationTime))
                       {
                           doneIntegrating[imgIdx] |= (1<<curRoachInd);
                           //printf("SharedImageWriter: Roach %d done Integrating\n", boardNums[curRoachInd]);

                       }

                       //printf("SharedImageWriter: curTs %lld\n", curTs);
                       pcount++;

                       if(doneIntegrating[imgIdx]==doneIntMask) //check to see if all boards are done integrating
                       {
                           sharedImages[imgIdx].md->takingImage = 0;
                           clock_gettime(CLOCK_REALTIME, &stopSpec);
                           //nsElapsed = stopSpec.tv_nsec - startSpec.tv_nsec;
                           MKIDShmImage_postDoneSem(sharedImages + imgIdx, -1);
                           printf("SharedImageWriter: done image at %lu\n", curTs);
                           printf("SharedImageWriter: int time %lu\n", curTs-sharedImages[imgIdx].md->integrationTime);
                           //printf("SharedImageWriter: real time %d ms\n", (nsElapsed)/1000000);
                           printf("SharedImageWriter: Parse rate = %lu pkts/img. Data in buffer = %lu\n",pcount,oldbr); fflush(stdout);
                           //printf("SharedImageWriter: forLoopIters %d\n", forLoopIters);
                           //printf("SharedImageWriter: whileLoopIters %d\n", whileLoopIters);
                           printf("SharedImageWriter: oldbr: %lu\n", oldbr);
                           pcount = 0;

                       }
                   
                   }

               }
		pstart = i*8;   // move start location for next packet	                      
             }
          }

	  // if there is data remaining save it for next run through
          memmove(olddata,&olddata[pstart],oldbr-pstart);
          oldbr = oldbr-pstart;          

       }                           
    }

    printf("SharedImageWriter: Freeing stuff\n");
    free(olddata);
    free(boardNums);
    for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++)
        MKIDShmImage_close(sharedImages+imgIdx);
    free(sharedImages);
    free(doneIntegrating);
    sem_close(streamSem);
    sem_close(quitSem);

    //fclose(timeFile);
    printf("SharedImageWriter: Closing\n");
    return NULL;
}

void* reader(void *prms){
    //set up a socket connection
    struct sockaddr_in si_me, si_other;
    int s, ret, i;
    unsigned char buf[BUFLEN];
    ssize_t nBytesReceived = 0;
    ssize_t nTotalBytes = 0;
    READOUT_STREAM *rptrs;
    READER_PARAMS *params;
    sem_t *quitSem;
    sem_t **streamSems;
    char streamSemName[80];

    params = (READER_PARAMS*) prms;
    
    if(params->cpu != -1)
        ret = MaximizePriority(params->cpu);

    printf("READER: Connecting to Socket!\n"); fflush(stdout);

    rptrs = params->roachStreamList; //pointer to list of stream buffers, assume this is allocated

    //open semaphores
    quitSem = sem_open(params->quitSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    streamSems = (sem_t**) malloc(params->nRoachStreams * sizeof(sem_t*));
    for(i=0; i<params->nRoachStreams; i++){
        snprintf(streamSemName, 80, "%s%d", params->streamSemBaseName, i);
        streamSems[i] = sem_open(streamSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    
    }
        

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

    while(sem_trywait(quitSem)==-1) //(access( "/home/ramdisk/QUIT", F_OK ) == -1)
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
        //        inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port), buf);
        //printf("Received %d bytes. Data: ",nBytesReceived);

        if( nBytesReceived % 8 != 0 ) {
            printf("Misalign in reader %ld\n",nBytesReceived); fflush(stdout);
        }

        ++nFrames; 

        //printf("read from socket %d %d!\n",nFrames, nBytesReceived);
        
        // write the socket data to shared memory
        for(i=0; i<params->nRoachStreams; i++){
            sem_wait(streamSems[i]);
            if(rptrs[i].unread >= (SHAREDBUF - BUFLEN)) {
                perror("Data overflow in reader.\n");

            }

            else{
                memmove(&(rptrs[i].data[rptrs[i].unread]), buf, nBytesReceived);
                rptrs[i].unread += nBytesReceived;
            }
            sem_post(streamSems[i]);

        }

    }

    //fclose(dump_file);
    printf("received %ld frames, %ld bytes\n",nFrames,nTotalBytes);
    close(s);

    for(i=0; i<params->nRoachStreams; i++)
        sem_close(streamSems[i]);

    sem_close(quitSem);

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
    READOUT_STREAM *rptr;
    BIN_WRITER_PARAMS *params;
    sem_t *quitSem;
    sem_t *streamSem;

    params = (BIN_WRITER_PARAMS*)prms; //cast param struct
    if(params->cpu!=-1)
        ret = MaximizePriority(params->cpu);

    wp = NULL;

    printf("Rev up the RAID array, WRITER is active!\n");

    quitSem = sem_open(params->quitSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    streamSem = sem_open(params->streamSemName, O_CREAT, S_IRUSR | S_IWUSR, 0);
    sem_post(streamSem);

    
    // open shared memory block 1 for photon data
    rptr = params->roachStream;
    
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
          sem_wait(streamSem);
	      rptr->unread = 0;
	      sem_post(streamSem);
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
	         sem_wait(streamSem);
	         if( rptr->unread > 0 ) {
	            fwrite( rptr->data, 1, rptr->unread, wp);    // could probably speed this up by copying data to new array and doing the fwrite after setting busy to 0
	            outcount += rptr->unread; 
	            rptr->unread = 0;
	         }
	         sem_post(streamSem);
          }
       }

    }

    if(wp!=NULL)
	  fclose(wp);
    sem_close(quitSem);
    sem_close(streamSem);

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

float getWavelength(PHOTON_WORD *photon, WAVECAL_BUFFER *wavecal){
    float phase = (float)photon->phase/PHASE_BIN_PT;
    int bufferInd = 3*(wavecal->nCols * photon->ycoord + photon->xcoord);
    float energy = phase*phase*wavecal->data[bufferInd] + phase*wavecal->data[bufferInd+1]
        + wavecal->data[bufferInd+2];
    return H_TIMES_C/energy;

}

void resetQuitSem(const char *quitSemName){
    sem_t *quitSem;
    char name[80];
    snprintf(name, 80, "%s", quitSemName);
    quitSem = sem_open(name, 0);
    while(sem_trywait(quitSem) == 0);

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
