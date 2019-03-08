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

int startReaderThread(ReaderParams *rparams, ThreadParams *tparams){
    int rc; 
    pthread_attr_init(&(tparams->attr));
    rc = pthread_create(&(tparams->thread), &(tparams->attr), reader, rparams);
    if (rc){
        printf("ERROR creating reader(); return code from pthread_create() is %d\n", rc);
        //exit(-1);
    } 

    return rc;

}

int startBinWriterThread(BinWriterParams *rparams, ThreadParams *tparams){
    int rc; 
    pthread_attr_init(&(tparams->attr));
    rc = pthread_create(&(tparams->thread), &(tparams->attr), binWriter, rparams);
    if (rc){
        printf("ERROR creating binWriter(); return code from pthread_create() is %d\n", rc);
        //exit(-1);
    } 

    return rc;

}

int startShmImageWriterThread(ShmImageWriterParams *rparams, ThreadParams *tparams){
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
    struct hdrpacket *hdr;
    uint64_t swp,swp1;
    readoutstream_t *rptr;
    uint64_t pstart;

    uint64_t curTs;
    uint16_t *boardNums;
    uint16_t curRoachInd;
    uint32_t *doneIntegrating; //Array of bitmasks (one for each image, bits are roaches)
    uint32_t doneIntMask; //constant - each place value corresponds to a roach board
    int *takingImage;
    ShmImageWriterParams *params;
    MKID_IMAGE *sharedImages;

    params = (shmImageWriterParams*)prms; //cast param struct

    if(params->cpu != -1)
        ret = MaximizePriority(params->cpu);
    printf("SharedImageWriter online.\n");

    doneIntMask = (1<<(params->nRoach))-1;
    // open shared memory block 3 for photon data 
    rptr = params->roachStream;
        
    olddata = (char *) malloc(sizeof(char)*SHAREDBUF);
    
    memset(olddata, 0, sizeof(olddata[0])*2048);    // zero out array
    memset(data, 0, sizeof(data[0]) * 1024);    // zero out array
    memset(packet, 0, sizeof(packet[0]) * 808 * 2);    // zero out array
    boardNums = calloc(params->nRoach, sizeof(uint16_t));

    doneIntegrating = calloc(params->nSharedImages, sizeof(uint32_t));
    takingImage = calloc(params->nSharedImages, sizeof(int));
    sharedImages = (MKID_IMAGE*)malloc(params->nSharedImages*sizeof(MKID_IMAGE));

    for(imgIdx=0; imgIdx<params->nSharedImages; imgIdx++){
        MKIDShmImage_open(sharedImages+imgIdx, params->sharedImageNames[imgIdx]);
        memset(sharedImages[imgIdx].image, 0, sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nXPix * sharedImages[imgIdx].md->nYPix); 
        printf("zeroing block w/ size %d\n" ,sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nXPix * sharedImages[imgIdx].md->nYPix);

    }

    printf("SharedImageWriter done initializing\n");

    while (sem_trywait(params->quitSem) == -1)
    {
       // read in new data and parse it
       sem_wait(&sem[2]);        
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
       sem_post(&sem[2]);  


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
                      printf("SharedImageWriter: taking image %s\n", params->sharedImageNames[imgIdx]);
                      takingImage[imgIdx] = 1;
                      doneIntegrating[imgIdx] = 0;   
                      // zero out array:
                      memset(sharedImages[imgIdx].image, 0, sizeof(*(sharedImages[imgIdx].image)) * sharedImages[imgIdx].md->nXPix * sharedImages[imgIdx].md->nYPix); 
                      if(sharedImages[imgIdx].md->startTime==0)
                          sharedImages[imgIdx].md->startTime = curTs;
                   
                  }

             }
             
             swp = *((uint64_t *) (&olddata[i*8]));
             swp1 = __bswap_64(swp);
             hdr = (struct hdrpacket *) (&swp1);             

             if (hdr->start == 0b11111111) {        // found new packet header!
                // fill packet and parse
                memmove(packet,&olddata[pstart],i*8 - pstart);
                curRoachInd = 0;

                curTs = (uint64_t)hdr->timestamp;
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
                   if(takingImage[imgIdx])
                   {
                       //printf("curRoachTs: %lld\n", curTs);
                       if((curTs>sharedImages[imgIdx].md->startTime)&&(curTs<=(sharedImages[imgIdx].md->startTime+sharedImages[imgIdx].md->integrationTime)))
                           ParsePacketShm(sharedImages+imgIdx,packet,i*8 - pstart);
                       else if(curTs>(sharedImages[imgIdx].md->startTime+sharedImages[imgIdx].md->integrationTime))
                       {
                           doneIntegrating[imgIdx] |= (1<<curRoachInd);
                           //printf("SharedImageWriter: Roach %d done Integrating\n", boardNums[curRoachInd]);

                       }

                       //printf("SharedImageWriter: curTs %lld\n", curTs);
                       pcount++;

                       if(doneIntegrating[imgIdx]==doneIntMask) //check to see if all boards are done integrating
                       {
                           takingImage[imgIdx] = 0;
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
        MKIDShmImage_close(sharedImages+i);
    free(sharedImages);
    free(takingImage);
    free(doneIntegrating);

    //fclose(timeFile);
    printf("SharedImageWriter: Closing\n");
    return NULL;
}
