libpacketmaster.so:
	gcc -shared -o libpacketmaster.so -O0 -g -fPIC packetmaster.c -lmkidshm -lrt -lpthread

clean: 
	$(RM) *.o *.so 

install: libpacketmaster.so packetmaster.h
	cp libpacketmaster.so /usr/local/lib/
	cp *.h /usr/local/include/
