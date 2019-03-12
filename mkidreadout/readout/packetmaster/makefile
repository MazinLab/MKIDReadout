packetmaster.so: pypacketmaster.c libpacketmaster.so
	gcc pypacketmaster.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o packetmaster.so -L. -lpacketmaster -lmkidshm -lpthread -lrt

pypacketmaster.c:
	cython packetmaster.pyx -o pypacketmaster.c

libpacketmaster.so:
	gcc -shared -o libpacketmaster.so -fPIC packetmaster.c -lmkidshm -lrt -lpthread

clean: 
	$(RM) *.o *.so pypacketmaster.c

install: packetmaster.so packetmaster.h
	cp *.so /usr/local/lib/
	cp *.h /usr/local/include/
