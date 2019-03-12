packetmonster.so: pypacketmonster.c libpacketmonster.so
	gcc pypacketmonster.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o packetmonster.so -L. -lpacketmonster -lmkidshm -lpthread -lrt

pypacketmonster.c:
	cython packetmonster.pyx -o pypacketmonster.c

libpacketmonster.so:
	gcc -shared -o libpacketmonster.so -fPIC packetmonster.c -lmkidshm -lrt -lpthread

clean: 
	$(RM) *.o *.so pypacketmonster.c

install: packetmonster.so packetmonster.h
	cp *.so /usr/local/lib/
	cp *.h /usr/local/include/
