pymkidshm.so: pymkidshm.c libmkidshm.so
	gcc pymkidshm.c -fPIC -shared -I/home/neelay/anaconda2/envs/readout/include/python2.7/ -o pymkidshm.so -lmkidshm -lpthread -lrt

pymkidshm.c:
	cython pymkidshm.pyx -o pymkidshm.c

libmkidshm.so:
	gcc -shared -o libmkidshm.so -fPIC mkidshm.c -lrt -lpthread

clean: 
	$(RM) *.o *.so pymkidshm.c

install: libmkidshm.so mkidshm.h
	cp *.so /usr/local/lib/
	cp *.h /usr/local/include/
