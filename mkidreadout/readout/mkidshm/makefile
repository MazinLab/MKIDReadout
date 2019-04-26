libmkidshm.so:
	gcc -shared -g -O0 -o libmkidshm.so -fPIC mkidshm.c -lrt -lpthread

clean: 
	$(RM) *.o *.so pymkidshm.c

install: libmkidshm.so mkidshm.h
	cp *.so /usr/local/lib/
	cp *.h /usr/local/include/
