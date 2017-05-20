CC=icc
CFLAGS=-O3 -I./libxsmm/include -xHost -qopenmp
LDFLAGS=-L./libxsmm/lib -lxsmm -lxsmmext -lpthread

default: bench 
./libxsmm/include/libxsmm.h:
	rm -rf libxsmm/
	git clone https://github.com/hfp/libxsmm.git
	$(MAKE) realclean -C libxsmm
	$(MAKE) BLAS=0 -C libxsmm

bench: bench.c ./libxsmm/include/libxsmm.h
	$(CC) $(CFLAGS) bench.c $(LDFLAGS) -o bench

clean: 
	rm -rf bench
