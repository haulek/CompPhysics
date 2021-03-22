C++ = mpicxx
CFLAGS =  -D_MPI -Ofast -std=c++11 -fPIC -fno-strict-aliasing -fno-common -dynamic -arch x86_64 -arch x86_64 -Wno-deprecated-declarations
LIBS = -L/usr/local/lib -lgsl -lgslcblas -framework Accelerate
BLITZ = -I/usr/local/include -I./cmn

highd_metropolis : highd_metropolis.cc
	$(C++) $(CFLAGS) $(BLITZ) highd_metropolis.cc -o highd_metropolis $(LIBS)
clean:
	rm -f highd_metropolis

