C++ = g++
CFLAGS = -O3
LIBS = -lgsl -lgslcblas

all : vegas 

vegas : vegas.o
	g++ $(CFLAGS) -o $@ $? $(LIBS)

clean :
	rm -f vegas.o vegas 


.SUFFIXES : .cc
.cc.o:  
	$(C++) $(CFLAGS) -c $<

.SUFFIXES : .f
.f.o:
	$(F77) $(FFLAGS) -c $<

