CC = g++-12 
F90 = gfortran
CFLAGS = -O3 -fopenmp

all : mandc mandf

mandc : mandc.cc
	$(CC) $(CFLAGS) -o $@ $<

mandf : mandf.f90
	$(F90) $(CFLAGS) -o $@ $<

clean :
	rm -f mandc mandf

