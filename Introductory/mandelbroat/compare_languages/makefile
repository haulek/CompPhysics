CC = g++-10
F90 = gfortran

all : mandc mandf

mandc : mandc.cc
	$(CC) -O3 -fopenmp -o $@ $<

mandf : mandf.f90
	$(F90) -O3 -fopenmp -o $@ $<

clean :
	rm -f mandc mandf

