CC = g++-10
PYBIND = `python3 -m pybind11 --includes` -undefined dynamic_lookup

imanc.so : imanc.cc
	$(CC) $(PYBIND) -O3 -fopenmp -shared -std=c++11 -fPIC imanc.cc -o imanc.so

clean :
	rm imanc.so
