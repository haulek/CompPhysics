
madel.so : mandel.f90
	f2py -c $? --f90flags='-fopenmp' -m mandel

clean:
	rm -r mandel*.so
