#include <iostream> // for printing to stdout
#include <complex>  // for complex numbers
#include <ctime>    // for measuring time
#include <vector>   // for using array/vector
#include <omp.h>    // openMP and measuing time with openMP
using namespace std;// everything from standard template library will be in global scope
// std::cout and std::endl can be called cout, endl, etc...

int Mandelb(const complex<double>& z0, int max_steps)
{
  complex<double> z=0; // specify type where we start using it. (still need to specify type not like python with auto-typing)
  for (int i=0; i<max_steps; i++){ // specify int type inside the loop, so it is local to the loop. useful for reducing bugs.
    if (abs(z)>2.) return i;
    z = z*z + z0;
  }
  return max_steps;
}

int main()
{
  const int Nx = 1000; // constants are named cost rather than parameter
  const int Ny = 1000;
  int max_steps = 1000; // allowed to change later.
  double ext[]={-2,1,-1,1}; // this specifies fixed array of four numbers and initializes it.

  vector<int> mand(Nx*Ny);  // allocating one dimensional array (vector) of size Nx*Ny
  // multidimensional dynamic arrays are not standard in C++. One can use pointers to allocate/deallocate memory,
  // and write one's own class interface for that. Or one has to use extension of C++ (blitz++ is excellent).
  // Unfortunately the standard C++ still does not support standard class for that.
  clock_t startTimec = clock();  // cpu time at the start 
  double start = omp_get_wtime(); // wall time at the start
  
  #pragma omp parallel for
  for (int i=0; i<Nx; i++){
    for (int j=0; j<Ny; j++){
      double x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.); // x in the interval ext[0]...ext[1]
      double y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.); // y in the interval ext[2]...ext[3]
      mand[i*Ny+j] = Mandelb(complex<double>(x,y), max_steps); // storing values in 2D array using 1D array
    }
  }

  clock_t endTimec = clock();
  double diffc = double(endTimec-startTimec)/CLOCKS_PER_SEC; // how to get seconds from cpu time
  double diff = omp_get_wtime()-start;                       // openMP time is already in seconds
  
  clog<<"clock time : "<<diffc<<"s"<<" with wall time="<<diff<<"s "<<endl; // printout of time
  
  for (int i=0; i<Nx; i++){
    for (int j=0; j<Ny; j++){
      double x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.);
      double y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.);
      cout<<x<<" "<<y<<" "<< 1./mand[i*Ny+j] << endl; // prinout of mandelbrot set
    }
  }
}
