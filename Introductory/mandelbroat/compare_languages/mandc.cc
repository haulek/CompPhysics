#include <iostream>
#include <complex>
#include <ctime>
#include <vector>
#include <omp.h>
using namespace std;

int Mandelb(const complex<double>& z0, int max_steps)
{
  complex<double> z=0;
  for (int i=0; i<max_steps; i++){
    if (abs(z)>2.) return i;
    z = z*z + z0;
  }
  return max_steps;
}

int main()
{
  const int Nx = 1000;
  const int Ny = 1000;
  int max_steps = 1000;
  double ext[]={-2,1,-1,1};

  vector<int> mand(Nx*Ny);
  clock_t startTimec = clock();
  double start = omp_get_wtime();
  
  #pragma omp parallel for
  for (int i=0; i<Nx; i++){
    for (int j=0; j<Ny; j++){
      double x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.);
      double y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.);
      mand[i*Ny+j] = Mandelb(complex<double>(x,y), max_steps);
    }
  }

  clock_t endTimec = clock();
  double diffc = double(endTimec-startTimec)/CLOCKS_PER_SEC;
  double diff = omp_get_wtime()-start;
  
  clog<<"clock time : "<<diffc<<"s"<<" with wall time="<<diff<<"s "<<endl;
  
  for (int i=0; i<Nx; i++){
    for (int j=0; j<Ny; j++){
      double x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.);
      double y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.);
      cout<<x<<" "<<y<<" "<< 1./mand[i*Ny+j] << endl;
    }
  }
}
