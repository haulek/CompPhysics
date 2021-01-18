#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
using namespace std;

double f(double x){
  return 4.0/(1.0+x*x);
}
double calcPi(int n)
{
  const double dx = 1.0/n;
  double fSum = 0.0;
  #pragma omp parallel for reduction(+:fSum)
  for (int i=0; i<n; ++i){
    double x = (i+0.5)*dx;
    fSum += f(x);
  }
  return fSum*dx;
}


double calcPi_bad(int n)
{
  const double dx = 1.0/n;
  double fSum = 0.0;
  #pragma omp parallel for
  for (int i=0; i<n; ++i){
    double x = (i+0.5)*dx;
    double df = f(x);
    #pragma omp critical
    fSum += df;
  }
  return fSum*dx;
}


int main()
{
  int n=1000000;

  clock_t startTimec = clock();
  double start = omp_get_wtime();

  double fpi = calcPi(n);
  //double gpi = calcPi_bad(n);
  
  clock_t endTimec = clock();
  double diffc = double(endTimec-startTimec)/CLOCKS_PER_SEC;
  double diff = omp_get_wtime()-start;
  clog<<"clock time : "<<diffc<<"s"<<" with wall time="<<diff<<"s "<<endl;

  std::cout << "At n=" << n << " approximation for pi= " << fpi << " with error= " << std::abs(fpi-M_PI) << std::endl; //<< " " << gpi << std::endl;
}
