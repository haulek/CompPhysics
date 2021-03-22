#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include "gnusl.h"
using namespace std;

double IntegratingFunction(double* x, size_t dim, void*)
{
 double A = 1.0 / (M_PI * M_PI * M_PI);
 return A/(1.0 - cos(x[0])*cos(x[1])*cos(x[2]));  
}

int main()
{
  RanDef ran(time(0)); // Our random number generator

  MontePlain monte_plain(IntegratingFunction, 3);
  double error;
  size_t calls = 500000;
  vector<double> xl(3), xr(3);
  for (int i=0; i<3; i++) {xl[i]=0; xr[i]=M_PI;}
  
  double result = monte_plain.Integrate(xl, xr, ran, calls, error);
  cout<<"result="<<result<<" error="<<error<<endl;
  
  return 0;
}
