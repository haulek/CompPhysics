#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include "gnusl.h"

// Computs the integral,
// I = int (dx dy dz)/(2pi)^3  1/(1-cos(x)cos(y)cos(z))
// over (-pi,-pi,-pi) to (+pi, +pi, +pi).  The exact answer
// is Gamma(1/4)^4/(4 pi^3).  This example is taken from
// C.Itzykson, J.M.Drouffe, "Statistical Field Theory -
// Volume 1", Section 1.1, p21, which cites the original
// paper M.L.Glasser, I.J.Zucker, Proc.Natl.Acad.Sci.USA 74
//  1800 (1977) 
// For simplicity we compute the integral over the region 
//   (0,0,0) -> (pi,pi,pi) and multiply by 8 
double g (double *k, size_t dim, void *params)
{
  double A = 1.0 / (M_PI * M_PI * M_PI);
  return A/(1.0 - cos(k[0])*cos(k[1])*cos(k[2]));
}

void display_results (const std::string& title, double result, double error)
{
  static const double exact = 1.3932039296856768591842462603255;
  std::cout<<title<<" =================="<<std::endl;
  std::cout<<"result = "<<result<<std::endl;
  std::cout<<"sigma  = "<<error<<std::endl;
  std::cout<<"exact  = "<<exact<<std::endl;
  std::cout<<"error  = "<<result-exact<<" = "<<fabs(result-exact)/error<<" sigma"<<std::endl;
}


using namespace std;
int main ()
{
  vector<double> rl(3);// lower left corner of the integration region
  vector<double> ru(3);// upper right corner of the integration region
  for (int i=0; i<rl.size(); i++) {rl[i]=0;  ru[i]=M_PI;}
  
  RanDef ran(time(0)); // Our random number generator

  // All three MC integration methods
  MontePlain monte_plain(g,3);
  MonteMiser monte_miser(g,3);
  MonteVegas monte_vegas(g,3);
  
  double error, result;
  size_t calls = 500000;

  result = monte_plain.Integrate(rl, ru, ran, calls, error);
  display_results ("plain", result, error);
  cout<<"Used "<<calls<<" function evaluations"<<endl;
  
  result = monte_miser.Integrate(rl, ru, ran, calls, error);
  display_results ("miser", result, error);
  cout<<"Used "<<calls<<" function evaluations"<<endl;
  
  size_t callv = 10000;
  result = monte_vegas.Integrate(rl, ru, ran, callv, error); // first some warmup
  display_results("vegas warm-up", result, error);
  cout<<"converging..."<<endl;
  do{
    result = monte_vegas.Integrate(rl, ru, ran, calls/5, error);
    cout<<"result = "<<result<<" sigma = "<<error<<" chisq/dof = "<<monte_vegas.chisq()<<endl;
    callv += calls/5;
  } while (fabs (monte_vegas.chisq() - 1.0) > 0.5);
  display_results ("vegas final", result, error);
  cout<<"Used "<<callv<<" function evaluations"<<endl;
  
  return 0;
}
