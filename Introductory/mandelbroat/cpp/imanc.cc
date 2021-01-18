#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <cstdint>

namespace py = pybind11;
using namespace std;

void mand(py::array_t<double>& data, int Nx, int Ny, int max_steps, const vector<int>& ext)
{
  auto dat = data.mutable_unchecked<2>();
  #pragma omp parallel for
  for (int i=0; i<Nx; i++){
    for (int j=0; j<Ny; j++){
      dat(j,i) = max_steps;
      double x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.);
      double y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.);
      complex<double> z0(x,y);
      complex<double> z=0;
      for (int itr=0; itr<max_steps; itr++){
	if (norm(z)>4.){
	  dat(j,i) = itr;
	  break;
	}
	z = z*z + z0;
      }
      //if (norm(z)<4.) dat(j,i) = max_steps;
    }
  }
}

PYBIND11_MODULE(imanc,m){
  m.doc() = "pybind11 wrap for mandelbrot";
  m.def("mand", &mand);
}
