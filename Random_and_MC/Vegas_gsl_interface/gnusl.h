#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

// Default random number generator to be used
// in gnu scientific library
class RanDef{
  gsl_rng* random;
public:
  RanDef(int iseed=0){
    gsl_rng_env_setup();                     // To setup random number generator
    const gsl_rng_type* T = gsl_rng_default; // default random number generator is set-up
    random = gsl_rng_alloc(T);   // and allocated, default generator is mt19937 generator with a seed of 0
    gsl_rng_set(random, iseed);
  }
  ~RanDef(){
    gsl_rng_free(random);
  }
  gsl_rng* operator()(){
    return random;
  }
  double rand(){
    return gsl_rng_uniform(random);
  }
};

// Wrapper class to be used with gnu scientific library
// to perform plain Monte Carlo multidimensional integration
class MontePlain{
  gsl_monte_function G;
  gsl_monte_plain_state* MCdata;
  int ndim;
public:
  MontePlain(double (*MCfunction) (double*,size_t, void*), int ndim_, void* G_params=NULL) : ndim(ndim_)
  {
    G.f = MCfunction;
    G.dim = ndim_;
    G.params = G_params;
    MCdata = gsl_monte_plain_alloc(ndim); // Allocated the rest of data for integration
  }
  ~MontePlain(){// Deallocates data for MC integration
    gsl_monte_plain_free(MCdata);
  }    
  template <class container, class RanGen>
  double Integrate(const container& xl_, const container& xu_, RanGen& random, size_t calls, double& err){
    if (xl_.size()!=ndim || xu_.size()!=ndim) std::cerr<<"Sizes not correct in Integrate MC"<<std::endl;
    // need to change containers to c arrays (double []).
    double* xl = new double[ndim];
    double* xu = new double[ndim];
    for (int i=0; i<ndim; i++) {xl[i] = xl_[i]; xu[i] = xu_[i];}
    double res;
    gsl_monte_plain_integrate(&G, xl, xu, ndim, calls, random(), MCdata, &res, &err);
    delete[] xl;
    delete[] xu;
    return res;
  }
};

// Wrapper class to be used with gnu scientific library
// to perform Misser Monte Carlo multidimensional integration
class MonteMiser{
  gsl_monte_function G;
  gsl_monte_miser_state *MCdata;
  int ndim;
public:
  MonteMiser(double (*MCfunction) (double*,size_t, void*), int ndim_, void* G_params=NULL) : ndim(ndim_)
  {
    G.f = MCfunction;
    G.dim = ndim_;
    G.params = G_params;
    MCdata = gsl_monte_miser_alloc(ndim); // Allocated the rest of data for integration
  }
  ~MonteMiser()
  { gsl_monte_miser_free (MCdata); }    // Deallocates data for MC integration
  template <class container, class RanGen>
  double Integrate(const container& xl_, const container& xu_, RanGen& random, size_t calls, double& err){
    if (xl_.size()!=ndim || xu_.size()!=ndim) std::cerr<<"Sizes not correct in Integrate MC"<<std::endl;
    double* xl = new double[ndim];
    double* xu = new double[ndim];
    for (int i=0; i<ndim; i++) {xl[i] = xl_[i]; xu[i] = xu_[i];}
    double res;
    gsl_monte_miser_integrate(&G, xl, xu, ndim, calls, random(), MCdata, &res, &err);
    delete[] xl;
    delete[] xu;
    return res;
  }
};

// Wrapper class to be used with gnu scientific library
// to perform Vegas Monte Carlo multidimensional integration
class MonteVegas{
  gsl_monte_function G;
  //  gsl_rng* random;
  gsl_monte_vegas_state *MCdata;
  int ndim;
public:
  MonteVegas(double (*MCfunction) (double*,size_t, void*), int ndim_, void* G_params=NULL) : ndim(ndim_){
    G.f = MCfunction;
    G.dim = ndim_;
    G.params = G_params;
    MCdata = gsl_monte_vegas_alloc(ndim); // Allocated the rest of data for integration
  }
  ~MonteVegas(){
    gsl_monte_vegas_free (MCdata);
  }    // Deallocates data for MC integration
  template <class container, class RanGen>
  double Integrate(const container& xl_, const container& xu_, RanGen& random, size_t calls, double& err){
    if (xl_.size()!=ndim || xu_.size()!=ndim) std::cerr<<"Sizes not correct in Integrate MC"<<std::endl;
    double *xl = new double[ndim], *xu = new double[ndim];
    for (int i=0; i<ndim; i++) {xl[i] = xl_[i]; xu[i] = xu_[i];}
    double res;
    gsl_monte_vegas_integrate(&G, xl, xu, ndim, calls, random(), MCdata, &res, &err);
    delete[] xl;
    delete[] xu;
    return res;
  }
  double chisq(){return MCdata->chisq;}
};
