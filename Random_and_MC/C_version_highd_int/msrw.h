namespace bl = blitz;
using namespace std;

#ifndef MY_UTIL

template<typename T>
inline T sqr(const T& x){ return x*x;}

inline double ipower(double base, int exp)
{
  switch (exp){
  case 0 : return 1.; break;
  case 1 : return base; break;
  case 2 : return base*base; break;
  case 3 : return base*base*base; break;
  default :
    if (exp<0){
      exp = -exp;
      base = 1./base;
    }
    double result = 1;
    while (exp){
      if (exp & 1) result *= base;
      exp >>= 1;
      base *= base;
    }
    return result;
  }
}
//********** TinyVector summation and subtraction **************/
bl::TinyVector<double,3> operator-(const bl::TinyVector<double,3>& a, const bl::TinyVector<double,3>& b)
{ return bl::TinyVector<double,3>(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}
bl::TinyVector<double,3> operator+(const bl::TinyVector<double,3>& a, const bl::TinyVector<double,3>& b)
{ return bl::TinyVector<double,3>(a[0]+b[0],a[1]+b[1],a[2]+b[2]);}
bl::TinyVector<double,3> operator*(const bl::TinyVector<double,3>& a, double c)
{return bl::TinyVector<double,3>(a[0]*c,a[1]*c,a[2]*c);}

template<typename T, int N>
inline double norm2(const bl::TinyVector<T,N>& vector)
{
  double sum = 0.0;
  for (int i=0; i < N; ++i){
    sum += vector[i]*vector[i];
  }
  return sum;
}

double pi = M_PI;
inline bool isPowerOfTwo (int x)
{
  /* First x in the below expression is for the case when x is 0 */
  return x && (!(x&(x-1)));
}
inline double romberg(const bl::Array<double,1>& ff, double dh)
{
  int m = ff.extent(0);
  if ( !isPowerOfTwo(m-1) ) std::cout<<"ERROR : for Romberg method the size of the array should be 2^k+1" << std::endl;
  int n=m-1, Nr=0;
  while (n!=1){
    n = n/2;
    Nr++;
  }
  int N=Nr+1;

  bl::Array<double,1> h(N+1);
  bl::Array<double,2> r(N+1,N+1);
  
  for (int i = 1; i < N + 1; ++i) {
    h(i) = dh / ( 1<<(i-1) ) ;
  }
  r(1,1) = h(1) / 2 * (ff(0) + ff(m-1));
  for (int i = 2; i < N + 1; ++i) {
    double coeff = 0;
    int dr =  1 << (N-i);
    for (int k = 1; k <= (1<<(i-2)); ++k) coeff += ff((2*k-1)*dr);
    r(i,1) = 0.5 * (r(i-1,1) + h(i-1)*coeff);
  }
  for (int i = 2; i < N + 1; ++i) {
    for (int j = 2; j <= i; ++j) {
      r(i,j) = r(i,j - 1) + (r(i,j-1) - r(i-1,j-1))/( (1<<(2*(j-1))) -1 );
    }
  }
  return r(N,N);
}


#endif


class RanGSL{
  const gsl_rng_type *T;
  gsl_rng *r;
public:
  RanGSL(int idum)
  {
    gsl_rng_env_setup();
    T = gsl_rng_ranlux389;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, idum);
  }
  double operator()()
  { return gsl_rng_uniform (r);}
  ~RanGSL()
  {gsl_rng_free (r);}
};


class CumulativeInt{
public:
  bl::Array<double,1> h;
  bl::Array<double,1> Ih;
  double dh, Iall;
  CumulativeInt(const bl::Array<double,1>& _h_, double _dh_) : dh(_dh_), h(_h_.extent(0)), Ih(_h_.extent(0))
  {
    int Nbin = _h_.extent(0);
    for (int i=0; i<Nbin; ++i) h(i) = _h_(i);
    double cum = 0;
    Ih = 0;
    for (int i=0; i<Nbin; ++i){
      Ih(i) = cum;
      cum += h(i)*(i+0.5)*dh*dh;
    }
    Iall = cum;
  }
  double operator()(double x)
  {
    int ii = x/dh;
    if (ii >= Ih.extent(0)) return Iall;
    return Ih(ii) + h(ii)*0.5*(x*x - ii*dh*ii*dh);
  }
};

class meassureWeight{
protected:
  /*
   The operator() returns the value of a meassuring diagram, which is a function that we know is properly normalized to unity.

   The assumed form for the normalized function is:
      (\Product_{i} g(k_i) ) (\sum_j (\Product_{i\ne j} f(k_i-k_j) ) )
     
   Note here that the convolutions are such that every pair of i,j enters into this expression, and no j is prefered.
   When self-consistently optimized, g(k_i) is proportional to projection of the actual function to corresponding momentum
   and f(k_i-k_j) is proportional to the correlation between momentum k_i and k_j.

   We start with the flag self_consistent=0, in which case we use a simple function : 
                 f0(k) = theta(k<kF) + theta(k>kF) * (kF/k)^dexp
   Notice that f0(k) needs to be normalized so that \int f0(k) 4*pi*k^2 dk = 1.

   If given a histogram from previous MC data, and after call to Recompute(), it sets self_consistent=1, in which case we use
   separable approximation for the integrated function. Namely, if histogram(k) ~ h(k), then f1(k) ~ h(k)/k^2 for each momentum variable.
   We use linear interpolation for function g(k) and we normalize it so that \int g(k)*4*pi*k^2 dk = 1. 
   Notice that when performing the integral, g(k) should be linear function on the mesh i*dh, while the integral 4*pi*k^2*dk should be perform exactly.
   */
  int dexp;
  double kF, cutoff, integral0, dh;
  int self_consistent, Nbin;
  bl::Array<double,2> gx;
  int i_first_momentum;
  int Nloops;
  int Noff;  // Number of off-diagonal terms included, such as R(|r_5-r_4|)*R(|r_5-r_3|)*R(|r_5-r_2|)*...
  bl::Array<double,3> gx_off2;
public:
  bl::Array<double,3> K_hist2;
  const bool OFFD = true;
public:
  meassureWeight(double _dexp_, double _cutoff_, double _kF_, int _Nbin_, int _Nloops_, int _i_first_momentum_=1):
    dexp(_dexp_), kF(_kF_), cutoff(_cutoff_),
    Nbin(_Nbin_), i_first_momentum(_i_first_momentum_), Nloops(_Nloops_)
  {
    integral0 =  (4*pi*ipower(kF,3)/3.) * ( 1 + 3./(dexp-3.) * ( 1 - ipower(kF/cutoff,dexp-3) ) );
    self_consistent=0;
    dh = cutoff/Nbin;
    K_hist2.resize(Nloops,Nloops,Nbin);
    K_hist2=0;
    if (OFFD){
      Noff = Nloops-1-i_first_momentum;
    }else{
      Noff = 0;
    }
  }
  template<typename real>
  double f0(real x){
    if (x>kF) return ipower(kF/x,dexp)/integral0;
    else return 1./integral0;
  }
  template<typename real>
  double f1(real x, int ik){
    const double small=1e-16;
    // we will use discrete approximation for gx(ik, x)
    int ip = static_cast<int>(x/dh);
    if (ip>=Nbin) return 0;
    double res = gx(ik,ip);
    return res > small ? res : small;
  }
  double f1_off(double x, int isk, int ik){
    const double small=1e-16;
    int ip = static_cast<int>(x/dh);
    if (ip>=Nbin) return 0;
    double res = gx_off2(isk,ik,ip);
    return res > small ? res : small;
  }
  template<typename real>
  double operator()(const bl::Array<bl::TinyVector<real,3>,1>& momentum)
  {
    double PQ_new = 1.0;
    if (! self_consistent){
      for (int ik=i_first_momentum; ik<momentum.extent(0); ik++){
	PQ_new *= f0( norm(momentum(ik)) );
      }
    }else{
      for (int ik=i_first_momentum; ik<momentum.extent(0); ik++)
	PQ_new *= f1( norm(momentum(ik)), ik);
      // example : loops_order={0,2,3,4,5}; loops_special=1;
      // we store: K_hist=[ k0-k1, k2-k1, k3-k1, k4-k1, k5-k1]
      if (OFFD){
	double Psm=0;
	for (int isk=i_first_momentum; isk<Nloops; ++isk){
	  double P = 1.0;
	  for (int ik=i_first_momentum; ik<Nloops; ++ik){
	    if (ik!=isk){
	      bl::TinyVector<real,3> dk = momentum(ik)-momentum(isk);
	      double ndk = norm(dk);
	      P *= f1_off( ndk, isk, ik );
	    }
	  }
	  Psm += P;
	}
	PQ_new *= (Psm * 1.0/(Nloops-i_first_momentum));
      }
    }
    return PQ_new;
  }
  // We do not literly take histogram from MC, but we transform it slightly, so that large
  // peaks become less large, and small values are less small.
  double trs(double x){ return fabs(x);}
  double intg3D(const bl::Array<double,1>& f, double dh){
    // integrates picewise constant function (which is assumed to be constant in each interval),
    // but with weight 4*pi*k^2 f(k) dk
    double dsum=0;
    for (int i=0; i<f.extent(0); i++) dsum += f(i)*((i+1)*(i+1)*(i+1)-i*i*i);
    return dsum * dh*dh*dh * 4*pi/3.;
  }

  bl::Array<double,1> NumericInt3DType1(const bl::Array<double,2>& gx, const bl::Array<double,3>& gx_off, double dh, int i_first_momentum=1, int r=3)
  {
    //  This routine numerically (exactly) integrates the 3D function of the form
    //   It = \sum_{j=1,...N} \int d^3r_j gx[j](r_j)  \product_{i\ne j and i=1}^{N} \int d^3r_i gx[i](r_i) gx_off[j,i](|r_j-r_i|)
    //  in which gx[j] and gx_off[j,i] are pice-wise step functions.
    //
    //  This integral is transformed into the angle and radial integral of the form
    //   It = 4*pi * (2*pi)^(N-1) \sum_{j=1,...N} \int dr_j r_j^(3-N) gx[j](r_j) \product_{i\ne j and i=1}^{N} \int dr_i r_i gx[i](r_i) \int_{|r_i-r_j|}^{r_i+r_j} du u gx_off[j,i](u)
    //
    //  if Noff == N-1, we also can write
    //   It = 4*pi * (2*pi)^Noff \sum_{j=1,...N} \int dr_j r_j^(2-Noff) gx[j](r_j) \product_{i\ne j and i=1}^{N} \int dr_i r_i gx[i](r_i) \int_{|r_i-r_j|}^{r_i+r_j} du u gx_off[j,i](u)
    //
    //  For pice-wise function h(x) == gx_off[j,i](x) we can compute exact integral of the form
    //   I_ci(r) = \int_0^r du u h(u) == \sum_{i; dh*i<r} dh^2 * h[i]*(i+0.5)  + h[i]*0.5*(r^2-(dh*i)^2)
    //  If we precompute cumulative sum
    //      C[j] = \sum_{j<i} h[i]*(i+0.5)*dh^2
    //  we see
    //   I_ci(r) = C[i] + h[i]*0.5*(r^2-(dh*i)^2) , where i = floor(r/dh)
    //  We also know that I_ci(r) is cubic polynomial in each interval, hence Simpson's rule is exact integral for such function over each interval.
    // Hence integral over single bin of the form
    //     \int dr_i r_i gx[i](r_i) \int_{|r_i-r_j|}^{r_i+r_j} du u gx_off[j,i](u)
    // where gx[i](r_i) is constant in the interval, can be computed with a single Simpson's rule, and needs only function at the midpoint and at the end-points.
    // 
    int Nloops = gx.extent(0); // This is N=Nloops-1 in the above equation
    int Nbin = gx.extent(1);   // number of bins
    int Noff = Nloops-1- i_first_momentum; // Number of off-diagonal functions Noff above
    int Nr = ipower(2,r)+1;                // How many points is used for romberg integration below. It needs to be exact for polynomial of correct degree.
  
    bl::Array<double,1> ri(3);             // will contain r at the beginning, midpoint and end of each interval
    bl::Array<double,2> vi(3,Nr);          // temporary arrays contains values at the three points
    bl::Array<double,1> Ps(Nr), Pt(Nr), Pti(Nr);
    bl::Array<double,1> rj(Nr);

    bl::Array<double,1> Integ(Nloops);
    Integ=0;
    for (int isk=i_first_momentum; isk<Nloops; isk++){ // variable j in the above equation
      double dsum = 0.0;
      for (int i5=0; i5<Nbin; ++i5){                        // bins corresponding to variable j
	double rn = (i5+0.5)*dh;                            // r_j at the midpoint of each interval, which is between i5*dh and (i5+1)*dh
	for (int l=0; l<Nr; ++l) rj(l) = dh*(i5+l/(Nr-1.)); // more precise mesh (for romberg integration) of variable r_j, i.e., the last integral in the above equation
	if (i5==0) rj(0) = 1e-15; // should not be zero.    Point r_j==0 is singular, and should be avoided
	Pt(bl::Range::all()) = 1.0;                         // Will contain the product of functions \product_{i\ne j and i=1}^{N} .....
	for (int ik=i_first_momentum; ik<Nloops; ++ik){     // Here we compute the product i\ne j, and we use i==ik and j==isk
	  if (ik!=isk){
	    CumulativeInt I_ci(gx_off(isk,ik,bl::Range::all()), dh); // This class calculates exact integral of picewise function gx_off[j,i](u)*u at an arbitrary point. See discussion on I_ci above.
	    ri(2) = 0;
	    vi(2,bl::Range::all()) = 0;
	    Ps = 0;                      // Will contain the product of functions \product_{i\ne j and i=1}^{N}
	    for (int j=0; j<Nbin; j++){
	      //ri(0) = j*dh;
	      ri(0) = ri(2); // j*dh;    // ri(0) and ri(2) are the end-points of the interval of variable r_i, and ri(1) is the midpoint. We will use Simpson's rule, which only needs midpoint.
	      vi(0,bl::Range::all()) = vi(2,bl::Range::all()); // previous value is the value at the end-point
	      ri(2) = (j+1)*dh;
	      ri(1) = 0.5*(ri(0)+ri(2)); // midpoint for Simpson's
	      for (int l=0; l<Nr; ++l){
		//vi(0,l) = (I_ci(rj(l)+ri(0))-I_ci(fabs(rj(l)-ri(0))))*ri(0);  // We just copied this above, because it was previously computed, as it is the end-point
		vi(2,l) = (I_ci(rj(l)+ri(2))-I_ci(fabs(rj(l)-ri(2))))*ri(2);    // The end-point for next step of function : r_i * \int_{|r_j-r_i|}^{r_j+r_i} du u gx_off[j,i](u)
		vi(1,l) = (I_ci(rj(l)+ri(1))-I_ci(fabs(rj(l)-ri(1))))*ri(1);    // midpoint for the same function r_i * \int_{|r_j-r_i|}^{r_j+r_i} du u gx_off[j,i](u)
	      }
	      Ps(bl::Range::all()) += (vi(0,bl::Range::all())+4*vi(1,bl::Range::all())+vi(2,bl::Range::all()))*dh/6.0 * gx(ik,j); // simpson's rule
	      // Integral over a single interval for the cubic pice-wise function
	      //  \int_{single-interval} dr_i gx[i](r_i) r_i * \int_{|r_j-r_i|}^{r_j+r_i} du u gx_off[j,i](u)
	      // once acumulated over all bins, gives exact integral for the function
	      // Ps = \int dr_i gx[i](r_i) r_i * \int_{|r_j-r_i|}^{r_j+r_i} du u gx_off[j,i](u)
	    }
	    Pt(bl::Range::all()) *= Ps(bl::Range::all()); // Product over all Ps functions. There are Noff products.
	  }
	}
	for (int l=0; l<Nr; ++l){
	  double r2p = ipower(rj(l),2-Noff);
	  Pti(l) = Pt(l)*r2p;  // Finally, preparing values for the last integral over r_j, i.e., \int dr_j r_j^(2-Noff) gx[j](r_j) * Pt(r_j)
	}
	double Pt_Int = romberg(Pti, dh);
	dsum += Pt_Int * gx(isk,i5);
      }
      dsum *= 2 * ipower(2*pi,Noff+1);
      Integ(isk) = fabs(dsum);
    }
    return Integ;
  }
  void Recompute(bool SaveData=true)
  {
    if (K_hist2.extent(2) != Nbin)
      std::cerr << "Wrong dimension of K_hist. Should be " << Nbin << " but is " << K_hist2.extent(2) << std::endl;
    gx.resize(Nloops,Nbin);
    gx=0;
    self_consistent=1;
    // first smoothen the histogram, by averaging over three points. Then transform function so that large variations are removed.
    for (int ik=i_first_momentum; ik<Nloops; ik++){ 
      gx(ik,0) = trs( 0.5*(K_hist2(ik,ik,0)+K_hist2(ik,ik,1)) );
      for (int i=1; i<Nbin-1; i++)
      	gx(ik,i) = trs( (K_hist2(ik,ik,i-1)+K_hist2(ik,ik,i)+K_hist2(ik,ik,i+1))/3. );
      gx(ik,Nbin-1) = trs( 0.5*(K_hist2(ik,ik,Nbin-2)+K_hist2(ik,ik,Nbin-1)) );
      
      double norm = 1/intg3D(gx(ik,bl::Range::all()), dh);
      for (int i=0; i<Nbin; i++) gx(ik,i) *= norm;
    }
    if (OFFD){
      gx_off2.resize(Nloops,Nloops,Nbin);
      gx_off2=0;
      // Off-Diagonal first round of normalization
      for (int isk=0; isk<Nloops; isk++){
	for (int ik=0; ik<isk; ik++){
	  gx_off2(isk,ik,0) = trs( 0.5*(K_hist2(isk,ik,0)+K_hist2(isk,ik,1)) );
	  gx_off2(ik,isk,0) = gx_off2(isk,ik,0);
	  for (int i=1; i<Nbin-1; i++){
	    gx_off2(isk,ik,i) = trs( (K_hist2(isk,ik,i-1)+K_hist2(isk,ik,i)+K_hist2(isk,ik,i+1))/3. );
	    gx_off2(ik,isk,i) = gx_off2(isk,ik,i);
	  }
	  gx_off2(isk,ik,Nbin-1) = trs( 0.5*(K_hist2(isk,ik,Nbin-2)+K_hist2(isk,ik,Nbin-1)) );
	  gx_off2(ik,isk,Nbin-1) = gx_off2(isk,ik,Nbin-1);
	}
      }
      for (int isk=0; isk<Nloops; isk++){
	for (int ik=0; ik<isk; ik++){
	  double sm = sum(gx_off2(isk,ik,bl::Range::all()));
	  double norm=1;
	  if (sm!=0){
	    norm = 1./sm;
	  }else{
	    cout<<"ERROR : The sum is zero, which should not happen!"<<endl;
	  }
	  gx_off2(isk,ik,bl::Range::all()) *= norm; // We make this off_diagonal function normalized so that each value is on average 1.0
	  gx_off2(ik,isk,bl::Range::all()) *= norm; // We make this off_diagonal function normalized so that each value is on average 1.0
	}
      }
    }
    
    // We need to calculate the following integral
    //
    //  Norm = \int d^3r1...d^3r5  g5(r5) * g1(r1)*g15(|\vr1-\vr5|) * g2(r2)*g25(|\vr2-\vr5|) * ...* g4(r4)*g45(|\vr4-\vr5|)
    // 
    //  This can be turned into radial and angle integral. The important property is that angle between \vr_i-\vr_j| appears in a single term
    //  hence each term can be independetly integrated over phi, and over cos(theta_{ij}) = x_i
    //  We get the following result
    //
    //  Norm = 2*(2*pi)^5 Integrate[ r5^{2-4}*g5(r5) *
    //                               * Integrate[ r1*g1(r1)*u*g15(u) , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
    //                               * Integrate[ r2*g2(r2)*u*g25(u) , {r2,0,cutoff}, {u, |r2-r5|, r2+r5} ]
    //                               * Integrate[ r3*g3(r3)*u*g35(u) , {r3,0,cutoff}, {u, |r3-r5|, r3+r5} ]
    //                               * Integrate[ r4*g4(r4)*u*g45(u) , {r4,0,cutoff}, {u, |r4-r5|, r4+r5} ]
    //                               , {r5, 0, cutoff}]
    //
    //     In the above function "Normalization_X1_X2" we compute polynomial for
    //
    //     F( r5=(i5,t) ) = Integrate[ r1 g_diag(r1) g_off(u) u , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
    //
    //     where g_diag(r1) and g_off(u) are picewise constant functions on the equidistant mesh i*Dlt (i=0,1,2,,,N-1)
    //     Here
    //            r1 = i1 + x, with i1 integer and x=[0,1]
    //            r5 = i5 + t, with i5 integer and t=[0,1]
    //     and therefore
    //
    //     F(i5,t) = \sum_{i1} Integrate[ (i1+x) g_diag[i1]  Integrate[ g_off(u) u , {u, |i5-i1+t-x|, i5+i1+t+x}], {x,0,1}]
    //  The normalization is than
    //
    //  Norm = (4*pi)^5/2^4 Integrate[ r5^{2-4}*g5(r5) * F_{g1}(i5,t) * F_{g2}(i5,t) * F_{g3}(i5,t) * F_{g4}(i5,t), {r5, 0, cutoff}]
    //
    //  Norm = (4*pi*Dlt)^5/2^4 \sum_{i5} Integrate[ (i5+t)^{2-4} * g5[i5] * F_{g1}(i5,t) * F_{g2}(i5,t) * F_{g3}(i5,t) * F_{g4}(i5,t), {t,0,1}]
    //
    if (OFFD){
      bool NUMERIC=true;
      bl::Array<double,1> nmc(Nloops);
      nmc = NumericInt3DType1(gx, gx_off2, dh, i_first_momentum);
      for (int isk=i_first_momentum; isk<Nloops; isk++){
	double nrm = 1./std::pow(nmc(isk),1./Noff);
	for (int ik=i_first_momentum; ik<Nloops; ++ik)
	  gx_off2(isk,ik,bl::Range::all()) *= nrm;
      }
    }
    if (SaveData){
      for (int ik=i_first_momentum; ik<Nloops; ik++){
	std::ofstream out( (std::string("measure_weight.")+to_string(ik)).c_str() );
	out << "# Nloops=" << Nloops << " Noff=" << Noff << endl;
	for (int i=0; i<Nbin; i++){
	  double k=dh*(i+0.5);
	  double dV = 4*pi*k*k;
	  out<< dh*(i+0.5) << " "<< gx(ik,i)/*dV*/ << " "<<f0(dh*(i+0.5))/*dV*/ <<endl;
	}
	out.close();
      }
    }
    if (SaveData and OFFD){
      for (int isk=i_first_momentum; isk<Nloops; isk++){
	for (int ik=i_first_momentum; ik<Nloops; ik++){
	  if (isk!=ik){
	    std::ofstream out( (std::string("dmeasure_weight.")+to_string(isk)+"."+to_string(ik)).c_str() );
	    out << "#  kF=" << kF << endl;
	    for (int i=0; i<Nbin; i++){
	      double k=dh*(i+0.5);
	      double dV = 4*pi*k*k;
	      out<< dh*(i+0.5) << " "<< gx_off2(isk,ik,i)/*dV*/ <<endl;
	    }
	    out.close();
	  }
	}
      }
    }
  }
  template<typename real>
  void Add_to_K_histogram(double dk_hist, const bl::Array<bl::TinyVector<real,3>,1>& momentum, double cutoffq, double cutoffk)
  {
    if (i_first_momentum>0){
      double Q = norm(momentum(0));
      // external variable histogram
      if (Q<cutoffq){
	int iik = static_cast<int>(Q/cutoffq * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist2(0,0,iik) += dk_hist;
      }
    }
    // histogram of other momenta, which we integrate over
    for (int ik=i_first_momentum; ik<Nloops; ik++){
      double k = norm(momentum(ik));
      if (k<cutoffk && k>1e-150){
	int iik = static_cast<int>(k/cutoffk * Nbin);
	if (iik>=Nbin) iik=Nbin-1;
	K_hist2(ik,ik,iik) += dk_hist/(k*k);
      }
    }
    for (int i_specialk=0; i_specialk<Nloops; i_specialk++){
      for (int ik=0; ik<i_specialk; ik++){
	bl::TinyVector<real,3> dkv = momentum(ik) - momentum(i_specialk);
	double dk = norm(dkv);
	if (dk < cutoffk && dk>1e-150 ){
	  int iik = static_cast<int>(dk/cutoffk * Nbin);
	  if (iik>=Nbin) iik=Nbin-1;
	  double dd = dk_hist/(dk*dk);
	  K_hist2(i_specialk,ik,iik) += dd;
	  K_hist2(ik,i_specialk,iik) += dd;
	}
      }
    }
  }
  double Normalize_K_histogram(){
    // We can get overflow during MPI for a long run. We should use a constant for normalization.
    // We could here normalize K_hist (with knorm), and than when we measure, we would
    // add instead of adding unity, we would add 1./knorm
    double dsum=0;
    for (int ik=0; ik<Nloops; ik++)
      for (int i=0; i<Nbin; i++)
	dsum += K_hist2(ik,ik,i);
    double dnrm = Nloops/dsum;
    if (std::isnan(dnrm)) cout<<"ERROR : Normalize_K_histogram encounter nan"<<endl;
    K_hist2 *= dnrm;
    return dnrm;
  }
  void WriteStatus(ofstream& out)
  {
    out << "# gx["<<Nloops<<","<<Nbin<<"]=" << endl;
    for (int i=0; i<Nbin; i++){
      for (int ik=0; ik<Nloops; ik++)
	out << gx(ik,i) << " ";
      out << endl;
    }
    if (OFFD){
      out << "# gx_off2["<<Nloops-i_first_momentum<<","<<Nloops-i_first_momentum<<","<<Nbin<<"]=" << endl;
      for (int i=0; i<Nbin; i++){
	for (int isk=i_first_momentum; isk<Nloops; isk++)
	  for (int ik=i_first_momentum; ik<Nloops; ik++)
	    out << gx_off2(isk,ik,i) << " ";
	out << endl;
      }
    }
    out << "# K_hist2["<<K_hist2.extent(0)<<","<<K_hist2.extent(1)<<","<<K_hist2.extent(2)<<"]=" << endl;
    for (int i=0; i<Nbin; ++i){
      for (int ik=0; ik<K_hist2.extent(0); ++ik)
	for (int jk=0; jk<K_hist2.extent(1); ++jk)
	  out << K_hist2(ik,jk,i) << " ";
      out << endl;
    }
  }
  bool ReadStatus(ifstream& inp){
    self_consistent=0; // not yet read
    std::string line;
    std::getline (inp,line); // comment gx[Nloops,Nbin]=
    if (! inp) return false;
    //cout << "line0=" << line << endl;
    gx.resize(Nloops,Nbin);
    for (int i=0; i<Nbin; i++){
      for (int ik=0; ik<Nloops; ik++){
	long double x;
	inp >> x;
	gx(ik,i) = x;
	if (!inp){
	  cout << "WARN: reading status file and could not read gx" << endl;
	  return false;
	}
      }
      std::getline (inp,line);
    }
    if (OFFD){
      std::getline (inp,line); // comment gx_off2[Nloops-1,Nbin]=
      gx_off2.resize(Nloops,Nloops,Nbin);
      gx_off2=0;
      for (int i=0; i<Nbin; i++){
	for (int isk=i_first_momentum; isk<Nloops; isk++){
	  for (int ik=i_first_momentum; ik<Nloops; ik++){
	    long double x;
	    inp >> x;
	    gx_off2(isk,ik,i) = x;
	    if (!inp){
	      cout << "WARN: reading status file and could not read gx_off" << endl;
	      return false;
	    }
	  }
	}
	std::getline (inp,line);
      }
    }
    std::getline (inp,line); // comment K_hist2[,Nbin]=
    K_hist2.resize(Nloops,Nloops,Nbin);
    for (int i=0; i<Nbin; ++i){
      for (int ik=0; ik<K_hist2.extent(0); ++ik){
	for (int jk=0; jk<K_hist2.extent(1); ++jk){
	  long double x;
	  inp >> x;
	  if (std::isnan(x)) x=0;
	  K_hist2(ik,jk,i) = x;
	  if (!inp){
	    cout << "WARN: reading status file and could not read K_hist" << endl;
	    return false;
	  }
	}
      }
      std::getline (inp,line);
    }
    self_consistent=1;
    cout << "mweight.status successfully read" << endl;
    return true;
  }
};

