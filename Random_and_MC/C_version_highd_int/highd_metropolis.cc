#include <iostream>
#include <fstream>
#include <cmath>
#include <blitz/array.h>
#include <gsl/gsl_rng.h>
#include "msrw.h"

#ifdef _MPI
#include <mpi.h>
#endif
int mpi_size=1, mrank=0, master=0;


namespace bl = blitz;
using namespace std;

class params{
public:
  double kF;          // typical magitude of the independent variable vector (momentum)
  double cutoff;      // cutoff for the integration.
  double dkF;         // size of the step, the change of the momentum =0.1*kF
  //
  int Nitt;           // total number of MC steps
  int Ncout;          // how often to print
  int Nwarm;          // warmup steps
  int tmeasure;      // how often to measure, every tmeasure steps
  int iseed;          // seed for random number generator
  //  
  int Nbin;           // number of bins for saving the histogram
  double V0norm;      // initial magnitude of the measuring function, i.e., V0*fm, where fm is normalized
  double dexp;        // exponent at the beginning for measuring function, i.e., fm ~ 1/k^(dexp)
  //	  
  int recomputew;     // how often do we check if V0 is appropriate 
  int per_recompute;  // how often to recompute auxiliary function, counter in recomputew steps
  //
  params(int _Nitt_=50000000, double _kF_=1.0, int _tmeasure_=10) : Nitt(_Nitt_), iseed(0), Ncout(500000), Nwarm(10000), tmeasure(_tmeasure_), recomputew(50000/tmeasure), per_recompute(7),
	     kF(_kF_), Nbin(129), V0norm(0.02), dexp(6.), cutoff(5*kF), dkF(0.1*kF)
  {}
};

inline double ferm(double x){
  if (x>700) return 0.;
  else return 1./(exp(x)+1.);
}

class Linhardt{
public:
  double T;
  double broad; // broadening
  int Ndim;
  double kF2;    // kF^2
  complex<double> i;
  Linhardt (double _T_, double kF, double _broad_) : T(_T_), broad(_broad_), Ndim(2), kF2(kF*kF), i(0.,1.)
  {}
  complex<double> operator()(const bl::Array<bl::TinyVector<double,3>,1>& momentum, double Omega)
  {
    double e_k_q = sqr(norm(momentum(1)-momentum(0)))-kF2;
    double e_k = sqr(norm(momentum(1)))-kF2;
    return -2*(ferm(e_k_q/T)-ferm(e_k/T))/(Omega-e_k_q+e_k+broad*i);
  }
};

class Linhardt_secondOrder{
public:
  int order;
  double T;
  int Ndim;
  double kF, kF2;    // kF^2
  complex<double> i_delta;
  double norm_theta, lmbda;
  Linhardt_secondOrder (int _order_, double _T_, double _kF_, double _lmbda_, double broad) : order(_order_), T(_T_), Ndim(3), kF(_kF_), kF2(kF*kF), lmbda(_lmbda_)
  {
    i_delta = complex<double>(0,1)*broad;
    norm_theta = ipower(2*pi/kF,3)/(4*pi/3);
  }
  complex<double> operator()(const bl::Array<bl::TinyVector<double,3>,1>& momentum, double Omega)
  {
    double k2   = norm2(momentum(1)); // k^2
    double p2   = norm2(momentum(2)); // p^2
    double k_q2 = norm2(momentum(1)+momentum(0)); // (k+q)^2
    double p_q2 = norm2(momentum(2)+momentum(0)); // (p+q)^2
    
    double e_k = k2-kF2;     // e(k)
    double e_p = p2-kF2;     // e(p)
    double e_k_q = k_q2-kF2; // e(k+q)
    double e_p_q = p_q2-kF2; // e(p+q)
    double theta_p = (e_p<0) ? norm_theta : 0;
    double theta_k = (e_k<0) ? norm_theta : 0;
    double f_k   = ferm(e_k/T);                // f(e(k))
    double f_p   = ferm(e_p/T);                // f(e(p))
    double f_k_q = ferm(e_k_q/T);              // f(e(k+q))
    double f_p_q = ferm(e_p_q/T);              // f(e(p+q))
    complex<double> G_k_q_k = 1./(Omega+e_k-e_k_q+i_delta); // G(k+q,w+e(k)) = 1/(w+e(k)-e(k+q)+i*d)
    complex<double> G_p_q_p = 1./(Omega+e_p-e_p_q+i_delta); // G(p+q,w+e(p)) = 1/(w+e(p)-e(p+q)+i*d)
    complex<double> Pk = (f_k_q-f_k)*G_k_q_k; // (f(k+q)-f(k))/(w+e(k)-e(k+q))
    complex<double> Pp = (f_p_q-f_p)*G_p_q_p; // (f(p+q)-f(p))/(w+e(p)-e(p+q))
    complex<double> P0 = -Pk*theta_p -Pp*theta_k; // Symmetrized Linhardt function, i.e., just the first order bubble
    if (order==0){
      return P0;
    }else if (order>0){
      
      double p_k2  = norm2(momentum(2)-momentum(1)); // (p-k)^2
      double V_p_k = 8*pi/(p_k2+lmbda);              // Vc(p-k)
      complex<double> Pladder = -2 * V_p_k * Pk * Pp;
      if (order==1) return P0 + Pladder;          // This is the lowest order ladder diagram
      
      double e_k_m_q   = norm2(momentum(1)-momentum(0)) - kF2;  // e(k-q) = (k-q)^2 -kF^2
      double f_k_m_q   = ferm( e_k_m_q/T);                      // f( e(k-q))
      double f_m_k_m_q = ferm(-e_k_m_q/T);                      // f(-e(k-q))
      double df_dk_m_q = -f_k_m_q*f_m_k_m_q/T;                  // d f(e(k-q))/de(k-q) = -f(e(k-q))*f(-e(k-q))/T
      
      double e_p_m_q = norm2(momentum(2)-momentum(0))-kF2;      // e(p-q) = (p-q)^2-kF^2
      double f_p_m_q = ferm(e_p_m_q/T);                         // f(e(p-q))
      
      double f_m_k_q = ferm(-e_k_q/T);                          // f(-e(k+q))
      double df_k_q = -f_k_q*f_m_k_q/T;                         // df(e(k+q))/de(k+q) = -f(e(k+q))*f(-e(k+q))/T
      
      complex<double> Sp = G_k_q_k*G_k_q_k*(f_k_q-f_k) + G_k_q_k*df_k_q; // G(k+q,w+e(k))^2*(f(k+q)-f(k)) + G(k+q,w+e(k))*df(e(k+q))/de(k+q) = (f(k+q)-f(k))/(w+e(k)-e(k+q)+i*d)^2 - f(-e(k+q))*f(e(k+q))/T * 1/(w+e(k)-e(k+q)+i*d)
      complex<double> G_k_k_m_q = 1./(Omega+e_k_m_q-e_k+i_delta);        // G(k,w+e(k-q)) = 1/(w+e(k-q)-e(k)+i*d)
      complex<double> Sm = G_k_k_m_q*G_k_k_m_q*(f_k_m_q-f_k) - df_dk_m_q*G_k_k_m_q; // G(k,w+e(k-q))^2*(f(k-q)-f(k)) - G(k,w+e(k-q))*df(e(k-q))/de(k-q) = (f(k-q)-f(k))/(w+e(k-q)-e(k)+i*d)^2 - f(-e(k-q))*f(e(k-q))/T * 1/(-w-e(k-q)+e(k)-i*d)
      complex<double> P_HF = 2 * V_p_k * (f_p_q * Sp + f_p_m_q * Sm);
      return P0+Pladder+P_HF;
    }
    return 0;
  }
};
inline double MCW(const bl::TinyVector<complex<double>,2>& fQ)
{
  return abs(fQ[0]) + abs(fQ[1]);
}

template<class Fn>
void Metropolis2(Fn& func, bl::Array<complex<double>,2>& Pval, const params& p, const bl::Array<double,1>& qx, const bl::Array<double,1>& Om)
{
  RanGSL rand(p.iseed); // GSL random number generator
  bool Print = mrank==master;
  if (Print) cout << "Random number = " << rand() << " iseed=" << p.iseed << endl;
  Pval = 0;
  
  double Pnorm = 0.0;
  complex<double> Pval_sum = 0.0;
  double Pnorm_sum = 0.0;
  double V0norm = p.V0norm;
  double dk_hist = 1.0;
  int Ndim = func.Ndim;  // dimensions of the problem
  double per_recompute = p.per_recompute;
  double inc_recompute = (per_recompute+0.52)/per_recompute;

  bl::Array<bl::TinyVector<double,3>,1> momentum(Ndim);
  int iQ = qx.extent(0)*rand();         // iQ is current qx[iQ]
  // Set up initial random state of momenta
  momentum(0) = 0, 0, qx(iQ);
  for (int ik=1; ik<Ndim; ++ik)
    momentum(ik) = rand()*p.kF/sqrt(3.), rand()*p.kF/sqrt(3.), rand()*p.kF/sqrt(3.);
  int iOm = Om.extent(0)*rand();
  
  if (Print) cout << " Initial momentum=" << momentum << endl;
  
  meassureWeight mweight(p.dexp, p.cutoff, p.kF, p.Nbin, Ndim); // fm in alternative space
  
  bl::TinyVector<complex<double>,2> fQ, fQ_new;

  fQ = func(momentum,Om(iQ)), V0norm * mweight( momentum );
  if (Print) cout << "starting with f=" << fQ[0] << "," << fQ[1] << "\nstarting momenta=" <<  momentum << endl;
  cout.precision(5);
  
  int Nmeasure = 0;
  int Nall_q=0, Nall_k=0, Nall_w=0, Nacc_q=0, Nacc_k=0, Nacc_w;
  int c_recompute = 0;
  bl::Array<bl::TinyVector<double,3>,1> tmomentum(momentum.extent(0));
  
  for (int itt=0; itt < p.Nitt; ++itt){
    int iloop = static_cast<int>( (Ndim+1) * rand() ) - 1; // which variable to change
    bl::TinyVector<double,3> K_new;
    bool accept = false;
    int tiQ;
    int tiOm=iOm;
    double trial_ratio = 1;
    double Ka_new;
    if (iloop==-1){ // changing frequency
      Nall_w += 1;
      accept = true;
      tiOm = static_cast<int>( rand()*Om.extent(0) );
    }else if (iloop == 0){  // changing external momentum Q
      Nall_q += 1;
      accept = true;
      { 
	tiQ = static_cast<int>( rand()*qx.extent(0) );
	Ka_new = qx(tiQ);  // new length of the vector
	double th = pi*rand(), phi = 2*pi*rand();
	double sin_th = sin(th);
	double Q_sin_th = Ka_new * sin_th;
	K_new = Q_sin_th*cos(phi), Q_sin_th*sin(phi), Ka_new*cos(th);
	bl::TinyVector<double,3> Qc = momentum(iloop);
	double q2_sin2_old = sqr(Qc[0]) + sqr(Qc[1]);
	double q2_old = q2_sin2_old + sqr(Qc[2]);
	if (q2_old != 0){
	  double sin_th_old = sqrt(fabs(q2_sin2_old/q2_old));
	  if (sin_th_old != 0){
	    trial_ratio = sin_th/sin_th_old;
	  }
	}
      }
    } else {  // changing momentum ik>0
      Nall_k += 1;
      bl::TinyVector<double,3> dk;
      dk = (2*rand()-1)*p.dkF, (2*rand()-1)*p.dkF, (2*rand()-1)*p.dkF;
      K_new = momentum(iloop) + dk;
      Ka_new = norm(K_new);
      accept = Ka_new <= p.cutoff;
    }
    if (accept){ // trial step successful
      tmomentum = momentum;
      if (iloop>=0) tmomentum(iloop) = K_new;
      fQ_new = func(tmomentum,Om(tiOm)), V0norm * mweight(tmomentum);
      double ratio = MCW(fQ_new)/MCW(fQ) * trial_ratio;
      accept = ratio > 1-rand();
      if (accept){  // the step succeeded
	if (iloop>=0) momentum(iloop) = K_new;
	fQ = fQ_new;
	if (iloop==-1){
	  Nacc_w += 1;
	  iOm = tiOm;
	}else if (iloop==0){
	  Nacc_q += 1;
	  iQ = tiQ;
	}else
	  Nacc_k += 1;
      }
    }
    if (itt >= p.Nwarm and itt % p.tmeasure==0){
      Nmeasure += 1;
      double W = MCW(fQ);
      complex<double> f0=fQ[0]/W;
      double f1 = abs(fQ[1])/W;
      double Wphs  = abs(f0);
      Pval(iOm,iQ)  += f0;
      Pnorm     += f1;
      Pnorm_sum += f1;
      Pval_sum  += Wphs;
      mweight.Add_to_K_histogram(dk_hist*Wphs, momentum, p.cutoff, p.cutoff);
      if (itt>100000 and itt % (p.recomputew*p.tmeasure) == 0 ){
	double P_v_P = abs(Pval_sum)/Pnorm_sum * 0.1;
#ifdef _MPI
	double P_v_P_here = P_v_P;
	int ierr = MPI_Allreduce(&P_v_P_here, &P_v_P, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (ierr!=0) { cout << "MPI_Allreduce(P_v_P) returned error="<< ierr << endl; exit(1);}
	P_v_P = P_v_P/mpi_size;
#endif	
	int change_V0 = 0;
	if (P_v_P < 0.25 and itt < 0.1*p.Nitt){
	  change_V0 = -1;
	  V0norm    /= 2;
	  Pnorm     /= 2;
	  Pnorm_sum /= 2;
	}
	if (P_v_P > 4.0 and itt < 0.1*p.Nitt){
	  change_V0 = 1;
	  V0norm    *= 2;
	  Pnorm     *= 2;
	  Pnorm_sum *= 2;
	}
	if (change_V0){
	  string schange = change_V0<0 ? "V0 reduced to " : "V0 increased to ";
	  if (Print) cout << "   " << itt/1e6 << "M P_v_P=" << P_v_P << schange << V0norm;
	  Pval = 0;
	  Pnorm = 0;
	  Nmeasure = 0;
	}
	if (c_recompute==0 and itt<0.5*p.Nitt){
	  per_recompute = per_recompute*inc_recompute;
#ifdef _MPI	  
	  int ierr1 = MPI_Allreduce(MPI_IN_PLACE, mweight.K_hist2.data(), mweight.K_hist2.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  if (ierr1!=0) {clog << "MPI_Allreduce(K_hist) returned error="<< ierr1 << endl; exit(1);}
#endif
	  if (fabs(P_v_P) > 0.01 and itt < 0.5*p.Nitt){
	    if (Print) cout << "Recomputing mweight" << " itt=" << itt << " per_recompute=" << per_recompute << " P_v_P=" << P_v_P << endl;
	    double renormalize = mweight.Normalize_K_histogram();
	    dk_hist *= sqrt(renormalize);
	    if (dk_hist < 1e-8) dk_hist = 1.0;
	    mweight.Recompute();
	    fQ[1] = V0norm * mweight(momentum);
	  }
	}
	c_recompute += 1;
	if (c_recompute>=per_recompute) c_recompute = 0;
      }
    }
    if ( (itt+1) % p.Ncout == 0 ){
      double P_v_P = abs(Pval_sum)/Pnorm_sum * 0.1;
      double Qa = qx(iQ);
      double ka = norm(momentum(1));
      double ratio = MCW(fQ_new)/MCW(fQ);
      if (Print) cout<<right<<setw(6)<<itt/1.0e6<<left<<"M iOm="<<setw(3)<<left<<iOm<<" Q="<<left<<setw(7)<<Qa<<" k="<<setw(7)<<ka<<" fQ_new="<<setw(11)<<left<<fQ_new[0]<<","<<setw(11)<<left<<fQ_new[1]<<" fQ_old="<<setw(11)<<left<<fQ[0]<<","<<setw(11)<<left<<fQ[1]<<" ratio="<<setw(10)<<left<<ratio<<" dk="<<setw(10)<<left<<dk_hist<<" P_v_P="<<setw(10)<<left<<P_v_P<<endl;
    }
  }

#ifdef _MPI
  if (mrank==master){
    MPI_Reduce(MPI_IN_PLACE, Pval.data(),          Pval.size()*2,        MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &Pnorm,               1,                    MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);
  }else{
    MPI_Reduce(Pval.data(),  Pval.data(),          Pval.size()*2,        MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);
    MPI_Reduce(&Pnorm,       &Pnorm,               1,                    MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);
  }
#endif
  
  Pval *= qx.extent(0)*Om.extent(0) * V0norm / Pnorm;

  Pval *= ipower( 1.0/((2*pi)*(2*pi)*(2*pi)) , Ndim-1);
  if (Print){
    cout << "Total acceptance rate=" << (Nacc_k+Nacc_q)/(p.Nitt+0.0) << " k-acceptance=" << Nacc_k/(Nall_k+0.0) << " q-acceptance=" << Nacc_q/(Nall_q+0.0) << endl;
    cout << "k-trials=" << Nall_k/(p.Nitt+0.0) << " q-trial=" << Nall_q/(p.Nitt+0.0) << endl;
  }
}

int main(int argc, char *argv[])
{
#ifdef _MPI  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
  master = 0;
#endif
  
  double rs = 2.;
  double kF = pow( 9*pi/4., 1./3.) / rs;
  double nF = kF/(2*pi*pi);

  double T = 0.02*sqr(kF);
  double lmbda = sqr(1.2*kF);
  double broad = 0.002*sqr(kF);
  int Nitt = 50000000;
  
  params p(Nitt,kF,10);
  p.iseed = time(0) + mrank*10;
  
  Linhardt func(T, p.kF, broad);
  //Linhardt_secondOrder func(1, T, p.kF, lmbda, broad);
  
  bl::Array<double,1> qx(1); qx(0) = 0.098437*kF;
  
  bl::Array<double,1> Om(100);
  for (int i=0; i<Om.extent(0); ++i) Om(i) = 0.5*kF*kF*i/(Om.extent(0)-1.0);
  
  bl::Array<complex<double>,2> Pval( Om.extent(0), qx.extent(0) );
  Metropolis2(func, Pval, p, qx, Om);

  if (mrank==master){
    ofstream fo("pval.dat");
    double EF = kF*kF;
    fo << "# rs=" << rs << " T=" << T << " broad=" << broad << " lmbda=" << lmbda << " nF=" << nF << " kF=" << kF << " EF=" << EF << endl;
    for (int iOm=0; iOm<Om.extent(0); ++iOm){
      fo << Om(iOm)/EF << " ";
      for (int iq=0; iq<qx.extent(0); ++iq)
	fo << Pval(iOm,iq).real()/nF << " " << Pval(iOm,iq).imag()/nF << " ";
      fo << endl;
    }
    cout << "nF=" <<  nF << endl;
  }
#ifdef _MPI  
  MPI_Finalize();
#endif  
  return 0;
}
