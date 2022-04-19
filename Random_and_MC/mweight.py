from scipy import *
from numpy import *
from numpy import linalg
from scipy import special
from numba import jit

@jit(nopython=True)
def self_f0(x, self_kF, self_dexp, self_integral0):
    if (x > self_kF): 
        return (self_kF/x)**self_dexp/self_integral0
    else:
        return 1./self_integral0
    
@jit(nopython=True)
def self_f1(x, ik, self_dh, self_Nbin, self_gx):
    # we will use discrete approximation for gx(ik, x)
    ip = int(x/self_dh)
    if (ip >= self_Nbin): # outside the mesh for the function
        return 0
    res = self_gx[ik,ip]
    return max(res,1e-16)

@jit(nopython=True)
def self_f1_off(x, ik, self_dh, self_Nbin, self_gx_off):
    # off diagonal h-function
    ip = int(x/self_dh)
    if (ip >= self_Nbin): 
        return 0
    res = self_gx_off[ik,ip]
    return max(res, 1e-16)

@jit(nopython=True)
def self_fm(momentum, self_self_consistent, self_kF, self_dexp, self_integral0, self_dh, self_Nbin, self_gx, self_gx_off):
    PQ_new = 1.0
    if not self_self_consistent:
        for ik in range(1,len(momentum)):
            k = linalg.norm(momentum[ik])
            PQ_new *= self_f0( k, self_kF, self_dexp, self_integral0 )
    else:
        for ik in range(1,len(momentum)):
            k = linalg.norm(momentum[ik])
            PQ_new *= self_f1( k, ik, self_dh, self_Nbin, self_gx )
        for ik in range(1,len(momentum)-1):
            dkq = linalg.norm(momentum[ik]-momentum[-1])
            PQ_new *= self_f1_off( dkq, ik, self_dh, self_Nbin, self_gx_off )
    return PQ_new

def self_Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk, self_K_hist, self_Nbin, self_Nloops):
    Q = linalg.norm(momentum[0])
    # external variable histogram
    if (Q < cutoffq):
        iik = int( Q/cutoffq * self_Nbin )
        self_K_hist[0,iik] += dk_hist
    # histogram of other momenta, which we integrate over
    for ik in range(1,self_Nloops):
        k = linalg.norm(momentum[ik]);
        if k < cutoffk:
            iik = int( k/cutoffk * self_Nbin )
            self_K_hist[ik,iik] += dk_hist/k**2
    # histogram for variable differences. We choose to use
    # the following combination of momenta
    #  |k_0-k_{N-1}|,  |k_1-k_{N-1}|,  |k_2-k_{N-1}|, ...., |k_{N-2}-k_{N-1}|
    for ik in range(self_Nloops-1):
        k = linalg.norm(momentum[ik]-momentum[-1])
        if k < cutoffk :
            iik = int( k/cutoffk * self_Nbin )
            self_K_hist[ik+self_Nloops,iik] += dk_hist/k**2

class meassureWeight:
    """
       The operator() returns the value of a meassuring diagram, which is a function that we know is properly normalized to unity.
       We start with the flag self_consistent=0, in which case we use a simple function : 
                     f0(k) = theta(k<kF) + theta(k>kF) * (kF/k)^dexp
       Notice that f0(k) needs to be normalized so that \int f0(k) 4*pi*k^2 dk = 1.
    
       If given a histogram from previous MC data, and after call to Recompute(), it sets self_consistent=1, in which case we use
       separable approximation for the integrated function. Namely, if histogram(k) ~ h(k), then f1(k) ~ h(k)/k^2 for each momentum variable.
       We use linear interpolation for function g(k) and we normalize it so that \int g(k)*4*pi*k^2 dk = 1. 
       Notice that when performing the integral, g(k) should be linear function on the mesh i*dh, while the integral 4*pi*k^2*dk should be perform exactly.
    """
    def __init__(self, dexp, cutoff, kF, Nbin, Nloops):
        self.dexp = dexp
        self.cutoff = cutoff
        self.kF = kF
        self.Nbin = Nbin
        self.Nloops = Nloops
        # \int f0(k) d^3k, where f0(k) is given above
        self.integral0 =  (4*pi*kF**3/3.) * ( 1 + 3./(dexp-3.) * ( 1 - (kF/cutoff)**(dexp-3)))
        # at the beginning we do not have self-consistent function yet, but just f0
        self.self_consistent = False
        self.Noff = Nloops-2 # how many off-diagonal h functions
        self.dh = cutoff/Nbin
        # History of configurations
        self.K_hist = zeros((2*Nloops-1,Nbin))
        self.gx     = zeros( (self.Nloops,self.Nbin) )
        self.gx_off = zeros( (self.Nloops-1, self.Nbin) )
        
    def f0(self, x):
        if (x > self.kF): 
            return (self.kF/x)**self.dexp/self.integral0
        else:
            return 1./self.integral0
    def f1(self, x, ik):
        # we will use discrete approximation for gx(ik, x)
        ip = int(x/self.dh)
        if (ip >= self.Nbin): # outside the mesh for the function
            return 0
        res = self.gx[ik,ip]
        return max(res,1e-16)
    def f1_off(self, x, ik):
        # off diagonal h-function
        ip = int(x/self.dh)
        if (ip >= self.Nbin): 
            return 0
        res = self.gx_off[ik,ip]
        return max(res, 1e-16)
    def __call__(self, momentum):
        #PQ_new = 1.0
        #if not self.self_consistent:
        #    for ik in range(1,len(momentum)):
        #        PQ_new *= self.f0( linalg.norm(momentum[ik]) )
        #else:
        #    for ik in range(1,len(momentum)):
        #        k = linalg.norm(momentum[ik])
        #        PQ_new *= self.f1( k, ik )
        #
        #    for ik in range(1,len(momentum)-1):
        #        dkq = linalg.norm(momentum[ik]-momentum[-1])
        #        PQ_new *= self.f1_off( dkq, ik )
        #return PQ_new
        return self_fm(momentum, self.self_consistent, self.kF, self.dexp, self.integral0, self.dh, self.Nbin, self.gx, self.gx_off)
        
    def Recompute(self, SaveData=True):
        self.gx     = zeros( (self.Nloops,self.Nbin) )
        self.gx_off = zeros( (self.Nloops-1, self.Nbin) )
      
        self.self_consistent = True
        # First smoothen the histogram, by averaging over three points.
        for ik in range(1,self.Nloops):
            self.gx[ik,  :] =  self.K_hist[ik,:]
            self.gx[ik, 1:] += self.K_hist[ik,:-1]
            self.gx[ik,:-1] += self.K_hist[ik,1:]
            self.gx[ik,1:-1] *= 1./3.
            self.gx[ik, 0]   *= 1/2.
            self.gx[ik,-1]   *= 1/2.
            
        # Next normalize by 3D integral as if we had a product function
        for ik in range(1,self.Nloops):
            self.gx[ik,:] *= 1./intg3D( self.gx[ik,:], self.dh)
            
        # Next smoothen the off-diagonal histogram.
        for ik in range(self.Nloops-1):
            iik = ik + self.Nloops
            self.gx_off[ik,  :] =  self.K_hist[iik,:]
            self.gx_off[ik, 1:] += self.K_hist[iik,:-1]
            self.gx_off[ik,:-1] += self.K_hist[iik,1:]
            self.gx_off[ik,1:-1] *= 1./3.
            self.gx_off[ik, 0]   *= 1/2.
            self.gx_off[ik,-1]   *= 1/2.
            
        # Off-Diagonal first round of normalization
        for ik in range(self.Nloops-1):
            # We make this off_diagonal function normalized so that each value is on average 1.0
            self.gx_off[ik,:] *= self.Nbin/sum(self.gx_off[ik,:])

        # We need to calculate the following integral
        #
        #  Norm = \int d^3r1...d^3r5  g5(r5) * g1(r1)*g15(|\vr1-\vr5|) * g2(r2)*g25(|\vr2-\vr5|) * ...* g4(r4)*g45(|\vr4-\vr5|)
        # 
        #  This can be turned into radial and angle integral. The important property is that angle between \vr_i-\vr_j| appears in a single term
        #  hence each term can be independetly integrated over phi, and over cos(theta_{ij}) = x_i
        #  We get the following result
        #
        #  Norm = 2*(2*pi)^5 Integrate[ r5^{2-4}*g5(r5) *
        #                               * Integrate[ r1*g1(r1)*u*g15(u) , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
        #                               * Integrate[ r2*g2(r2)*u*g25(u) , {r2,0,cutoff}, {u, |r2-r5|, r2+r5} ]
        #                               * Integrate[ r3*g3(r3)*u*g35(u) , {r3,0,cutoff}, {u, |r3-r5|, r3+r5} ]
        #                               * Integrate[ r4*g4(r4)*u*g45(u) , {r4,0,cutoff}, {u, |r4-r5|, r4+r5} ]
        #                               , {r5, 0, cutoff}]
        #
        #     In the above function "Normalization_X1_X2" we compute polynomial for
        #
        #     F( r5=(i5,t) ) = Integrate[ r1 g_diag(r1) g_off(u) u , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
        #
        #     where g_diag(r1) and g_off(u) are picewise constant functions on the equidistant mesh i*Dlt (i=0,1,2,,,N-1)
        #     Here
        #            r1 = i1 + x, with i1 integer and x=[0,1]
        #            r5 = i5 + t, with i5 integer and t=[0,1]
        #     and therefore
        #
        #     F(i5,t) = \sum_{i1} Integrate[ (i1+x) g_diag[i1]  Integrate[ g_off(u) u , {u, |i5-i1+t-x|, i5+i1+t+x}], {x,0,1}]
        #  The normalization is than
        #
        #  Norm = (4*pi)^5/2^4 Integrate[ r5^{2-4}*g5(r5) * F_{g1}(i5,t) * F_{g2}(i5,t) * F_{g3}(i5,t) * F_{g4}(i5,t), {r5, 0, cutoff}]
        #
        #  Norm = (4*pi*Dlt)^5/2^4 \sum_{i5} Integrate[ (i5+t)^{2-4} * g5[i5] * F_{g1}(i5,t) * F_{g2}(i5,t) * F_{g3}(i5,t) * F_{g4}(i5,t), {t,0,1}]

        if self.Nloops>2:
            dsum = 0.
            for i5 in range(self.Nbin):
                Pt = poly1d([1.0]) # Starting with identity polynomial
                for ik in range(self.Nloops-2,0,-1):
                    Pt *= Normalization_X1_X2( i5, self.Nbin, self.gx_off[ik,:], self.gx[ik,:] )
                # Computes the integral Integrate[ (i5+t)^{-(Nloops-4)} * Pt(t), {t,0,1}]
                dsum += CmpRationalIntegral(Pt, i5, 4-self.Nloops) * self.gx[-1,i5]
            # Noff = Nloops-2
            dsum *= ( 4*pi*self.dh**3)**(self.Nloops-1)/2**(self.Nloops-2)
            # print " dsum, which should be close to unity=", dsum, " and 1-dsum=", 1-dsum
            nrm = 1./dsum**(1./(self.Nloops-2.))
            self.gx_off[:,:] *= nrm;
        
        if (SaveData):
            for ik in range(1,self.Nloops):
                fo = open("pmeasure_weight."+str(ik), 'w')
                for i in range(self.Nbin):
                    print((self.dh*(i+0.5)), self.f1(self.dh*(i+0.5),ik), self.f0(self.dh*(i+0.5)), file=fo)
                fo.close()
            for ik in range(self.Nloops-1):
                fo = open("pmeasure_weight."+str(ik+self.Nloops), 'w')
                for i in range(self.Nbin):
                    print((self.dh*(i+0.5)), self.f1_off(self.dh*(i+0.5),ik), file=fo)
                fo.close()


    def Add_to_K_histogram(self, dk_hist, momentum, cutoffq, cutoffk):
        self_Add_to_K_histogram(dk_hist, momentum, cutoffq, cutoffk, self.K_hist, self.Nbin, self.Nloops)
        #Q = linalg.norm(momentum[0])
        ## external variable histogram
        #if (Q < cutoffq):
        #    iik = int( Q/cutoffq * self.Nbin )
        #    self.K_hist[0,iik] += dk_hist
        ## histogram of other momenta, which we integrate over
        #for ik in range(1,self.Nloops):
        #    k = linalg.norm(momentum[ik]);
        #    if k < cutoffk:
        #        iik = int( k/cutoffk * self.Nbin )
        #        self.K_hist[ik,iik] += dk_hist/k**2
        ## histogram for variable differences. We choose to use
        ## the following combination of momenta
        ##  |k_0-k_{N-1}|,  |k_1-k_{N-1}|,  |k_2-k_{N-1}|, ...., |k_{N-2}-k_{N-1}|
        #for ik in range(self.Nloops-1):
        #    k = linalg.norm(momentum[ik]-momentum[-1])
        #    if k < cutoffk :
        #        iik = int( k/cutoffk * self.Nbin )
        #        self.K_hist[ik+self.Nloops,iik] += dk_hist/k**2
        
    def Normalize_K_histogram(self):
        # We can get overflow during MPI for a long run. We should use a constant for normalization.
        # We could here normalize K_hist (with knorm), and than when we measure, we would
        # add instead of adding unity, we would add 1./knorm
        dnrm = self.Nloops/sum(self.K_hist)
        self.K_hist *= dnrm;
        return dnrm
    
def Normalization_X1_X2(i5, Nbin, g_off, g_diag):
    """ Here we calculate the following integral analytically
    
         F(r5) = Integrate[ r1 g_diag(r1) g_off(u) u , {r1,0,cutoff}, {u, |r1-r5|, r1+r5} ]
    
         where g_diag(r1) and g_off(u) are picewise constant functions on the equidistant mesh i*Dlt (i=0,1,2,,,N-1)
         Introducing continuous and descreate variables:
                r1 = i1 + x, with i1 integer and x=[0,1]
                r5 = i5 + t, with i5 integer and t=[0,1]
         we can write
    
         F(i5,t) = \sum_{i1} Integrate[ (i1+x) g_diag[i1]  Integrate[ g_off(u) u , {u, |i5-i1+t-x|, i5+i1+t+x}], {x,0,1}]
    
         which is a polynomial in t, returned by this function, and depends on input variable i5
    
      To calculate this function, we need to expand it, since g_off(u) is also discrete.
      We have to consider several special cases:
      case : i1 < i5:
             F(i5,t) =   \sum_{i1} g_diag[i1] * Integrate[(i1+x) u g_off[i5-i1-1], {x, t, 1}, {u, i5-i1 + t-x, i5-i1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5-i1]  , {x, t, 1}, {u, i5-i1,  i5-i1+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5-i1]  , {x, 0, t}, {u, i5-i1 + t-x, i5-i1+1}]
     +\sum_{j=|i1-i5|+1}^{i1+i5-1} g_diag[i1] * Integrate[(i1+x) u g_off[j]      , {x, 0, 1}, {u, j, j+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 0,1-t},{u, i1+i5, i1+i5 + t+x}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 1-t,1},{u, i1+i5, i1+i5+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1+1], {x, 1-t,1},{u, i1+i5+1,i1+i5 + t+x}]
      case : i1 > i5
             F(i5,t) =   \sum_{i1} g_diag[i1] * Integrate[(i1+x) u g_off[i1-i5-1], {x, 0, t}, {u, i1-i5 + x-t, i1-i5}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i1-i5]  , {x, 0, t}, {u, i1-i5, i1-i5+1}] + 
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i1-i5]  , {x, t, 1}, {u, i1-i5 + x-t, i1-i5+1}]
     +\sum_{j=|i1-i5|+1}^{i1+i5-1} g_diag[i1] * Integrate[(i1+x) u g_off[j]      , {x, 0, 1}, {u, j, j+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 0,1-t},{u, i1+i5, i1+i5 + t+x}] + 
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1]  , {x, 1-t,1},{u, i1+i5, i1+i5+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[i5+i1+1], {x, 1-t,1},{u, i1+i5+1,i1+i5 + t+x}]
      case : i1 == i5 and i1 != 0
           F(i5,t) =   \sum_{i1}   g_diag[i1] * Integrate[(i1+x) u g_off[0]       , {x, 0, 1}, {u, |t-x|, 1}]
              +\sum_{j=1}^{2*i5-1} g_diag[i1] * Integrate[(i1+x) u g_off[j]       , {x, 0, 1}, {u, j, j+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[2*i5]    , {x, 0,1-t},{u, i1+i5, i1+i5 + t+x}] + 
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[2*i5]    , {x, 1-t,1},{u, i1+i5, i1+i5+1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[2*i5+1]  , {x, 1-t,1},{u, i1+i5+1,i1+i5 + t+x}]
      case : i1 == i5 == 0
             F(i5,t) =   \sum_{i1} g_diag[i1] * Integrate[(i1+x) u g_off[0]       , {x, 0, 1-t},{u, |t-x|, t+x}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[0]       , {x, 1-t, 1},{u, |t-x|, 1}]
                                  +g_diag[i1] * Integrate[(i1+x) u g_off[1]       , {x, 1-t,1},{u, 1, t+x}]
    """
    Ps = zeros(5) # Coefficients of the final polynomial in t for F(i5,t). It can be at most forth order polynomial.
    for i1 in range(Nbin):
        # Stands for F0_45 = \sum_{j=|i1-i5|}^{i1+i5} Integrate[(i1+x) u g_off[j], {x, 0, 1}, {u, j, j+1}]
        #   which is F0_45 = \sum_{j=|i1-i5|}^{i1+i5} (i1+1/2) * (j+1/2) * g_off[j]
        F0_45 = 0.0
        for j in range(abs(i5-i1),i5+i1+1):
            if (j<Nbin):
                F0_45 += g_off[j]*(j+0.5)
            
        F0_45 *= (i1+0.5)
        # Now we start buliding the corrections at the beginning and the end of the [|i5-i1+t-x| , i5+i1+t+x] interval
        P1 = zeros(5)# These are polynomial coefficients
        if (i1==i5 and i5==0): # the above integrals can be computed analytically, and are
            P1[0] =  -1./4. * g_off[0] + F0_45;
            P1[1] =   2./3. * g_off[0]
            P1[2] = -1./2.  * ( g_off[0] - g_off[1] )
            P1[4] =  1./24. * ( g_off[0] - g_off[1] )
        else:
            # Correcting the beginning of the integral, where the exact integral starts at |i5-i1+t-x|
            # and the approximate F0_45 starts at |i5-i1|
            if (i5==i1):
                P1[0] = -(1./8.  + 1./6. * i1) * g_off[0] + F0_45
                P1[1] =  (1./3.  + 1./2. * i1) * g_off[0]
                P1[2] = -(1./4.  + 1./2. * i1) * g_off[0]
            elif (i1<i5):
                P1[0] = (-1./8. + 0.5*i1 * (i5-i1-1) + i5/3. ) * g_off[i5-i1-1] + F0_45
                P1[1] = ( 1./3. - i1 * (i5-i1-1) - i5/2. ) * g_off[i5-i1-1]
                P1[2] = 0.5 * i1 * (i5-i1) * ( g_off[i5-i1-1] - g_off[i5-i1] ) - 0.5*(i1+0.5) * g_off[i5-i1-1]
                P1[3] = ( g_off[i5-i1-1] - g_off[i5-i1] ) * i5/6.
                P1[4] = ( g_off[i5-i1-1] - g_off[i5-i1] ) / 24.
            else:  # i1>i5
                P1[0] =-( i1*(i1-i5+1)/2. - i5/3. + 1./8.) * g_off[i1-i5] + F0_45
                P1[1] = ( i1*(i1-i5+1)    - i5/2. + 1./3.) * g_off[i1-i5]
                P1[2] =  ( g_off[i1-i5-1] - g_off[i1-i5] ) * i1*(i1-i5)/2. - 0.5 * (i1+0.5) * g_off[i1-i5]
                P1[3] = -( g_off[i1-i5-1] - g_off[i1-i5] ) * i5/6.
                P1[4] = -( g_off[i1-i5-1] - g_off[i1-i5] ) / 24.
            # now correcting the end of the integral, where the exact integral stops at i5+i1+t+x
            # and the approximate integrals stops at i5+i1+1
            if (i1+i5 < Nbin):
                P1[0] += -( i1*(i1+i5+1) + i5/3. + 1./4.) * 0.5 * g_off[i1+i5]
                P1[1] +=  ( i1*(i1+i5+1) + i5/2. + 1./3.) *       g_off[i1+i5]
                P1[2] += -( (i1+1)*(i1+i5+1) - (i1+0.5) ) * 0.5 * g_off[i1+i5]
                P1[3] +=  g_off[i1+i5] * i5/6.;
                P1[4] +=  g_off[i1+i5] / 24.;
                if (i1+i5+1 < Nbin):
                    P1[2] +=  (i1+1)*(i1+i5+1) * 0.5 * g_off[i1+i5+1] 
                    P1[3] += -g_off[i5+i1+1] * i5/6.;
                    P1[4] += -g_off[i5+i1+1] / 24.;
                
        P1 *= g_diag[i1]  # Need to multiply with the diagonal function g_diag[i1]
        Ps   += P1        # Finally, sums polynomials over i1
    return poly1d(Ps[::-1])
    #return Ps







def intg3D(f, dh):
    # integrates function which is assumed to be constant in each interval,
    # but with weight 4*pi*k^2 f(k) dk
    i3 = arange(len(f)+1)**3  # i^3
    ii3 = i3[1:] - i3[:-1]    # (i+1)^3 - i^3
    return sum( ii3 * f ) * dh**3 * 4*pi/3.



def CmpRationalIntegral(Pt, i_n, power): 
    """ Computes the integral 
          Integrate[ (i_n+t)^power * Pt(t), {t,0,1}]
        which can be expanded to
          \sum_i a_i Integrate[ t^i (t+i_n)^power , {t,0,1}]
    For positive powers it is simple:
          \sum_{k=0,power} Cobinatorial(power,k) i_n^k \sum_i a_i/(power-k+i+1)
    For power == -1, the integral is
          I[i] = Integrate[ t^i /(t+i_n) , {t,0,1}] 
    and satisfies recursion relation:
          In[i+1] = 1/(i+1) - i_n * In[i]
    """
    def PolyInt_m1(i_n, N):
        """
           Computes :
                Integrate[ t^i /(i_n + t ) , {t,0,1}]
            can be shown to satisfy the following recursion relation:
                In[n+1] = 1/(n+1) - i_n * In[n]
        """
        In = zeros(N)  # need In for all nonzero orders
        if N>0:
            #  Upward recursion is very straighforard:
            #  In[n+1] = 1/(n+1) - i_n * In[n]   and  I[0] = log((i_n+1)/i_n)
            if i_n==0:
              In[1:] = 1./arange(1,N)
            else:
              In[0] = log((i_n+1.0)/i_n)
              if (N>1): 
                  In[1] = 1 - i_n * In[0]
                  for n in range(1,N-1):
                    In[n+1] = 1/(n+1.) - i_n * In[n]
        return In
    
    n = Pt.o
    if power>=0:
        Cn = special.binom(power, range(power+1))
        i_n_2_k = 1.
        dsum = 0.0
        for k in range(power+1):
            n = Pt.o
            ds = sum([Pt.c[n-i]/(power-k+i+1.) for i in range(n+1)])
            dsum += Cn[k]*i_n_2_k*ds;
            i_n_2_k *= i_n;
        return dsum
    else:
        In = zeros(n+1)
        if power==-1:
            In = PolyInt_m1(i_n, n+1)
        elif power==-2:
            if (i_n==0):
                In[2:] = 1./(arange(2,n+1) - 1.)
            else:
                Kn = PolyInt_m1(i_n, n)
                In[0] = 1.0/i_n - 1.0/(i_n+1.0)
                for n in range(1,n+1):
                    In[n] = -1./(i_n+1.) + n * Kn[n-1]
        elif power==-3:
            if (i_n==0):
                In[3:] = 1./(arange(3,n+1) - 2.)
            else:
                Kn = PolyInt_m1(i_n, n-1)
                In[0] = 0.5*( 1./i_n**2 - 1./(i_n+1.)**2 )
                In[1] = -0.5/(i_n+1.)*( 1.0/(i_n+1.) + 1 ) + 0.5/i_n
                for n in range(2,n+1):
                    In[n] = -0.5/(i_n+1.)*( 1.0/(i_n+1.) + n ) + 0.5*(n-1)*n*Kn[n-2]
        else:
            k = -power-1
            if (i_n==0):
                In[k+1:] = 1./(arange(k+1,n+1)-k)
            else:
                Kn = PolyInt_m1(i_n, n+1-k)
                c0 = 0
                cp, cq = zeros(k+1), zeros(k+1)
                cp[0] = ( (1./(i_n+1.))**k )/k
                cq[0] = ( (1./(i_n+0.))**k )/k
                # cp(0) = 1/(i+1)^k/k                   cq(0) = 1/i^k/k
                # cp(1) = 1/(i+1)^{k-1}/(k*(k-1))       cq(1) = 1/i^{k-1}/(k*(k-1))
                # cp(2) = 1/(i+1)^{k-2}/(k*(k-1)*(k-2)) cq(2) = 1/i^{k-2}/(k*(k-1)*(k-2))
                # cp(k) = 1/(k!)                        cq(k) = 1/(k!)
                for i in range(1,k):
                    cp[i] = cp[i-1] * (i_n+1.)/(k-i)
                    cq[i] = cq[i-1] * (i_n+0.)/(k-i)
                cp[k] = cp[k-1] * (i_n+1.)
                cq[k] = cq[k-1] * (i_n+0.)
                # In[0] = -[ cp(0) - cq(0)];
                # In[1] = -[ cp(0) + ( cp(1) - cq(1) ) ];
                # In[2] = -[ cp(0) + 2*cp(1) + 2*1*( cp(2) - cq(2) ) ];
                # In[3] = -[ cp(0) + 3*cp(1) + 3*2* cp(2) + 3*2*1*(cp(3)-cq(3)) ];
                # In[i] = -[ cp(0) + i*cp(1) + i*(i-1)*cp(i-1) + i*(i-1)*(i-2)*cp(i-2) ... ]
                for i in range(k):
                    dsm = 0.0
                    dd  = 1.0
                    for j in range(i):
                        dsm += dd * cp[j]
                        dd *= (i-j)
                    dsm += dd*(cp[i]-cq[i])
                    In[i] = -dsm
                for i in range(k,n+1):
                    dsm = 0.0
                    dd  = 1.0
                    for j in range(k):
                        dsm += -dd * cp[j]
                        dd *= (i-j)
                    In[i] = dsm + dd*cp[k]*Kn[i-k]
    return dot(In[::-1], Pt.c)

