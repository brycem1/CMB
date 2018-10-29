import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
#from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
#import matplotlib.colors as mcolors
from scipy.integrate import quad,simps,romberg
from scipy.special import spherical_jn, gamma

from scipy.interpolate import UnivariateSpline


#import rotation as ro
#import lensing as le
import numba
import camb

from IPython import embed

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad
#Camb initialisation
pars= camb.CAMBparams()
data= camb.get_background(pars)

#SI constants
Omega_b = 0.0486
p_cr0 = 8.62E-27
chi = (1-0.24)/(1-0.12)
mu_e = 1.14
m_p = 1.6726E-27
sig_t = 6.65246E-29
omega_lamb = 0.6911
omega_k = 0.
omega_r = 9.2364E-5
omega_m = 0.3089
#H_0 = 2.195E-18
#H_0 is in km/s/Mpc
H_0 = 67.74
h = 0.6774
#c is in km/s
c = 2.9979E5


#Gaussian-cgs constants
c_cgs = 2.9979E10
e_cgs = 4.8E-10
f_cgs = 150E9
n_B = -2.9
B_l = 4.5E-9

@numba.jit
def S_0(nb,B_lamba):
    lamba = 1.
    s0 = (2.*np.pi*B_lamba)**2.*(lamba)**(nb+3.)/(2.*gamma((nb+3.)/2.))
    return s0

@numba.jit
def k_D(nb,B_lamba):
    lamba = 1.
    kd = (5.5E4*h*(B_lamba/(1.E-9))**(-2)*(2.*np.pi/lamba)**(nb+3.)*(0.0223/0.022))**(1./(nb+5.))
    return kd


@numba.jit
#integrand of eta 
def integrand_eta(a):
    return 1./np.sqrt((omega_lamb*a**4+omega_m*a+omega_r+omega_k*a**2))
  
#initialising redshift and scale factor array for 0<=z<=1100, 1/1101 <=a <= 1 for n = 3000 entries
zstar = data.get_derived_params()
zstar = zstar['zstar']

z_array = np.linspace(0.,zstar,num=10000)
eta_0 = data.conformal_time(0)
eta_star = data.conformal_time(zstar)
eta_array = np.linspace(eta_star,eta_0,num = 10000)
opacity = data.get_background_time_evolution(eta_array,vars='opacity',format='array')
opacity = opacity[:,0]

tck = interpolate.splrep(eta_array, opacity, s=0)


numpoints = 100

eta_test_array = np.linspace(eta_star,eta_0,numpoints)


@numba.jit
def opacity_(eta):
    return interpolate.splev(eta,tck)

@numba.jit
#Transfer function using simpsons rule
def T_lk(k,L):
    integrand = []
    for jj in range(len(eta_array)):
        integrand.append(opacity[jj]*spherical_jn(L,k*(eta_0-eta_array[jj])))
    integrand = np.array(integrand)
    return simps(integrand)

@numba.jit
#Transfer function integrand
def T_lk_integrand(eta,k,L):
    return opacity_(eta)*spherical_jn(L,k*(eta_0-eta))

@numba.jit
#Transfer function quad
def T_lk_quad(k,L):
    #return quad(T_lk_integrand,eta_star,eta_0,args=(k,L),epsabs=1.49e-015, epsrel=1.49e-015)[0]
    #return quad(T_lk_integrand,eta_star,eta_0,args=(k,L),)
    Ty = np.zeros(numpoints)
    for i in range(1,numpoints):
        Ty[i] = Ty[i-1] + quad(T_lk_integrand,eta_test_array[i-1],eta_test_array[i],args=(k,L))[0]
    return Ty[-1]

@numba.jit
#Transfer function using simpsons rule with Bessel derivative
def T_1lk(k,L):
    integrand = []
    for jj in range(len(eta_array)):
        integrand.append(opacity[jj]*spherical_jn(L,k*(eta_0-eta_array[jj]),derivative=True))
    integrand = np.array(integrand)
    return simps(integrand)


#Transfer function integrand with bessel derivative
@numba.jit
def T_1lk_integrand(eta,k,L):
    return opacity_(eta)*spherical_jn(L,k*(eta_0-eta),derivative=True)

#Transfer function quad with bessel derivative
@numba.jit
def T_1lk_quad(k,L):
    #return quad(T_1lk_integrand,eta_star,eta_0,args=(k,L),epsabs=1.49e-015, epsrel=1.49e-015)[0]
    #return quad(T_1lk_integrand,eta_star,eta_0,args=(k,L))
    T1y = np.zeros(numpoints)
    for i in range(1,numpoints):
        T1y[i] = T1y[i-1] + quad(T_1lk_integrand,eta_test_array[i-1],eta_test_array[i],args=(k,L))[0]
    return T1y[-1]

ell_test = np.arange(1000,10000,100)
k_test = np.logspace(-3.,1.,num=1000)
Tl_m1k = np.zeros((len(ell_test),len(k_test)))
Tl_p1k = np.zeros((len(ell_test),len(k_test)))
T1lk = np.zeros((len(ell_test),len(k_test)))
for ll in range(len(ell_test)):
   for kk in range(len(k_test)):
       Tl_m1k[ll,kk] = T_lk_quad(k_test[kk],ell_test[ll]-1)
       Tl_p1k[ll,kk] = T_lk_quad(k_test[kk],ell_test[ll]+1)
       T1lk[ll,kk] = T_1lk_quad(k_test[kk],ell_test[ll])
       print(ll,kk)

embed()

data = {'ell' : ell_test, 'k' : k_test, 'Tl_m1k' : Tl_m1k, 'Tl_p1k' : Tl_p1k, 'T1lk' : T1lk}
with open("transfer101.pkl", "wb") as infile:
    pickle.dump(data, infile)

