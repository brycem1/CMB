import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
#from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
#import matplotlib.colors as mcolors
from scipy.integrate import quad,simps,romberg
from scipy.special import spherical_jn, gamma

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


'''
#LEGACY: density of gas(z) 
def p_g(z):
    p_g0 = Omega_b * p_cr0
    return p_g0*(1.+z)**3.
#LEGACY: electron number density
def n_e(z):
    return (chi*p_g(z))/(mu_e*m_p)   
#LEGACY: dtaudeta
def tau_dot(z):
    return n_e(z)*sig_t/(1.+z)
'''
@numba.jit
#integrand of eta 
def integrand_eta(a):
    return 1./np.sqrt((omega_lamb*a**4+omega_m*a+omega_r+omega_k*a**2))
  
#initialising redshift and scale factor array for 0<=z<=1100, 1/1101 <=a <= 1 for n = 3000 entries
z_array = np.linspace(0.,1100.,num=3000)
a_array = 1./(1.+z_array)
'''
#LEGACY: my gut says this legacy code
n_e_array = []
tau_dot_array = []
for i in range(len(z_array)):
    n_e_array.append(n_e(z_array[i]))
    tau_dot_array.append(tau_dot(z_array[i]))
n_e_array = np.array(n_e_array)
tau_dot_array = np.array(tau_dot_array)
'''
#generating a list of eta values from scale factor
eta_array = []
for j in range(len(a_array)):
    eta_array.append(quad(integrand_eta,0.,a_array[j]))
eta_array = np.array(eta_array)[:,0]
eta_array = c*eta_array/H_0

#flips the eta, z, and a arrays for reference
z_array_rev = np.flip(z_array)
a_array_rev = np.flip(a_array)
eta_array_rev = np.flip(eta_array)
cs = interpolate.CubicSpline(eta_array_rev,a_array_rev)

#cubic spline with new eta atm
tck = interpolate.splrep(eta_array_rev, a_array_rev, s=0) 

#eta_0
eta_0 = c*quad(integrand_eta,0.,1.)[0]/H_0
#eta_star	
visibility = data.get_background_redshift_evolution(z_array,vars='visibility',format='array')[:,0]
visibility_2 = data.get_background_time_evolution(eta_array,vars='visibility',format='array')[:,0]
#plt.plot(z_array,visibility)
#plt.show()
z_star = z_array[np.argmax(visibility)]
eta_star = c*quad(integrand_eta,0.,1./(1+z_star))[0]/H_0
#Eta array for integrating over
eta4transferint = np.linspace(eta_star,eta_0,num=3000) 

@numba.jit
#function of a(eta)
def a_(eta):
    return interpolate.splev(eta,tck)

@numba.jit
#kept units in SI then converted m to Mpc
def tau_dot_eta(eta):
    p_g0 = Omega_b * p_cr0
    constants = (chi*p_g0*sig_t)/(mu_e*m_p) 
    return constants/(3.24077928966636E-23*a_(eta)**(2.))
@numba.jit
#Transfer function using simpsons rule
def T_lk(k,L):
    integrand = []
    for jj in range(len(eta4transferint)):
        integrand.append(tau_dot_eta(eta4transferint[jj])*spherical_jn(L,k*(eta_0-eta4transferint[jj])))
    integrand = np.array(integrand)
    return simps(integrand)

@numba.jit
#Transfer function integrand
def T_lk_integrand(eta,k,L):
    return tau_dot_eta(eta)*spherical_jn(L,k*(eta_0-eta))

@numba.jit
#Transfer function quad
def T_lk_quad(k,L):
    return quad(T_lk_integrand,eta_star,eta_0,args=(k,L),epsabs=1.49e-015, epsrel=1.49e-015)[0]

@numba.jit
#Transfer function using simpsons rule with Bessel derivative
def T_1lk(k,L):
    integrand = []
    for jj in range(len(eta4transferint)):
        integrand.append(tau_dot_eta(eta4transferint[jj])*spherical_jn(L,k*(eta_0-eta4transferint[jj]),derivative=True))
    integrand = np.array(integrand)
    return simps(integrand)


#Transfer function integrand with bessel derivative
@numba.jit
def T_1lk_integrand(eta,k,L):
    return tau_dot_eta(eta)*spherical_jn(L,k*(eta_0-eta),derivative=True)

#Transfer function quad with bessel derivative
@numba.jit
def T_1lk_quad(k,L):
    return quad(T_1lk_integrand,eta_star,eta_0,args=(k,L),epsabs=1.49e-015, epsrel=1.49e-015)[0]


#def T_lk_integrand(L,k,eta):
#    return tau_dot_eta(eta)*spherical_jn(L,k*(eta_0-eta))
@numba.jit
def del_m2(k):
    return k**3.*S_0(n_B,B_l)*k**n_B*((3*c_cgs**2.)/(16*np.pi**2.*e_cgs*f_cgs**2.))**2.

#quad?
@numba.jit
def Cl_aa_integrand(k,l):
    return (1./k)*del_m2(k)*((l/(2.*l+1))*T_lk_quad(k,l-1)**2.+((l+1)/(2.*l+1))*T_lk_quad(k,l+1)**2.-T_1lk_quad(k,l)**2.)


#k terminals
k_min = 1.0E-4
k_max = k_D(n_B,B_l)
@numba.jit
#Claa
def Cl_aa(l):
    #embed()
    return quad(Cl_aa_integrand,k_min,k_max,args=(l,))
'''
T = []
T1 = []
ks = np.linspace(k_min,k_max,num=3000)
for kk in range(500):
    T.append(T_lk_quad(ks[kk],100))
    T1.append(T_1lk_quad(ks[kk],100))
    print(kk)
'''
ell = np.arange(20,1000,10)
Claa = []
tol = []
#embed()
for jj in range(len(ell)):
    Claa.append((2./np.pi)*Cl_aa(ell[jj])[0])
    tol.append(Cl_aa(ell[jj])[1])
    print(jj)


#test eta values n=2500
#eta_new = np.linspace(2.86141419e+16,1.45993085e+18,num=2500)
#a_new = interpolate.splev(eta_new,tck)  


#Spherical test rubbish
#spherical_jn(n,z,derivative=False) as j_n(z); n>=0; 
#T= np.array(T)
#T1 = np.array(T1)

Claa = np.array(Claa)
#claa
embed()


T = []
T1 = []
ks = np.linspace(k_min,k_max,num=3000)


for jj in range(len(ell)): 
    Claa.append(Cl_aa(ell[jj]))
    print(jj)
#testing transfer function tolerance
for kk in range(len(ks)):
    T.append(T_lk_quad(ks[kk],200))
    T1.append(T_1lk_quad(ks[kk],200))
    print(kk)
     
T= np.array(T)
T1 = np.array(T1)

#testing integrand of transfer function
sjn_eta_ks = np.zeros((len(ks),len(eta4transferint)))   

for kk in range(len(ks)):                                            
    for ee in range(len(eta4transferint)):
        sjn_eta_ks[kk,ee] = spherical_jn(200,ks[kk]*(eta_0-eta4transferint[ee]))

plt.plot(eta4transferint,sjn_eta_ks[50]) 
plt.plot(eta4transferint,tau_dot_eta(eta4transferint)*sjn_eta_ks[50])
plt.plot(eta4transferint,tau_dot_eta(eta4transferint))

#seconds to Gyr is 3.16E16	
'''
In [9]: plt.figure()
Out[9]: <matplotlib.figure.Figure at 0x7f95bf796a90>

In [10]: plt.plot(eta_array_rev, a_array_rev, 'x', xnew,ynew)
Out[10]: 
[<matplotlib.lines.Line2D at 0x7f95bf7cd810>,
 <matplotlib.lines.Line2D at 0x7f95a661a290>]

In [11]: plt.legend(['Data', 'Cubic Spline'])
Out[11]: <matplotlib.legend.Legend at 0x7f95a6642090>

In [12]: plt.show()
'''



