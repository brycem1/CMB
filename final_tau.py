import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
#import matplotlib.colors as mcolors
from scipy.integrate import quad,simps,romberg
from scipy.special import spherical_jn, gamma

from scipy.interpolate import UnivariateSpline

import pickle
import rotation as ro
import lensing as le
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
f_cgs = 70E9
n_B = 2.0
B_l = 10.E-9
Bl = B_l*(1.E9)

data = pickle.load(open("transfer.pkl","rb"))
ell_array = data['ell']
k_array = data['k']
Tl_m1k = data['Tl_m1k']
Tl_p1k = data['Tl_p1k']
T1lk = data['T1lk'] 

cosmo = Cosmo()
cmbspec = cosmo.cmb_spectra(1200)
cmbspec_r = Cosmo({'r':1.}).cmb_spectra(1200,spec='tensor')[:,2]

def GimmeClBBRot(cmbspec, claa, dl=10, n=512, nwanted=100):
	lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(2,nwanted)*dl    
	
	clee = cmbspec[:,1].copy()
	
	eepowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
	
	ell = np.arange(cmbspec.shape[0])
	
					
        #embed()
	aapowerspec2d = np.interp(lgrid, L, claa)
	
	clrot = ro.ClBB_rot(eepowerspec2d, aapowerspec2d, dl, n, nwanted)
	clrot=clrot[2:nwanted]
	return L, clrot

#@numba.jit
def S_0(nb,B_lamba):
    lamba = 1.
    s0 = (2.*np.pi*B_lamba)**2.*(lamba)**(nb+3.)/(2.*gamma((nb+3.)/2.))
    return s0

#@numba.jit
def k_D(nb,B_lamba):
    lamba = 1.
    kd = (5.5E4*h*(B_lamba/(1.E-9))**(-2)*(2.*np.pi/lamba)**(nb+3.)*(0.0223/0.022))**(1./(nb+5.))
    return kd

#embed()

#def T_lk_integrand(L,k,eta):
#    return tau_dot_eta(eta)*spherical_jn(L,k*(eta_0-eta))
#@numba.jit
def del_m2(k,nb,Blamba):
    return k**2.*S_0(nb,Blamba)*k**n_B*((3*c_cgs**2.)/(16*np.pi**2.*e_cgs*f_cgs**2.))**2.

#quad?
#@numba.jit
def Cl_aa_integrand(k_index,l_index,nb,B_lamba):
    k = k_array[k_index]
    l = ell_array[l_index]
    return del_m2(k,nb,B_lamba)*((l/(2.*l+1))*Tl_m1k[l_index,k_index]**2.+((l+1)/(2.*l+1))*Tl_p1k[l_index,k_index]**2.-T1lk[l_index,k_index]**2.)

def Cl_aa_spline(l_index,nb,B_lamba):
    integrand = []
    for jj in range(len(k_array)):
       integrand.append(Cl_aa_integrand(jj,l_index,nb,B_lamba))
       #print(jj)
    integrand = np.array(integrand)
    abs_integrand = np.abs(integrand)
    #print(abs_integrand[np.argmax(abs_integrand):-1])
    #embed()
    #sp = UnivariateSpline(k_array[np.argmax(abs_integrand):-1], abs_integrand[np.argmax(abs_integrand):-1], k=1, s=0)
    #return sp.integral(k_array[np.argmax(abs_integrand)],k_array[-1])
    sp = UnivariateSpline(k_array, abs_integrand, k=1, s=0)
    return sp.integral(k_array[np.argmax(abs_integrand)],k_array[-1])

Claa = []
for ii in range(len(ell_array)):
    Claa.append(Cl_aa_spline(ii,n_B,B_l))
    print(ii)
Claa = np.array(Claa)
#with open("Claa_70_20.pkl", "wb") as infile:
#    pickle.dump(data, infile)

embed()
l_aa, clBBrot_i = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)
plt.loglog(l_aa, clBBrot_i*l_aa**2)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell^2C_\ell^{BB}$',size = 14)
'''
B_array = np.arange(0.5E-9,4.6E-9,0.5E-9)
B_nb_neg29 = []
B_nb_neg25 = []
B_nb_neg20 = []
B_nb_neg15 = []
B_nb_neg10 = []
B_nb_0 = []
B_nb_10 = []

for kk in range(len(B_array)):
    n_B = -2.9
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_neg29.append((l_aa,Claa,ClBB))
    print(kk)

for kk in range(len(B_array)):
    n_B = -2.5
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_neg25.append((l_aa,Claa,ClBB))
    print(kk)

for kk in range(len(B_array)):
    n_B = -2.0
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_neg20.append((l_aa,Claa,ClBB))
    print(kk)

for kk in range(len(B_array)):
    n_B = -1.5
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_neg15.append((l_aa,Claa,ClBB))
    print(kk)

for kk in range(len(B_array)):
    n_B = -1.0
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_neg10.append((l_aa,Claa,ClBB))
    print(kk)

for kk in range(len(B_array)):
    n_B = 0.0
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_0.append((l_aa,Claa,ClBB))
    print(kk)

for kk in range(len(B_array)):
    n_B = 1.0
    Claa = []
    for ii in range(len(ell_array)):
        Claa.append(Cl_aa_spline(ii,n_B,B_array[kk]))
    Claa = np.array(Claa)
    l_aa, ClBB = GimmeClBBRot(cmbspec,Claa,dl=10, n=512, nwanted=100)    
    B_nb_10.append((l_aa,Claa,ClBB))
    print(kk)

with open("transfer1.pkl", "wb") as infile:
    pickle.dump(data, infile)

cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(1200,spec='tensor')[:,2]

B_nb_neg29 = np.array(B_nb_neg29)
B_nb_neg25 = np.array(B_nb_neg25)
B_nb_neg20 = np.array(B_nb_neg20)
B_nb_neg15 = np.array(B_nb_neg15)
B_nb_neg10 = np.array(B_nb_neg10)
B_nb_0 = np.array(B_nb_0)
B_nb_10 = np.array(B_nb_10)

ClBB_neg29 = B_nb_neg29[:,2][8]
ClBB_neg25 = B_nb_neg25[:,2][8]
ClBB_neg20 = B_nb_neg20[:,2][8]
ClBB_neg15 = B_nb_neg15[:,2][8]
ClBB_neg10 = B_nb_neg10[:,2][8]
ClBB_0  = B_nb_0[:,2][8]
ClBB_10 = B_nb_10[:,2][8]

DlBB_neg29 = ell_array*(ell_array+1.)*ClBB_neg29/(2.*np.pi)
DlBB_neg25 = ell_array*(ell_array+1.)*ClBB_neg25/(2.*np.pi)
DlBB_neg20 = ell_array*(ell_array+1.)*ClBB_neg20/(2.*np.pi)
DlBB_neg15 = ell_array*(ell_array+1.)*ClBB_neg15/(2.*np.pi)
DlBB_neg10 = ell_array*(ell_array+1.)*ClBB_neg10/(2.*np.pi)
DlBB_0  = ell_array*(ell_array+1.)*ClBB_0/(2.*np.pi)
DlBB_10 = ell_array*(ell_array+1.)*ClBB_10/(2.*np.pi)

plt.loglog(ell_array,DlBB_neg29,label = r'$n_B = -2.9$')
plt.loglog(ell_array,DlBB_neg25,label = r'$n_B = -2.5$')
plt.loglog(ell_array,DlBB_neg20,label = r'$n_B = -2.0$')
plt.loglog(ell_array,DlBB_neg15,label = r'$n_B = -1.5$')
plt.loglog(ell_array,DlBB_neg10,label = r'$n_B = -1.0$')
plt.loglog(ell_array,DlBB_0, label = r'$n_B = 0.0$')
plt.loglog(ell_array,DlBB_10,label = r'$n_B = 1.0$')
plt.loglog(cmbspec[:,2]*np.arange(cmbspec.shape[0])*(np.arange(cmbspec.shape[0])+1.)/(2*np.pi), 'k', label='Lensing')
plt.loglog(cmbspec_r*np.arange(cmbspec_r.shape[0])*(np.arange(cmbspec_r.shape[0])+1.)/(2*np.pi), 'k', label=r'GW $r=0.1$')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell^{BB}/2\pi \quad [\mu {\rm K}^2]$')
plt.title(r'FR-BB Power Spectra for different $n_B$ and $B_{{1Mpc}} = 4.5 {nG}$')
plt.legend(loc = 'best')
plt.show()

ClBB_neg29 = B_nb_neg29[:,1][8]
ClBB_neg25 = B_nb_neg25[:,1][8]
ClBB_neg20 = B_nb_neg20[:,1][8]
ClBB_neg15 = B_nb_neg15[:,1][8]
ClBB_neg10 = B_nb_neg10[:,1][8]
ClBB_0  = B_nb_0[:,1][8]
ClBB_10 = B_nb_10[:,1][8]

DlBB_neg29 = ell_array*(ell_array+1.)*ClBB_neg29/(2.*np.pi)
DlBB_neg25 = ell_array*(ell_array+1.)*ClBB_neg25/(2.*np.pi)
DlBB_neg20 = ell_array*(ell_array+1.)*ClBB_neg20/(2.*np.pi)
DlBB_neg15 = ell_array*(ell_array+1.)*ClBB_neg15/(2.*np.pi)
DlBB_neg10 = ell_array*(ell_array+1.)*ClBB_neg10/(2.*np.pi)
DlBB_0  = ell_array*(ell_array+1.)*ClBB_0/(2.*np.pi)
DlBB_10 = ell_array*(ell_array+1.)*ClBB_10/(2.*np.pi)

plt.loglog(ell_array,ClBB_neg29,label = r'$n_B = -2.9$')
plt.loglog(ell_array,ClBB_neg25,label = r'$n_B = -2.5$')
plt.loglog(ell_array,ClBB_neg20,label = r'$n_B = -2.0$')
plt.loglog(ell_array,ClBB_neg15,label = r'$n_B = -1.5$')
plt.loglog(ell_array,ClBB_neg10,label = r'$n_B = -1.0$')
plt.loglog(ell_array,ClBB_0, label = r'$n_B = 0.0$')
plt.loglog(ell_array,ClBB_10,label = r'$n_B = 1.0$')
#plt.loglog(cmbspec[:,2]*np.arange(cmbspec.shape[0])*(np.arange(cmbspec.shape[0])+1.)/(2*np.pi), 'k', label='Lensing')
#plt.loglog(cmbspec_r*np.arange(cmbspec_r.shape[0])*(np.arange(cmbspec_r.shape[0])+1.)/(2*np.pi), 'k', label=r'GW $r=0.1$')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\alpha\alpha}$')
plt.title(r'FR Rotation Power Spectra for different $n_B$ and $B_{{1Mpc}} = 4.5 {nG}$')
plt.legend(loc = 'best')
plt.show()

embed()

'''
plt.rc('text', usetex=True)
plt.loglog(ell_array,Claa)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\alpha\alpha}$')
plt.title(r'Faraday rotation $C_{\ell}^{\alpha\alpha}$ for $n_B = %.1f,B_{{1Mpc}} = %.1f {nG}$' %(n_B, Bl))
plt.show()


'''
plt.contourf(Bl_array,n_B_array,kd_array);plt.colorbar()
plt.xlabel(r'$B_{{1Mpc}}$ [nG]',size=14)
plt.ylabel(r'$n_B$',size=14)
plt.title(r'$k_D$ contour')
plt.show()

Bl_array = np.linspace(1.0E-9, 10.0E-9)

In [2]: Bl_array.shape
Out[2]: (50,)

In [3]: n_B_array = np.arange(-3.0,3.1)

In [4]: n_B_array.shape
Out[4]: (7,)

In [5]: n_B_array = np.arange(-3.0,3.1,0.1)

In [6]: n_B_array.shape
Out[6]: (61,)

In [7]: kd_array = np.zeros((n_B_array.shape[0],Bl_array.shape[0]))

In [8]: kd_array.shape
Out[8]: (61, 50)

In [9]: for kk in range(len(n_B_array)):
   ...:     for pp in range(len(Bl_array)):
   ...:         kd_array[kk,pp] = k_D(n_B_array[kk],Bl_array[pp])
   ...:         

'''



