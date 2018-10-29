import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
import matplotlib.colors as mcolors

import rotation as ro
import lensing as le
import numba

from IPython import embed

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad

cosmo = Cosmo()

cmbspec = cosmo.cmb_spectra(5000)
cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(3000, spec='tensor')

#embed()
@numba.jit
def snr(cmbspec, beam, noise,fsky=0.1, kind='default', A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=400):
    L     = np.arange(5,nwanted)*dl    
    Nlaa = ro.GimmeNl(cmbspec,beam,noise,dl = dl,n=n,nwanted = nwanted)[1]
    Nlaa = Nlaa[5:len(Nlaa)]
    if kind == 'default':
	claa = np.nan_to_num(A_CB*1e-5*2*np.pi/L/(L+1))
    elif (kind == 'pmf') or (kind == 'PMF'):
	claa = np.nan_to_num(2.3e-5 * (30./nu)**4 * B**2 * 2*np.pi/L/(L+1))
    elif (kind == 'pseudoscalar') or (kind == 'CS'):
	claa = np.nan_to_num((H_I/2/np.pi/f_a)**2 * 2*np.pi/L/(L+1))
  
    del_claa = np.sqrt(2./((2.*L+1)*fsky*dl))*(claa+Nlaa)
    return np.sqrt(np.sum((claa/del_claa)**2))

@numba.jit
def nl_AA(cmbspec, beam, noise, dl=10,n=512,nwanted=400):
    L = np.arange(5,nwanted)*dl  
    Nlaa = ro.GimmeNl(cmbspec,beam,noise,dl = dl,n=512,nwanted = nwanted)[1]
    Nlaa = Nlaa[5:len(Nlaa)]
    return L, Nlaa

@numba.jit
def snr_ACB(L, Nlaa = None, fsky=0.1, kind='default', A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=400):
    if kind == 'default':
	claa = np.nan_to_num(A_CB*1e-5*2*np.pi/L/(L+1))
    elif (kind == 'pmf') or (kind == 'PMF'):
	claa = np.nan_to_num(2.3e-5 * (30./nu)**4 * B**2 * 2*np.pi/L/(L+1))
    elif (kind == 'pseudoscalar') or (kind == 'CS'):
	claa = np.nan_to_num((H_I/2/np.pi/f_a)**2 * 2*np.pi/L/(L+1))
  
    del_claa = np.sqrt(2./((2.*L+1)*fsky*dl))*(claa+Nlaa)
    return np.sqrt(np.sum((claa/del_claa)**2))



#embed()
ACB = np.logspace(-3.0,3.0,num=1000)

snr0 = []
snr1 = []
snr2 = []
snr3 = []
snr_tot = []
#cmbspec, beam, noise
l0, Nlaa0 = nl_AA(cmbspec,1.3,9.8)
l1, Nlaa1 = nl_AA(cmbspec,30.0,3.4)
l2, Nlaa2 = nl_AA(cmbspec,3.5,11.8)
l3, Nlaa3 = nl_AA(cmbspec,1.1,4.5)

#nlaa_tot =np.nan_to_num(((1./(Nlaa0))+(1./(Nlaa2))+(1./(Nlaa3)))**(-1))

for i in range(len(ACB)):
    snr0.append(snr_ACB(l0, Nlaa=Nlaa0,fsky=0.5,A_CB=ACB[i]))
    #embed()
    snr1.append(snr_ACB(l1, Nlaa=Nlaa1,fsky=0.01,A_CB=ACB[i]))
    snr2.append(snr_ACB(l2, Nlaa=Nlaa2,fsky=0.65,A_CB=ACB[i]))
    snr3.append(snr_ACB(l3, Nlaa=Nlaa3,fsky=0.06,A_CB=ACB[i]))
    snr_tot.append(snr_ACB(l0, Nlaa=nlaa_tot,fsky=0.5,A_CB=ACB[i]))
    print(i)

#snr_tot.append(snr_ACB(l0, Nlaa=nlaa_tot,fsky=0.5))
snr0 = np.array(snr0)
snr1 = np.array(snr1)
snr2 = np.array(snr2)
snr3 = np.array(snr3)
snr_tot = np.array(snr_tot)

snr_tot1 = []
for j in range(len(ACB)):
    snr_tot1.append(snr_ACB(l0, Nlaa=nlaa_tot,fsky=1.0,A_CB=ACB[j]))

snr_tot1 = np.array(snr_tot1)

embed()

plt.loglog(ACB,snr0,label='ACTPol')
plt.loglog(ACB,snr1,label='BICEP3')
plt.loglog(ACB,snr2,label='SA')
plt.loglog(ACB,snr3,label='SPT3G')
#plt.loglog(ACB,snr_tot,label=r'ACT+SA+SPT3G,$f_{sky}=0.5$')
#plt.loglog(ACB,snr_tot1,label=r'ACT+SA+SPT3G,$f_{sky}=1.0$')
plt.axhline(y=3,c='k',ls='--')
plt.xlabel(r'$A_{CB}$')
plt.ylabel(r'$S/N$')
plt.legend(loc='best')
plt.show()
