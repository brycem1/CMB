import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
import matplotlib.colors as mcolors
import pylab as py
import camb

import rotation as ro
import lensing as le
import numba

from IPython import embed

ell_tens = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research12/r_test2_tensCls.dat')[:,0]
tens = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research12/r_test2_tensCls.dat')[:,3]

ell_vec = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research12/r_test_vecCls.dat')[:,0]
vec = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research12/r_test_vecCls.dat')[:,3]

tens_vec = tens+vec
cl_tens_vec = np.nan_to_num((tens_vec * 2*np.pi)/(ell_vec*(ell_vec+1.)))

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad
#Camb initialisation
pars= camb.CAMBparams()
data_camb= camb.get_background(pars)

cosmo = Cosmo()
cmbspec = cosmo.cmb_spectra(3000)
#cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(12000,spec='tensor')[:,2]
clbb_r = Cosmo({'r':1.}).cmb_spectra(3000,spec='tensor')[:,2]
clbb_r1 = Cosmo({'r':0.0042}).cmb_spectra(3000,spec='tensor')[:,2]
clbb_r2 = Cosmo({'r':0.01}).cmb_spectra(3000,spec='tensor')[:,2]
clbb_r3 = Cosmo({'r':0.1}).cmb_spectra(3000,spec='tensor')[:,2]



#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
lensedCL=powers['lensed_scalar']
unlensedCL=powers['unlensed_scalar']
lens_pot = powers['lens_potential']
print(totCL.shape)
EE_lensedCL = lensedCL[:,1]
BB_lensedCL = lensedCL[:,2]

ell_test = np.arange(totCL.shape[0])
Cl_EE = np.nan_to_num((EE_lensedCL * 2*np.pi)/(ell_test*(ell_test+1.)))
Cl_BB = np.nan_to_num((BB_lensedCL * 2*np.pi)/(ell_test*(ell_test+1.)))
Dl_BB_lens = BB_lensedCL
Dl_BB_r = clbb_r*np.arange(clbb_r.shape[0])*(np.arange(clbb_r.shape[0])+1.)/(2.*np.pi)
Dl_BB_r1 = clbb_r1*np.arange(clbb_r.shape[0])*(np.arange(clbb_r.shape[0])+1.)/(2.*np.pi)
Dl_BB_r2 = clbb_r2*np.arange(clbb_r.shape[0])*(np.arange(clbb_r.shape[0])+1.)/(2.*np.pi)
Dl_BB_r3 = clbb_r3*np.arange(clbb_r.shape[0])*(np.arange(clbb_r.shape[0])+1.)/(2.*np.pi)

#embed()
'''
plt.loglog(np.arange(Dl_BB_r.shape[0]),Dl_BB_r,label=r'$r=1$')
plt.loglog(np.arange(Dl_BB_r.shape[0]),Dl_BB_r1,label=r'$r=0.0042$')
plt.loglog(np.arange(Dl_BB_r.shape[0]),Dl_BB_r2,label=r'$r=0.01$')
plt.loglog(np.arange(Dl_BB_r.shape[0]),Dl_BB_r3,label=r'$r=0.1$')
plt.loglog(np.arange(Dl_BB_lens.shape[0]),Dl_BB_lens,label='Lensing')
plt.loglog(ell_vec,tens_vec,label='tens+vec')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell^{BB}$')
plt.legend(loc='best')
plt.show()
'''

clbb_r_200 = clbb_r[20:201]

def delclaa(Nlaa,Claa,fsky,dl,ell):
    return np.sqrt(2./((2.*ell+1.)*dl*fsky))*(Claa+Nlaa)

def delclbb(Nlbb,clbblens,fsky,dl,ell):
    return np.sqrt(2./((2.*ell+1.)*dl*fsky))*(clbblens+Nlbb)

def chi_r(tens,vec,clbb_r,sig_clbb, r):
    return np.sum(((tens+vec-r*clbb_r)/sig_clbb)**2.)

def chi_2r(tens,vec,clbb_r,sig_clbb):
    return np.sum(((tens+vec-clbb_r)/sig_clbb)**2.)

def clbb2r(r):
    return Cosmo({'r':r}).cmb_spectra(220,spec='tensor')[:,2][20:201]

#Advanced ACTPol
fsky= 0.5
noice = 9.8
fwhm = 1.3

clbblens = Cl_BB[20:201]
nlbb = ro.nl_cmb(noice,fwhm)[20:201]
clbblens_ell = np.arange(20,201)
MC_ell = np.arange(20,201)

sig_clbb = delclbb(nlbb,clbblens,fsky,1.,clbblens_ell)
tens_200 = tens[18:199]
tens_200 = np.nan_to_num((tens_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))
vec_200 = vec[18:199]
tens_200 = np.nan_to_num((vec_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))


r_array = np.logspace(-4,0,num = 1000)
clbb_r_array = []
for jj in range(len(r_array)):
    clbb_r_array.append(clbb2r(r_array[jj]))
    print(jj)
    
clbb_r_array = np.array(clbb_r_array)
chi2_act = []
chi22act = []
for i in range(len(r_array)): 
    chi2_act.append(chi_r(tens_200,vec_200,clbb_r_200,sig_clbb,r_array[i]))
    chi22act.append(chi_2r(tens_200,vec_200,clbb_r_array[i],sig_clbb))
    print (i)
chi2_act = np.array(chi2_act)
chi2_min = np.min(chi2_act)
chi2_act = chi2_act - chi2_min
like_act = np.exp(-chi2_act/2.)
chi22act = np.array(chi22act)
chi22min = np.min(chi22act)
chi22act = chi22act - chi22min
like2act = np.exp(-chi22act/2.)

#SA
fsky= 0.65
noice = 11.8
fwhm = 3.5

clbblens = Cl_BB[20:201]
nlbb = ro.nl_cmb(noice,fwhm)[20:201]
clbblens_ell = np.arange(20,201)
MC_ell = np.arange(20,201)

sig_clbb = delclbb(nlbb,clbblens,fsky,1.,clbblens_ell)
tens_200 = tens[18:199]
tens_200 = np.nan_to_num((tens_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))
vec_200 = vec[18:199]
tens_200 = np.nan_to_num((vec_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))

chi2_sa = []
chi22sa = []
for i in range(len(r_array)): 
    chi2_sa.append(chi_r(tens_200,vec_200,clbb_r_200,sig_clbb,r_array[i]))
    chi22sa.append(chi_2r(tens_200,vec_200,clbb_r_array[i],sig_clbb))
    print (i)
chi2_sa = np.array(chi2_sa)
chi2_min = np.min(chi2_sa)
chi2_sa = chi2_sa - chi2_min
like_sa = np.exp(-chi2_sa/2.)
chi22sa = np.array(chi22sa)
chi22min = np.min(chi22sa)
chi22sa = chi22sa - chi22min
like2sa = np.exp(-chi22sa/2.)

#SPT-3G
fsky= 0.06
noice = 4.5
fwhm = 1.1

clbblens = Cl_BB[20:201]
nlbb = ro.nl_cmb(noice,fwhm)[20:201]
clbblens_ell = np.arange(20,201)
MC_ell = np.arange(20,201)

sig_clbb = delclbb(nlbb,clbblens,fsky,1.,clbblens_ell)
tens_200 = tens[18:199]
tens_200 = np.nan_to_num((tens_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))
vec_200 = vec[18:199]
tens_200 = np.nan_to_num((vec_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))

chi2_spt = []
chi22spt = []
for i in range(len(r_array)): 
    chi2_spt.append(chi_r(tens_200,vec_200,clbb_r_200,sig_clbb,r_array[i]))
    chi22spt.append(chi_2r(tens_200,vec_200,clbb_r_array[i],sig_clbb))
    print (i)
chi2_spt = np.array(chi2_spt)
chi2_min = np.min(chi2_spt)
chi2_spt = chi2_spt - chi2_min
like_spt = np.exp(-chi2_spt/2.)
chi22spt = np.array(chi22spt)
chi22min = np.min(chi22spt)
chi22spt = chi22spt - chi22min
like2spt = np.exp(-chi22spt/2.)

#CMBS4
fsky= 0.5
noice = 1.5
fwhm = 3.0

clbblens = Cl_BB[20:201]
nlbb = ro.nl_cmb(noice,fwhm)[20:201]
clbblens_ell = np.arange(20,201)
MC_ell = np.arange(20,201)

sig_clbb = delclbb(nlbb,clbblens,fsky,1.,clbblens_ell)
tens_200 = tens[18:199]
tens_200 = np.nan_to_num((tens_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))
vec_200 = vec[18:199]
tens_200 = np.nan_to_num((vec_200 * 2*np.pi)/(MC_ell*(MC_ell+1.)))

chi2_s4 = []
chi22s4 = []
for i in range(len(r_array)): 
    chi2_s4.append(chi_r(tens_200,vec_200,clbb_r_200,sig_clbb,r_array[i]))
    chi22s4.append(chi_2r(tens_200,vec_200,clbb_r_array[i],sig_clbb))
    print (i)
chi2_s4 = np.array(chi2_s4)
chi2_min = np.min(chi2_s4)
chi2_s4 = chi2_s4 - chi2_min
like_s4 = np.exp(-chi2_s4/2.)
chi22s4 = np.array(chi22s4)
chi22min = np.min(chi22s4)
chi22s4 = chi22s4 - chi22min
like2s4 = np.exp(-chi22s4/2.)


embed()
plt.plot(r_array, np.sqrt(chi2_act),label='AdvAct')
plt.plot(r_array, np.sqrt(chi2_sa),label='SA')
plt.plot(r_array, np.sqrt(chi2_spt),label='SPT-3G')
plt.plot(r_array, np.sqrt(chi2_s4),label='CMBS4')
plt.xscale('log')
plt.xlabel(r'$r$')
plt.ylabel(r'$S/N$')
plt.legend(loc='best')
plt.show()

plt.plot(r_array, like2act,label='AdvAct')
plt.plot(r_array, like2sa,label='SA')
plt.plot(r_array, like2spt,label='SPT-3G')
plt.plot(r_array, like2s4,label='CMBS4')
plt.xscale('log')
plt.xlabel(r'$r$')
plt.ylabel('$Likelihood$')
plt.legend(loc='best')
plt.show()

