#http://camb.readthedocs.io/en/latest/CAMBdemo.html

#%matplotlib inline
import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower
print('CAMB version: %s '%camb.__version__)

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

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

#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
PP_lenspot = lens_pot[:,0]

TT_totCL = totCL[:,0]
EE_totCL = totCL[:,1]
BB_totCL = totCL[:,2]
TE_totCL = totCL[:,3]
TT_unlensedCL = unlensedCL[:,0]
EE_unlensedCL = unlensedCL[:,1]
BB_unlensedCL = unlensedCL[:,2]
TE_unlensedCL = unlensedCL[:,3]

ells = np.arange(totCL.shape[0])

Cl_TT = np.nan_to_num((TT_unlensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_EE = np.nan_to_num((EE_unlensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_BB = np.nan_to_num((BB_unlensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_TE = np.nan_to_num((TE_unlensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_PP = np.nan_to_num((PP_lenspot  * 2*np.pi)/(ells*(ells+1)))**2

embed()
