import numpy as np
from scipy import interpolate
import configparser
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
data_camb= camb.get_background(pars)

cosmo = Cosmo()
cmbspec = cosmo.cmb_spectra(7000)
#cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(12000,spec='tensor')[:,2]
clbb_r = Cosmo({'r':1.}).cmb_spectra(400,spec='tensor')[:,2]

MC_lin = pickle.load(open("MAGCAMB_ONLY_linear.pkl","rb"))
MC_log = pickle.load(open("MAGCAMB_ONLY_log.pkl","rb"))
FR = pickle.load(open("FINAL_FR.pkl","rb"))
FR_lin = FR['mydic']
FR_log = FR['mydic1']

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(12000, lens_potential_accuracy=0)

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

def delclaa(Nlaa,Claa,fsky,dl,ell):
    return np.sqrt(2./((2.*ell+1.)*dl*fsky))*(Claa+Nlaa)

def delclbb(Nlbb,clbblens,fsky,dl,ell):
    return np.sqrt(2./((2.*ell+1.)*dl*fsky))*(clbblens+Nlbb)

def argmax_3d(array,len_x,len_y,len_z):
    i = np.argmax(array)
    if i==0:
        x,y,z=0,0,0
    else:
        test = np.array(array.flat)
        len_test = len(test)
        hope = np.zeros(len_test)
        hope[i]=i
        new_hope = hope.reshape(len_x,len_y,len_z)
        ind = np.argwhere(new_hope)[0]
    	x,y,z = ind[0],ind[1],ind[2]
    return x,y,z


fsky= 0.5
noice = 9.8
fwhm = 1.3

Nlaa_ell, Nlaa = ro.GimmeNl(cmbspec,fwhm,noice,dl=10, nwanted = 501)
Nlaa_del10 = Nlaa[2:100]
Nlaa_del100 = Nlaa[100::10]
Nlaa_ells_del10 = Nlaa_ell[2:100]
Nlaa_ells_del100 = Nlaa_ell[100::10]
clbblens = Cl_BB[20:4991:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:4991:10]
clbblens_ell = np.arange(0,5001)
clbblens_ell = clbblens_ell[20:4991:10]
MC_ell = np.arange(2,5001)
MC_ells = MC_ell[18:4991:10]

#sig_claa1 = delclaa(Nlaa,fsky,10.,Nlaa_ells)
sig_clbb = delclbb(nlbb,clbblens,fsky,10.,clbblens_ell)

beta = np.arange(6,17.1)
B_G = np.arange(0.5E-9,5.1E-9,0.5E-9)
BnG = B_G*(1.E9)
log_BnG = np.logspace(-3,np.log10(5),num=10)
log_B_G = log_BnG*(1.E9)
#nb_FR = np.linspace(-2.9,1.3,num=10)
nb_FR = np.arange(-2.9,1.31,0.1)
nb_MAGCAMB = np.arange(-2.9,2.1,0.1)
chi_4d = {}
chi_2d = {}
chi4_FRMC_lin = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_lin = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_FRMC_log = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_log = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_lin = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_log = np.zeros((len(beta),len(BnG),len(nb_FR)))

for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
            #vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:4991:10]
	    #tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:4991:10]
            #vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:4991:10] 
	    #tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:4991:10]

            FR_AA_lin_del10 = FR_lin[gg,nn]['claa_FR'][0:98]
            FR_AA_lin_del100 = FR_lin[gg,nn]['claa_FR'][98:139]
            FR_BB_lin = FR_lin[gg,nn]['clBB_FR'][2:]
            FR_AA_log_del10 = FR_log[gg,nn]['claa_FR'][0:98]
            FR_AA_log_del100 = FR_log[gg,nn]['claa_FR'][98:139]
            FR_BB_log = FR_log[gg,nn]['clBB_FR'][2:]
            sig_claa_lin10 = delclaa(Nlaa_del10,FR_AA_lin_del10,fsky,10.,Nlaa_ells_del10)
            sig_claa_lin100 = delclaa(Nlaa_del100,FR_AA_lin_del100,fsky,100.,Nlaa_ells_del100)
            sig_claa_log10 = delclaa(Nlaa_del10,FR_AA_log_del10,fsky,10.,Nlaa_ells_del10)
            sig_claa_log100 = delclaa(Nlaa_del100,FR_AA_log_del100,fsky,100.,Nlaa_ells_del100)	    
	    #embed()
            #chi4_lin = np.sum(((FR_BB_lin+vec_lin+tens_lin-clbblens)/sig_clbb)**2.)*10.
            #chi2_lin = np.sum((FR_AA_lin_del10/sig_claa_lin10)**2.)*10.+np.sum((FR_AA_lin_del100/sig_claa_lin100)**2.)*100.
            #chi4_log = np.sum(((FR_BB_log+vec_log+tens_log-clbblens)/sig_clbb)**2.)*10.
            #chi2_log = np.sum((FR_AA_log_del10/sig_claa_log10)**2.)*10.+np.sum((FR_AA_log_del100/sig_claa_log100)**2.)*100.

            chi4_lin = np.sum(((abs(FR_BB_lin-clbblens))/sig_clbb)**2.)*10.
            chi2_lin = np.sum((FR_AA_lin_del10/sig_claa_lin10)**2.)*10.+np.sum((FR_AA_lin_del100/sig_claa_lin100)**2.)*100.
            chi4_log = np.sum(((abs(FR_BB_lin-clbblens))/sig_clbb)**2.)*10.
            chi2_log = np.sum((FR_AA_log_del10/sig_claa_log10)**2.)*10.+np.sum((FR_AA_log_del100/sig_claa_log100)**2.)*100.      
      	    
            chi_4d[bb,gg,nn] = {}
            chi_2d[bb,gg,nn] = {} 
            chi_4d[bb,gg,nn]['lin'] = chi4_lin 
            chi_4d[bb,gg,nn]['log'] = chi4_log
            chi_2d[bb,gg,nn]['lin'] = chi2_lin
            chi_2d[bb,gg,nn]['log'] = chi2_log
            
	    chi4_FRMC_lin[bb,gg,nn] = chi4_lin
	    chi2_FRMC_lin[bb,gg,nn] = chi2_lin
	    chi4_FRMC_log[bb,gg,nn] = chi4_log
	    chi2_FRMC_log[bb,gg,nn] = chi2_log
	    chi_FRMC_lin[bb,gg,nn] = chi4_lin+chi2_lin
	    chi_FRMC_log[bb,gg,nn] = chi4_log+chi2_log
            print(bb,gg,nn)

clbblens = Cl_BB[20:201:10]
clbblens_ell = np.arange(0,400)
clbblens_ell = clbblens_ell[20:201:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:201:10]
GW_R1 = clbb_r[20:201:10]
sig_clbb = delclbb(nlbb,clbblens,fsky,10.,clbblens_ell)

x_lin,y_lin,z_lin = argmax_3d(chi_FRMC_lin,len(beta),len(BnG),len(nb_FR))
x_log,y_log,z_log = argmax_3d(chi_FRMC_log,len(beta),len(BnG),len(nb_FR))
vec_lin = MC_lin[x_lin,y_lin,z_lin]['clBB_MC_vec'][18:208:10]
tens_lin = MC_lin[x_lin,y_lin,z_lin]['clBB_MC_tens'][18:208:10]
vec_log = MC_log[x_log,y_log,z_log]['clBB_MC_vec'][18:208:10] 
tens_log = MC_log[x_log,y_log,z_log]['clBB_MC_tens'][18:208:10]
FR_BB_lin = FR_lin[gg,nn]['clBB_FR'][2:21]
FR_BB_log = FR_log[gg,nn]['clBB_FR'][2:21]
#MC_FR_lin = FR_BB_lin+vec_lin+tens_lin
#MC_FR_log = FR_BB_log+vec_log+tens_log

num_r_lin = np.sum(((abs(FR_BB_lin-clbblens))*GW_R1/sig_clbb)**2.)*10.
denom_r_lin =np.sum((GW_R1/sig_clbb)**2.)*10.
r_eff_lin = num_r_lin/denom_r_lin
beta_lin = beta[x_lin]
BnG_lin = BnG[y_lin]
nb_FR_lin = nb_FR[z_lin] 

num_r_log = np.sum(((abs(FR_BB_log-clbblens))*GW_R1/sig_clbb)**2.)*10.
denom_r_log =np.sum((GW_R1/sig_clbb)**2.)*10.
r_eff_log = num_r_log/denom_r_log
beta_log = beta[x_log]
BnG_log = BnG[y_log]
nb_FR_log = nb_FR[z_log]

'''
num_r_lin = np.sum(((MC_FR_lin-clbblens)*GW_R1/sig_clbb)**2.)*10.
denom_r_lin =np.sum((GW_R1/sig_clbb)**2.)*10.
r_eff_lin = num_r_lin/denom_r_lin
beta_lin = beta[x_lin]
BnG_lin = BnG[y_lin]
nb_FR_lin = nb_FR[z_lin] 

num_r_log = np.sum(((MC_FR_log-clbblens)*GW_R1/sig_clbb)**2.)*10.
denom_r_log =np.sum((GW_R1/sig_clbb)**2.)*10.
r_eff_log = num_r_log/denom_r_log
beta_log = beta[x_log]
BnG_log = BnG[y_log]
nb_FR_log = nb_FR[z_log] 
'''
embed()


    

    

