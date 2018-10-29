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
from scipy.optimize import curve_fit

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


MC_lin = pickle.load(open("MG_lin_cls.pkl","rb"))
MC_log = pickle.load(open("MG_log_cls.pkl","rb"))
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
clbb_r_200 = clbb_r[20:201]

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


def argmin_3d(array,len_x,len_y,len_z):
    i = np.argmin(array)
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

def clbbr_func(l,r):
    ells=np.arange(20,201)
    GW = clbb_r[20:201]
    sp = UnivariateSpline(ells,GW,k=3,s=0)
    return r*sp(l)
#Advanced ACTPol
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


sig_clbb = delclbb(nlbb,clbblens,fsky,10.,clbblens_ell)

beta = np.arange(6,17.1)
B_G = np.arange(0.5E-9,5.1E-9,0.5E-9)
BnG = B_G*(1.E9)
log_BnG = np.logspace(-3,np.log10(5),num=10)
log_B_G = log_BnG*(1.E9)
nb_FR = np.arange(-2.9,1.31,0.1)
nb_MAGCAMB = np.arange(-2.9,2.1,0.1)
chi_4d = {}
chi_2d = {}
chi4_FRMC_lin1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_lin1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_FRMC_log1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_log1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_lin1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_log1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_tot1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_tot1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_lin1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_log1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_lin_array1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_log_array1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_lin_array1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_log_array1 = np.zeros((len(beta),len(BnG),len(nb_FR)))
MC_FR1 = {}

for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:4991:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:4991:10]
            vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:4991:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:4991:10]

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
            chi4_lin = np.sum(((abs(vec_lin+tens_lin))/sig_clbb)**2.)*10.
            chi2_lin = np.sum((FR_AA_lin_del10/sig_claa_lin10)**2.)*10.+np.sum((FR_AA_lin_del100/sig_claa_lin100)**2.)*100.
            chi4_log = np.sum(((abs(vec_log+tens_log))/sig_clbb)**2.)*10.
            chi2_log = np.sum((FR_AA_log_del10/sig_claa_log10)**2.)*10.+np.sum((FR_AA_log_del100/sig_claa_log100)**2.)*100.

            chi_4d[bb,gg,nn] = {}
            chi_2d[bb,gg,nn] = {} 
            chi_4d[bb,gg,nn]['lin'] = chi4_lin 
            chi_4d[bb,gg,nn]['log'] = chi4_log
            chi_2d[bb,gg,nn]['lin'] = chi2_lin
            chi_2d[bb,gg,nn]['log'] = chi2_log
            
	    chi4_FRMC_lin1[bb,gg,nn] = chi4_lin
	    chi2_FRMC_lin1[bb,gg,nn] = chi2_lin
	    chi4_FRMC_log1[bb,gg,nn] = chi4_log
	    chi2_FRMC_log1[bb,gg,nn] = chi2_log
	    chi_FRMC_lin1[bb,gg,nn] = chi4_lin+chi2_lin
	    chi_FRMC_log1[bb,gg,nn] = chi4_log+chi2_log

	    clbblens_r = Cl_BB[20:201:10]
	    clbblens_ell = np.arange(0,400)
	    clbblens_ell = clbblens_ell[20:201:10]
	    nlbb_r = ro.nl_cmb(noice,fwhm)[20:201:10]
	    GW_R1 = clbb_r[20:201:10]
	    sig_clbb_r = delclbb(nlbb_r,clbblens_r,fsky,10.,clbblens_ell)

            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:208:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:208:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    FR_BB_lin = FR_lin[gg,nn]['clBB_FR'][2:21]
	    FR_BB_log = FR_log[gg,nn]['clBB_FR'][2:21]
	    MC_FR_lin = vec_lin+tens_lin+FR_BB_lin
	    MC_FR_log = vec_log+tens_log+FR_BB_log

	    MC_FR1[bb,gg,nn] = {}
	    MC_FR1[bb,gg,nn]['lin'] = MC_FR_lin
	    MC_FR1[bb,gg,nn]['log'] = MC_FR_log

            xdata = np.arange(0,400)[20:201:10]
	    ydata_lin = abs(MC_FR_lin+clbblens_r)
	    ydata_log = abs(MC_FR_log+clbblens_r)

	    r1_lin,cov1_lin = curve_fit(clbbr_func,xdata,ydata_lin,10E-4,sig_clbb_r)
	    r1_log,cov1_log = curve_fit(clbbr_func,xdata,ydata_log,10E-4,sig_clbb_r)
	    r1_lin = r1_lin[0]
	    r1_log = r1_log[0]

	    xdata1 = xdata[0:9]
	    ydata1_lin = ydata_lin[0:9]
	    ydata1_log = ydata_log[0:9]

	    r2_lin,cov2_lin = curve_fit(clbbr_func,xdata1,ydata1_lin,10E-4,sig_clbb_r[0:9])
	    r2_log,cov2_log = curve_fit(clbbr_func,xdata1,ydata1_log,10E-4,sig_clbb_r[0:9])
	    r2_lin = r2_lin[0]
	    r2_log = r2_log[0]

	    r1_lin_array1[bb,gg,nn] = r1_lin
	    r1_log_array1[bb,gg,nn] = r1_log
	    r2_lin_array1[bb,gg,nn] = r2_lin
	    r2_log_array1[bb,gg,nn] = r2_log

	    
            print(bb,gg,nn)

x_lin,y_lin,z_lin = argmin_3d(chi_FRMC_lin1,len(beta),len(BnG),len(nb_FR))
x_log,y_log,z_log = argmin_3d(chi_FRMC_log1,len(beta),len(BnG),len(nb_FR))

chi_min_lin1 = chi_FRMC_lin1[x_lin,y_lin,z_lin] 
chi_min_log1 = chi_FRMC_log1[x_log,y_log,z_log]
x_chi1,y_chi1,z_chi1 = argmin_3d(chi4_FRMC_lin1,len(beta),len(BnG),len(nb_FR))
chi4_min_lin1 = chi4_FRMC_lin1[x_chi1,y_chi1,z_chi1]

for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
	    Likelihood_lin1[bb,gg,nn] = np.exp((-chi_FRMC_lin1[bb,gg,nn]+chi_min_lin1)/2.)
	    Likelihood_log1[bb,gg,nn] = np.exp((-chi_FRMC_log1[bb,gg,nn]+chi_min_log1)/2.)
	    chi_tot1[bb,gg,nn] = chi_FRMC_lin1[bb,gg,nn]-chi_min_lin1
	    chi4_tot1[bb,gg,nn] = chi4_FRMC_lin1[bb,gg,nn]-chi4_min_lin1

r1_lin_flat1 = np.array(r1_lin_array1.flat) 
r2_lin_flat1 = np.array(r2_lin_array1.flat)  

L_lin_flat1 = np.array(Likelihood_lin1.flat)
L_log_flat1 = np.array(Likelihood_log1.flat)


#SA
fsky= 0.65
noice = 11.8
fwhm = 3.5

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


sig_clbb = delclbb(nlbb,clbblens,fsky,10.,clbblens_ell)

beta = np.arange(6,17.1)
B_G = np.arange(0.5E-9,5.1E-9,0.5E-9)
BnG = B_G*(1.E9)
log_BnG = np.logspace(-3,np.log10(5),num=10)
log_B_G = log_BnG*(1.E9)
nb_FR = np.arange(-2.9,1.31,0.1)
nb_MAGCAMB = np.arange(-2.9,2.1,0.1)
chi_4d = {}
chi_2d = {}
chi4_FRMC_lin2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_lin2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_FRMC_log2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_log2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_lin2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_log2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_tot2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_tot2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_lin2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_log2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_lin_array2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_log_array2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_lin_array2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_log_array2 = np.zeros((len(beta),len(BnG),len(nb_FR)))
MC_FR2 ={}


for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:4991:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:4991:10]
            vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:4991:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:4991:10]

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
            chi4_lin = np.sum(((abs(vec_lin+tens_lin))/sig_clbb)**2.)*10.
            chi2_lin = np.sum((FR_AA_lin_del10/sig_claa_lin10)**2.)*10.+np.sum((FR_AA_lin_del100/sig_claa_lin100)**2.)*100.
            chi4_log = np.sum(((abs(vec_log+tens_log))/sig_clbb)**2.)*10.
            chi2_log = np.sum((FR_AA_log_del10/sig_claa_log10)**2.)*10.+np.sum((FR_AA_log_del100/sig_claa_log100)**2.)*100.

            chi_4d[bb,gg,nn] = {}
            chi_2d[bb,gg,nn] = {} 
            chi_4d[bb,gg,nn]['lin'] = chi4_lin 
            chi_4d[bb,gg,nn]['log'] = chi4_log
            chi_2d[bb,gg,nn]['lin'] = chi2_lin
            chi_2d[bb,gg,nn]['log'] = chi2_log
            
	    chi4_FRMC_lin2[bb,gg,nn] = chi4_lin
	    chi2_FRMC_lin2[bb,gg,nn] = chi2_lin
	    chi4_FRMC_log2[bb,gg,nn] = chi4_log
	    chi2_FRMC_log2[bb,gg,nn] = chi2_log
	    chi_FRMC_lin2[bb,gg,nn] = chi4_lin+chi2_lin
	    chi_FRMC_log2[bb,gg,nn] = chi4_log+chi2_log

	    clbblens_r = Cl_BB[20:201:10]
	    clbblens_ell = np.arange(0,400)
	    clbblens_ell = clbblens_ell[20:201:10]
	    nlbb_r = ro.nl_cmb(noice,fwhm)[20:201:10]
	    GW_R1 = clbb_r[20:201:10]
	    sig_clbb_r = delclbb(nlbb_r,clbblens_r,fsky,10.,clbblens_ell)

            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:208:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:208:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    FR_BB_lin = FR_lin[gg,nn]['clBB_FR'][2:21]
	    FR_BB_log = FR_log[gg,nn]['clBB_FR'][2:21]
	    MC_FR_lin = vec_lin+tens_lin+FR_BB_lin
	    MC_FR_log = vec_log+tens_log+FR_BB_log

	    MC_FR2[bb,gg,nn] = {}
	    MC_FR2[bb,gg,nn]['lin'] = MC_FR_lin
	    MC_FR2[bb,gg,nn]['log'] = MC_FR_log

            xdata = np.arange(0,400)[20:201:10]
	    ydata_lin = abs(MC_FR_lin+clbblens_r)
	    ydata_log = abs(MC_FR_log+clbblens_r)

	    r1_lin,cov1_lin = curve_fit(clbbr_func,xdata,ydata_lin,10E-4,sig_clbb_r)
	    r1_log,cov1_log = curve_fit(clbbr_func,xdata,ydata_log,10E-4,sig_clbb_r)
	    r1_lin = r1_lin[0]
	    r1_log = r1_log[0]

	    xdata1 = xdata[0:9]
	    ydata1_lin = ydata_lin[0:9]
	    ydata1_log = ydata_log[0:9]

	    r2_lin,cov2_lin = curve_fit(clbbr_func,xdata1,ydata1_lin,10E-4,sig_clbb_r[0:9])
	    r2_log,cov2_log = curve_fit(clbbr_func,xdata1,ydata1_log,10E-4,sig_clbb_r[0:9])
	    r2_lin = r2_lin[0]
	    r2_log = r2_log[0]

	    r1_lin_array2[bb,gg,nn] = r1_lin
	    r1_log_array2[bb,gg,nn] = r1_log
	    r2_lin_array2[bb,gg,nn] = r2_lin
	    r2_log_array2[bb,gg,nn] = r2_log

            print(bb,gg,nn)

x_lin,y_lin,z_lin = argmin_3d(chi_FRMC_lin2,len(beta),len(BnG),len(nb_FR))
x_log,y_log,z_log = argmin_3d(chi_FRMC_log2,len(beta),len(BnG),len(nb_FR))
chi_min_lin2 = chi_FRMC_lin2[x_lin,y_lin,z_lin] 
chi_min_log2 = chi_FRMC_log2[x_log,y_log,z_log]
x_chi2,y_chi2,z_chi2 = argmin_3d(chi4_FRMC_lin2,len(beta),len(BnG),len(nb_FR))
chi4_min_lin2 = chi4_FRMC_lin2[x_chi2,y_chi2,z_chi2]

for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
	    Likelihood_lin2[bb,gg,nn] = np.exp(-chi_FRMC_lin2[bb,gg,nn]+chi_min_lin2)
	    Likelihood_log2[bb,gg,nn] = np.exp(-chi_FRMC_log2[bb,gg,nn]+chi_min_log2)
	    chi_tot2[bb,gg,nn] = chi_FRMC_lin2[bb,gg,nn]-chi_min_lin2
	    chi4_tot2[bb,gg,nn] = chi4_FRMC_lin2[bb,gg,nn]-chi4_min_lin2

r1_lin_flat2 = np.array(r1_lin_array2.flat) 
r2_lin_flat2 = np.array(r2_lin_array2.flat)  

L_lin_flat2 = np.array(Likelihood_lin2.flat)
L_log_flat2 = np.array(Likelihood_log2.flat)

#SPT-3G
fsky= 0.06
noice = 4.5
fwhm = 1.1

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


sig_clbb = delclbb(nlbb,clbblens,fsky,10.,clbblens_ell)

beta = np.arange(6,17.1)
B_G = np.arange(0.5E-9,5.1E-9,0.5E-9)
BnG = B_G*(1.E9)
log_BnG = np.logspace(-3,np.log10(5),num=10)
log_B_G = log_BnG*(1.E9)
nb_FR = np.arange(-2.9,1.31,0.1)
nb_MAGCAMB = np.arange(-2.9,2.1,0.1)
chi_4d = {}
chi_2d = {}
chi4_FRMC_lin3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_lin3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_FRMC_log3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_log3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_lin3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_log3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_tot3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_tot3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_lin3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_log3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_lin_array3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_log_array3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_lin_array3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_log_array3 = np.zeros((len(beta),len(BnG),len(nb_FR)))
MC_FR3 = {}



for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:4991:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:4991:10]
            vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:4991:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:4991:10]

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
            chi4_lin = np.sum(((abs(vec_lin+tens_lin))/sig_clbb)**2.)*10.
            chi2_lin = np.sum((FR_AA_lin_del10/sig_claa_lin10)**2.)*10.+np.sum((FR_AA_lin_del100/sig_claa_lin100)**2.)*100.
            chi4_log = np.sum(((abs(vec_log+tens_log))/sig_clbb)**2.)*10.
            chi2_log = np.sum((FR_AA_log_del10/sig_claa_log10)**2.)*10.+np.sum((FR_AA_log_del100/sig_claa_log100)**2.)*100.

            chi_4d[bb,gg,nn] = {}
            chi_2d[bb,gg,nn] = {} 
            chi_4d[bb,gg,nn]['lin'] = chi4_lin 
            chi_4d[bb,gg,nn]['log'] = chi4_log
            chi_2d[bb,gg,nn]['lin'] = chi2_lin
            chi_2d[bb,gg,nn]['log'] = chi2_log
            
	    chi4_FRMC_lin3[bb,gg,nn] = chi4_lin
	    chi2_FRMC_lin3[bb,gg,nn] = chi2_lin
	    chi4_FRMC_log3[bb,gg,nn] = chi4_log
	    chi2_FRMC_log3[bb,gg,nn] = chi2_log
	    chi_FRMC_lin3[bb,gg,nn] = chi4_lin+chi2_lin
	    chi_FRMC_log3[bb,gg,nn] = chi4_log+chi2_log

	    clbblens_r = Cl_BB[20:201:10]
	    clbblens_ell = np.arange(0,400)
	    clbblens_ell = clbblens_ell[20:201:10]
	    nlbb_r = ro.nl_cmb(noice,fwhm)[20:201:10]
	    GW_R1 = clbb_r[20:201:10]
	    sig_clbb_r = delclbb(nlbb_r,clbblens_r,fsky,10.,clbblens_ell)

	    vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:208:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:208:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    FR_BB_lin = FR_lin[gg,nn]['clBB_FR'][2:21]
	    FR_BB_log = FR_log[gg,nn]['clBB_FR'][2:21]
	    MC_FR_lin = vec_lin+tens_lin+FR_BB_lin
	    MC_FR_log = vec_log+tens_log+FR_BB_log

	    MC_FR3[bb,gg,nn] = {}
	    MC_FR3[bb,gg,nn]['lin'] = MC_FR_lin
	    MC_FR3[bb,gg,nn]['log'] = MC_FR_log

            xdata = np.arange(0,400)[20:201:10]
	    ydata_lin = abs(MC_FR_lin+clbblens_r)
	    ydata_log = abs(MC_FR_log+clbblens_r)

	    r1_lin,cov1_lin = curve_fit(clbbr_func,xdata,ydata_lin,10E-4,sig_clbb_r)
	    r1_log,cov1_log = curve_fit(clbbr_func,xdata,ydata_log,10E-4,sig_clbb_r)
	    r1_lin = r1_lin[0]
	    r1_log = r1_log[0]

	    xdata1 = xdata[0:9]
	    ydata1_lin = ydata_lin[0:9]
	    ydata1_log = ydata_log[0:9]

	    r2_lin,cov2_lin = curve_fit(clbbr_func,xdata1,ydata1_lin,10E-4,sig_clbb_r[0:9])
	    r2_log,cov2_log = curve_fit(clbbr_func,xdata1,ydata1_log,10E-4,sig_clbb_r[0:9])
	    r2_lin = r2_lin[0]
	    r2_log = r2_log[0]

	    r1_lin_array3[bb,gg,nn] = r1_lin
	    r1_log_array3[bb,gg,nn] = r1_log
	    r2_lin_array3[bb,gg,nn] = r2_lin
	    r2_log_array3[bb,gg,nn] = r2_log

            print(bb,gg,nn)

x_lin,y_lin,z_lin = argmin_3d(chi_FRMC_lin3,len(beta),len(BnG),len(nb_FR))
x_log,y_log,z_log = argmin_3d(chi_FRMC_log3,len(beta),len(BnG),len(nb_FR))
chi_min_lin3 = chi_FRMC_lin3[x_lin,y_lin,z_lin] 
chi_min_log3 = chi_FRMC_log3[x_log,y_log,z_log]
x_chi3,y_chi3,z_chi3 = argmin_3d(chi4_FRMC_lin3,len(beta),len(BnG),len(nb_FR))
chi4_min_lin3 = chi4_FRMC_lin3[x_chi3,y_chi3,z_chi3]

for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
	    Likelihood_lin3[bb,gg,nn] = np.exp(-chi_FRMC_lin3[bb,gg,nn]+chi_min_lin3)
	    Likelihood_log3[bb,gg,nn] = np.exp(-chi_FRMC_log3[bb,gg,nn]+chi_min_log3)
	    chi_tot3[bb,gg,nn] = chi_FRMC_lin3[bb,gg,nn]-chi_min_lin3
	    chi4_tot3[bb,gg,nn] = chi4_FRMC_lin3[bb,gg,nn]-chi4_min_lin3

r1_lin_flat3 = np.array(r1_lin_array3.flat) 
r2_lin_flat3 = np.array(r2_lin_array3.flat) 

L_lin_flat3 = np.array(Likelihood_lin3.flat)
L_log_flat3 = np.array(Likelihood_log3.flat)

#CMBS4
fsky= 0.5
noice = 1.5
fwhm = 3.0

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


sig_clbb = delclbb(nlbb,clbblens,fsky,10.,clbblens_ell)

beta = np.arange(6,17.1)
B_G = np.arange(0.5E-9,5.1E-9,0.5E-9)
BnG = B_G*(1.E9)
log_BnG = np.logspace(-3,np.log10(5),num=10)
log_B_G = log_BnG*(1.E9)
nb_FR = np.arange(-2.9,1.31,0.1)
nb_MAGCAMB = np.arange(-2.9,2.1,0.1)
chi_4d = {}
chi_2d = {}
chi4_FRMC_lin4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_lin4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_FRMC_log4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi2_FRMC_log4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_lin4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_FRMC_log4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi_tot4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
chi4_tot4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_lin4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
Likelihood_log4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_lin_array4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r1_log_array4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_lin_array4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
r2_log_array4 = np.zeros((len(beta),len(BnG),len(nb_FR)))
MC_FR4 = {}


for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:4991:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:4991:10]
            vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:4991:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:4991:10]

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
            chi4_lin = np.sum(((abs(vec_lin+tens_lin))/sig_clbb)**2.)*10.
            chi2_lin = np.sum((FR_AA_lin_del10/sig_claa_lin10)**2.)*10.+np.sum((FR_AA_lin_del100/sig_claa_lin100)**2.)*100.
            chi4_log = np.sum(((abs(vec_log+tens_log))/sig_clbb)**2.)*10.
            chi2_log = np.sum((FR_AA_log_del10/sig_claa_log10)**2.)*10.+np.sum((FR_AA_log_del100/sig_claa_log100)**2.)*100.

            chi_4d[bb,gg,nn] = {}
            chi_2d[bb,gg,nn] = {} 
            chi_4d[bb,gg,nn]['lin'] = chi4_lin 
            chi_4d[bb,gg,nn]['log'] = chi4_log
            chi_2d[bb,gg,nn]['lin'] = chi2_lin
            chi_2d[bb,gg,nn]['log'] = chi2_log
            
	    chi4_FRMC_lin4[bb,gg,nn] = chi4_lin
	    chi2_FRMC_lin4[bb,gg,nn] = chi2_lin
	    chi4_FRMC_log4[bb,gg,nn] = chi4_log
	    chi2_FRMC_log4[bb,gg,nn] = chi2_log
	    chi_FRMC_lin4[bb,gg,nn] = chi4_lin+chi2_lin
	    chi_FRMC_log4[bb,gg,nn] = chi4_log+chi2_log

	    clbblens_r = Cl_BB[20:201:10]
	    clbblens_ell = np.arange(0,400)
	    clbblens_ell = clbblens_ell[20:201:10]
	    nlbb_r = ro.nl_cmb(noice,fwhm)[20:201:10]
	    GW_R1 = clbb_r[20:201:10]
	    sig_clbb_r = delclbb(nlbb_r,clbblens_r,fsky,10.,clbblens_ell)

            vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec'][18:208:10]
	    tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    vec_log = MC_log[bb,gg,nn]['clBB_MC_vec'][18:208:10] 
	    tens_log = MC_log[bb,gg,nn]['clBB_MC_tens'][18:208:10]
	    FR_BB_lin = FR_lin[gg,nn]['clBB_FR'][2:21]
	    FR_BB_log = FR_log[gg,nn]['clBB_FR'][2:21]
	    MC_FR_lin = vec_lin+tens_lin+FR_BB_lin
	    MC_FR_log = vec_log+tens_log+FR_BB_log

	    MC_FR4[bb,gg,nn] = {}
	    MC_FR4[bb,gg,nn]['lin'] = MC_FR_lin
	    MC_FR4[bb,gg,nn]['log'] = MC_FR_log

            xdata = np.arange(0,400)[20:201:10]
	    ydata_lin = abs(MC_FR_lin+clbblens_r)
	    ydata_log = abs(MC_FR_log+clbblens_r)

	    r1_lin,cov1_lin = curve_fit(clbbr_func,xdata,ydata_lin,10E-4,sig_clbb_r)
	    r1_log,cov1_log = curve_fit(clbbr_func,xdata,ydata_log,10E-4,sig_clbb_r)
	    r1_lin = r1_lin[0]
	    r1_log = r1_log[0]

	    xdata1 = xdata[0:9]
	    ydata1_lin = ydata_lin[0:9]
	    ydata1_log = ydata_log[0:9]

	    r2_lin,cov2_lin = curve_fit(clbbr_func,xdata1,ydata1_lin,10E-4,sig_clbb_r[0:9])
	    r2_log,cov2_log = curve_fit(clbbr_func,xdata1,ydata1_log,10E-4,sig_clbb_r[0:9])
	    r2_lin = r2_lin[0]
	    r2_log = r2_log[0]

	    r1_lin_array4[bb,gg,nn] = r1_lin
	    r1_log_array4[bb,gg,nn] = r1_log
	    r2_lin_array4[bb,gg,nn] = r2_lin
	    r2_log_array4[bb,gg,nn] = r2_log

            print(bb,gg,nn)

x_lin,y_lin,z_lin = argmin_3d(chi_FRMC_lin4,len(beta),len(BnG),len(nb_FR))
x_log,y_log,z_log = argmin_3d(chi_FRMC_log4,len(beta),len(BnG),len(nb_FR))
chi_min_lin4 = chi_FRMC_lin4[x_lin,y_lin,z_lin] 
chi_min_log4 = chi_FRMC_log4[x_log,y_log,z_log]
x_chi4,y_chi4,z_chi4 = argmin_3d(chi4_FRMC_lin4,len(beta),len(BnG),len(nb_FR))
chi4_min_lin4 = chi4_FRMC_lin4[x_chi4,y_chi4,z_chi4]


a,b,c = 10,9,35 
vec_lin = MC_lin[a,b,c]['clBB_MC_vec'][18:4991:10]
tens_lin = MC_lin[a,b,c]['clBB_MC_tens'][18:4991:10]
PMF_lin = ((abs(vec_lin+tens_lin))/sig_clbb)**2.
plt.loglog(MC_ells,PMF_lin,label=r'$\beta = %.1f, B_{\rm{1Mpc}} = %.1f \rm{nG}, n_B = %.1f$' %(beta[a],BnG[b],nb_FR[c]))
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\chi^2_{2pt}$')
plt.show()

for bb in range(len(beta)):
    for gg in range(len(BnG)):
        for nn in range(len(nb_FR)):
	    Likelihood_lin4[bb,gg,nn] = np.exp(-chi_FRMC_lin4[bb,gg,nn]+chi_min_lin4)
	    Likelihood_log4[bb,gg,nn] = np.exp(-chi_FRMC_log4[bb,gg,nn]+chi_min_log4)
	    chi_tot4[bb,gg,nn] = chi_FRMC_lin4[bb,gg,nn]-chi_min_lin4
	    chi4_tot4[bb,gg,nn] = chi4_FRMC_lin4[bb,gg,nn]-chi4_min_lin4

r1_lin_flat4 = np.array(r1_lin_array4.flat) 
r2_lin_flat4 = np.array(r2_lin_array4.flat) 

L_lin_flat4 = np.array(Likelihood_lin4.flat)
L_log_flat4 = np.array(Likelihood_log4.flat)

def posterior_2d(L_array,beta_size,BnG_size,nb_size,min_var = 'beta'):
    L2d_array = []
    if min_var == 'beta':
        L2d_array = np.zeros(BnG_size,nb_size)
        for i in range(BnG_size):
            for j in range(nb_size):
                L2d_array[i,j] = np.trapz(L_array[:,i,j],x=beta)
    if min_var == 'BnG':
        L2d_array = np.zeros(beta_size,nb_size)
        for i in range(beta_size):
            for j in range(nb_size):
                L2d_array[i,j] = np.trapz(L_array[i,:,j],x=BnG)    
    if min_var == 'log_BnG':
        L2d_array = np.zeros(beta_size,nb_size)
        for i in range(beta_size):
            for j in range(nb_size):
                L2d_array[i,j] = np.trapz(L_array[i,:,j],x=log_BnG)   
    if min_var == 'nb':
        L2d_array = np.zeros(beta_size,BnG_size)
        for i in range(beta_size):
            for j in range(BnG_size):
                L2d_array[i,j] = np.trapz(L_array[i,j,:],x=nb_FR)  
    return L2d_array

def posterior_1d(L2_array,beta_size,BnG_size,nb_size,post = 'beta'):
    L1d_array = []
    if post == 'beta':
    #Need beta_nb
        L1d_array = np.zeros(beta_size)
        for i in range(nb_size):
            L1d_array[i] = np.trapz(L2_array[:,i],x=nb_FR)
    #Need BnG_nb
    if post == 'BnG':
        L1d_array = np.zeros(BnG_size)
        for i in range(nb_size):
            L1d_array[i] = np.trapz(L2_array[:,i],x=nb_FR)  
    #Need log_BnG_nb  
    if post == 'log_BnG':
        L1d_array = np.zeros(BnG_size)
        for i in range(nb_size):
            L1d_array[i] = np.trapz(L2_array[:,i],x=nb_FR)  
    #Need beta_nb
    if post == 'nb':
        L1d_array = np.zeros(nb_size)
        for i in range(beta_size):
            L1d_array[i] = np.trapz(L2_array[i,:],x=beta)  
    return L1d_array

def index_1_3(index1d,len_x,len_y,len_z):
    if index1d == 0:
        x,y,z = 0,0,0
    else:
        hope = np.zeros(len_x*len_y*len_z)
        hope[index1d]=index1d
        new_hope = hope.reshape(len_x,len_y,len_z)
        ind = np.argwhere(new_hope)[0]
        x,y,z = ind[0],ind[1],ind[2]
    return x,y,z

index_1_3(172,len(beta),len(BnG),len(nb_FR))
 
embed()

plt.figure(1)
plt.errorbar(clbblens_ell,abs(MC_FR_log-clbblens),yerr=sig_clbb)
plt.loglog(clbblens_ell,abs(MC_FR_log-clbblens))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.figure(2)
plt.loglog(clbblens_ell,GW_R1*r_eff_log4)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.figure(3)
plt.loglog(clbblens_ell,abs(MC_FR_log-clbblens))
plt.loglog(clbblens_ell,GW_R1*r_eff_log4)
plt.errorbar(clbblens_ell,abs(MC_FR_log-clbblens),yerr=sig_clbb)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.show()

plt.figure(1)
plt.errorbar(clbblens_ell,abs(MC_FR_log-clbblens),yerr=sig_clbb)
plt.loglog(clbblens_ell,abs(MC_FR_log-clbblens))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.figure(2)
plt.loglog(clbblens_ell,GW_R1*r1)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.figure(3)
plt.loglog(clbblens_ell,abs(MC_FR_log-clbblens))
plt.loglog(clbblens_ell,GW_R1*r1)
plt.errorbar(clbblens_ell,abs(MC_FR_log-clbblens),yerr=sig_clbb)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.show()

plt.loglog(clbblens_ell[200:490],clbblens[200:490])
plt.errorbar(clbblens_ell[200:490],clbblens[200:490],yerr=sig_clbb[200:490])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{BB}$')
plt.show()

L_tot1 = np.exp(-chi_tot1/2.)
L_tot2 = np.exp(-chi_tot2/2.)
L_tot3 = np.exp(-chi_tot3/2.)
L_tot4 = np.exp(-chi_tot4/2.)
L2_lin1 = np.mean(chi_tot1,axis=(0,1))
L1_lin1 = np.mean(chi_tot1,axis=(0,2))
L0_lin1 = np.mean(chi_tot1,axis=(1,2))
L12_lin1 = np.mean(chi_tot1,axis=0)
L02_lin1 = np.mean(chi_tot1,axis=1)    
L01_lin1 = np.mean(chi_tot1,axis=2) 

L2_lin2 = np.mean(chi_tot2,axis=(0,1))
L1_lin2 = np.mean(chi_tot2,axis=(0,2))
L0_lin2 = np.mean(chi_tot2,axis=(1,2))
L12_lin2 = np.mean(chi_tot2,axis=0)
L02_lin2 = np.mean(chi_tot2,axis=1)    
L01_lin2 = np.mean(chi_tot2,axis=2) 

L2_lin3 = np.mean(chi_tot3,axis=(0,1))
L1_lin3 = np.mean(chi_tot3,axis=(0,2))
L0_lin3 = np.mean(chi_tot3,axis=(1,2))
L12_lin3 = np.mean(chi_tot3,axis=0)
L02_lin3 = np.mean(chi_tot3,axis=1)    
L01_lin3 = np.mean(chi_tot3,axis=2) 

L2_lin4 = np.mean(chi_tot4,axis=(0,1))
L1_lin4 = np.mean(chi_tot4,axis=(0,2))
L0_lin4 = np.mean(chi_tot4,axis=(1,2))
L12_lin4 = np.mean(chi_tot4,axis=0)
L02_lin4 = np.mean(chi_tot4,axis=1)    
L01_lin4 = np.mean(chi_tot4,axis=2) 

L2_lin1 = np.mean(L_tot1,axis=(0,1))
L1_lin1 = np.mean(L_tot1,axis=(0,2))
L0_lin1 = np.mean(L_tot1,axis=(1,2))
L12_lin1 = np.mean(L_tot1,axis=0)
L02_lin1 = np.mean(L_tot1,axis=1)    
L01_lin1 = np.mean(L_tot1,axis=2) 

L2_lin2 = np.mean(L_tot2,axis=(0,1))
L1_lin2 = np.mean(L_tot2,axis=(0,2))
L0_lin2 = np.mean(L_tot2,axis=(1,2))
L12_lin2 = np.mean(L_tot2,axis=0)
L02_lin2 = np.mean(L_tot2,axis=1)    
L01_lin2 = np.mean(L_tot2,axis=2) 

L2_lin3 = np.mean(L_tot3,axis=(0,1))
L1_lin3 = np.mean(L_tot3,axis=(0,2))
L0_lin3 = np.mean(L_tot3,axis=(1,2))
L12_lin3 = np.mean(L_tot3,axis=0)
L02_lin3 = np.mean(L_tot3,axis=1)    
L01_lin3 = np.mean(L_tot3,axis=2) 

L2_lin4 = np.mean(L_tot4,axis=(0,1))
L1_lin4 = np.mean(L_tot4,axis=(0,2))
L0_lin4 = np.mean(L_tot4,axis=(1,2))
L12_lin4 = np.mean(L_tot4,axis=0)
L02_lin4 = np.mean(L_tot4,axis=1)    
L01_lin4 = np.mean(L_tot4,axis=2) 


plt.subplot(3,3,4)
plt.contourf(nb_FR,beta,L02_lin3)
plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin3)
plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR,BnG,L12_lin3)
plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin3)
plt.xlabel(r'$\beta$')
plt.ylabel('Likelihood')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin3)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.ylabel('Likelihood')
plt.subplot(3,3,9)
plt.plot(nb_FR,L2_lin3)
plt.xlabel(r'$n_B$')
plt.ylabel('Likelihood')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR[0:23],beta,L02_lin3[:,0:23])
plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG[0:3],beta,L01_lin3[:,0:3])
plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR[0:23],BnG[0:3],L12_lin3[0:3,0:23])
plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin3)
plt.xlabel(r'$\beta$')
plt.ylabel('Likelihood')
plt.subplot(3,3,5)
plt.plot(BnG[0:3],L1_lin3[0:3])
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.ylabel('Likelihood')
plt.subplot(3,3,9)
plt.plot(nb_FR[0:23],L2_lin3[0:23])
plt.xlabel(r'$n_B$')
plt.ylabel('Likelihood')
plt.show()


plt.subplot(3,3,4)
plt.contourf(nb_FR,beta,L02_lin3)
plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin3)
plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR,BnG,L12_lin3)
plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin3)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$r$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin3)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.ylabel(r'$r$')
plt.subplot(3,3,9)
plt.plot(nb_FR,L2_lin3)
plt.xlabel(r'$n_B$')
plt.ylabel(r'$r$')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR,beta,L02_lin2,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin2,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR,BnG,L12_lin2,level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin2)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin2)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR,L2_lin2)
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR,beta,L02_lin3,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin3,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR,BnG,L12_lin3,level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin3)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin3)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR,L2_lin3)
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR,beta,L02_lin4,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin4,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR,BnG,L12_lin4,level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin4)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin4)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR,L2_lin4)
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()

L4_tot1 = np.exp(-chi4_tot1/2.)
L4_tot2 = np.exp(-chi4_tot2/2.)
L4_tot3 = np.exp(-chi4_tot3/2.)
L4_tot4 = np.exp(-chi4_tot4/2.)

L2_lin1 = np.mean(L4_tot1,axis=(0,1))
L1_lin1 = np.mean(L4_tot1,axis=(0,2))
L0_lin1 = np.mean(L4_tot1,axis=(1,2))
L12_lin1 = np.mean(L4_tot1,axis=0)
L02_lin1 = np.mean(L4_tot1,axis=1)    
L01_lin1 = np.mean(L4_tot1,axis=2) 

L2_lin2 = np.mean(L4_tot2,axis=(0,1))
L1_lin2 = np.mean(L4_tot2,axis=(0,2))
L0_lin2 = np.mean(L4_tot2,axis=(1,2))
L12_lin2 = np.mean(L4_tot2,axis=0)
L02_lin2 = np.mean(L4_tot2,axis=1)    
L01_lin2 = np.mean(L4_tot2,axis=2) 

L2_lin3 = np.mean(L4_tot3,axis=(0,1))
L1_lin3 = np.mean(L4_tot3,axis=(0,2))
L0_lin3 = np.mean(L4_tot3,axis=(1,2))
L12_lin3 = np.mean(L4_tot3,axis=0)
L02_lin3 = np.mean(L4_tot3,axis=1)    
L01_lin3 = np.mean(L4_tot3,axis=2) 

L2_lin4 = np.mean(L4_tot4,axis=(0,1))
L1_lin4 = np.mean(L4_tot4,axis=(0,2))
L0_lin4 = np.mean(L4_tot4,axis=(1,2))
L12_lin4 = np.mean(L4_tot4,axis=0)
L02_lin4 = np.mean(L4_tot4,axis=1)    
L01_lin4 = np.mean(L4_tot4,axis=2) 
##########################################
L2_lin1 = np.mean(chi4_tot1,axis=(0,1))
L1_lin1 = np.mean(chi4_tot1,axis=(0,2))
L0_lin1 = np.mean(chi4_tot1,axis=(1,2))
L12_lin1 = np.mean(chi4_tot1,axis=0)
L02_lin1 = np.mean(chi4_tot1,axis=1)    
L01_lin1 = np.mean(chi4_tot1,axis=2) 

L2_lin2 = np.mean(chi4_tot2,axis=(0,1))
L1_lin2 = np.mean(chi4_tot2,axis=(0,2))
L0_lin2 = np.mean(chi4_tot2,axis=(1,2))
L12_lin2 = np.mean(chi4_tot2,axis=0)
L02_lin2 = np.mean(chi4_tot2,axis=1)    
L01_lin2 = np.mean(chi4_tot2,axis=2) 

L2_lin3 = np.mean(chi4_tot3,axis=(0,1))
L1_lin3 = np.mean(chi4_tot3,axis=(0,2))
L0_lin3 = np.mean(chi4_tot3,axis=(1,2))
L12_lin3 = np.mean(chi4_tot3,axis=0)
L02_lin3 = np.mean(chi4_tot3,axis=1)    
L01_lin3 = np.mean(chi4_tot3,axis=2) 

L2_lin4 = np.mean(chi4_tot4,axis=(0,1))
L1_lin4 = np.mean(chi4_tot4,axis=(0,2))
L0_lin4 = np.mean(chi4_tot4,axis=(1,2))
L12_lin4 = np.mean(chi4_tot4,axis=0)
L02_lin4 = np.mean(chi4_tot4,axis=1)    
L01_lin4 = np.mean(chi4_tot4,axis=2) 

#####################################

plt.subplot(3,3,4)
plt.contourf(nb_FR[0:25],beta,L02_lin1[:,0:25],level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin1,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR[0:25],BnG,L12_lin1[:,0:25],level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin1)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin1)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR[0:25],L2_lin1[0:25])
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR[0:25],beta,L02_lin2[:,0:25],level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin2,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR[0:25],BnG,L12_lin2[:,0:25],level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin2)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin2)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR[0:25],L2_lin2[0:25])
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR[0:25],beta,L02_lin3[:,0:25],level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin3,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR[0:25],BnG,L12_lin3[:,0:25],level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin3)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin3)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR[0:25],L2_lin3[0:25])
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()

plt.subplot(3,3,4)
plt.contourf(nb_FR[0:15],beta,L02_lin4[:,0:15],level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,7)
plt.contourf(BnG,beta,L01_lin4,level=2.71),plt.colorbar()
plt.ylabel(r'$\beta$')
plt.xlabel(r'$B_{\rm{1Mpc}}$')
plt.subplot(3,3,8)
plt.contourf(nb_FR[0:15],BnG,L12_lin4[:,0:15],level=2.71),plt.colorbar()
plt.ylabel(r'$B_{\rm{1Mpc}}$')
plt.xlabel(r'$n_B$')
plt.subplot(3,3,1)
plt.plot(beta,L0_lin4)
plt.xlabel(r'$\beta$')
#plt.ylabel(r'$\n_B$')
plt.subplot(3,3,5)
plt.plot(BnG,L1_lin4)
plt.xlabel(r'$B_{\rm{1Mpc}}$')
#plt.ylabel(r'$\B_{\text{1Mpc}}$')
plt.subplot(3,3,9)
plt.plot(nb_FR[0:15],L2_lin4[0:15])
plt.xlabel(r'$n_B$')
#plt.ylabel(r'$\n_B$')
plt.show()
#################################
L2_lin3 = np.mean(r1_lin_array3,axis=(0,1))
L1_lin3 = np.mean(r1_lin_array3,axis=(0,2))
L0_lin3 = np.mean(r1_lin_array3,axis=(1,2))
L12_lin3 = np.mean(r1_lin_array3,axis=0)
L02_lin3 = np.mean(r1_lin_array3,axis=1)    
L01_lin3 = np.mean(r1_lin_array3,axis=2)

