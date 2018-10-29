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
'''
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
n_B = 1.3
B_l = 1.0E-9
Bl = B_l*(1.E9)

vec_lin = MC_lin[bb,gg,nn]['clBB_MC_vec']
tens_lin = MC_lin[bb,gg,nn]['clBB_MC_tens']
vec_log = MC_log[bb,gg,nn]['clBB_MC_vec']
tens_log = MC_log[bb,gg,nn]['clBB_MC_tens']

vec_lin = vec_lin*7.42835025e12 
tens_lin = tens_lin*7.42835025e12
vec_log = vec_log*7.42835025e12 
tens_log = tens_log*7.42835025e12  

data = pickle.load(open("../transfer.pkl","rb"))
ell_array = data['ell']
k_array = data['k']
Tl_m1k = data['Tl_m1k']
Tl_p1k = data['Tl_p1k']
T1lk = data['T1lk'] 


data1 = pickle.load(open("../transfer100.pkl","rb"))
ell_array1 = data['ell']
k_array1 = data['k']
Tl_m1k1 = data['Tl_m1k']
Tl_p1k1 = data['Tl_p1k']
T1lk1 = data['T1lk'] 

#cosmo = Cosmo()
#cmbspec = cosmo.cmb_spectra(12000)
#cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(12000,spec='tensor')[:,2]

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

def GimmeClBBRot(clee, l_aa, claa, dl=10, n=1024, nwanted=500):
	lxgrid, lygrid  = np.meshgrid(np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
	lgrid = np.sqrt(lxgrid**2 + lygrid**2)
	L     = np.arange(0,nwanted)*dl    
	
	eepowerspec2d = np.interp(lgrid, np.arange(clee.shape[0]), clee)
	
	ell = np.arange(clee.shape[0])
	
	#embed()				

	aapowerspec2d = np.interp(lgrid, l_aa, claa)
	
	clrot = ro.ClBB_rot(eepowerspec2d, aapowerspec2d, dl, n, nwanted)
	clrot = clrot[0:nwanted]
	return L, clrot

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

#embed()

#def T_lk_integrand(L,k,eta):
#    return tau_dot_eta(eta)*spherical_jn(L,k*(eta_0-eta))
@numba.jit
def del_m2(k,nb,Blamba):
    return k**2.*S_0(nb,Blamba)*k**n_B*((3*c_cgs**2.)/(16*np.pi**2.*e_cgs*f_cgs**2.))**2.

#quad?
@numba.jit
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
    #embed()
    sp = UnivariateSpline(k_array[np.argmax(abs_integrand):-1], abs_integrand[np.argmax(abs_integrand):-1], k=3, s=0)
    return sp.integral(k_array[np.argmax(abs_integrand)],k_array[-1])
#    sp = UnivariateSpline(k_array, abs_integrand, k=1, s=0)
#    return sp.integral(k_array[np.argmax(abs_integrand)],k_array[-1])

@numba.jit
def Cl_aa_integrand1(k_index,l_index,nb,B_lamba):
    k = k_array1[k_index]
    l = ell_array1[l_index]
    return del_m2(k,nb,B_lamba)*((l/(2.*l+1))*Tl_m1k1[l_index,k_index]**2.+((l+1)/(2.*l+1))*Tl_p1k1[l_index,k_index]**2.-T1lk1[l_index,k_index]**2.)

@numba.jit
def Cl_aa_spline1(l_index,nb,B_lamba):
    integrand = []
    for jj in range(len(k_array)):
       integrand.append(Cl_aa_integrand1(jj,l_index,nb,B_lamba))
       #print(jj)
    integrand = np.array(integrand)
    abs_integrand = np.abs(integrand)
    #embed()
    sp = UnivariateSpline(k_array[np.argmax(abs_integrand):-1], abs_integrand[np.argmax(abs_integrand):-1], k=3, s=0)
    return sp.integral(k_array[np.argmax(abs_integrand)],k_array[-1])
'''
beta = np.arange(6.,6.1)
B_G = np.arange(0.5E-9,5.1E-9,0.5E-9)
BnG = B_G*(1.E9)
BnG = np.arange(1.5,3.0,0.1)
log_BnG = np.logspace(-3,np.log10(5),num=10)
#nb_FR = np.linspace(-2.9,1.3,num=10)
nb_MAGCAMB = np.arange(-2.9,2.1,0.1)
nb_MAGCAMB = np.arange(-2.9,-2.7,0.02)

mydic = {}
def func(bb,gg,nn):
            config = configparser.ConfigParser()
	    config.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/params_mag_copy.ini')

	    output = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/testing_copy'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
	    config['output'] = {'output_root' : output}

	    config['cls'] = {'get_scalar_cls' :'F', 'get_vector_cls' : 'T','get_tensor_cls ': 'F', 'get_transfer': 'F'}

	    config['mag'] = {'magnetic_mode' : '1', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}
            config['norm'] = {'CMB_outputscale' : 7.42835025e12 }

	    with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile:
    	        config.write(configfile)

	    os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
            #embed()
            ell = np.loadtxt(output+'_vecCls.dat')[:,0]
            vec = np.loadtxt(output+'_vecCls.dat')[:,3]            


            config1 = configparser.ConfigParser()
	    config1.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/params_mag_copy.ini')

	    output1 = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/testing_copy2'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
	    config1['output'] = {'output_root' : output1}

	    config1['cls'] = {'get_scalar_cls' :'T', 'get_vector_cls' : 'F','get_tensor_cls ': 'T','get_transfer': 'F'}

	    config1['mag'] = {'magnetic_mode' : '2', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}


            config1['norm'] = {'CMB_outputscale' : 7.42835025e12 }
	    with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile1:
    	        config1.write(configfile1)

	    os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
            ell1 = np.loadtxt(output1+'_tensCls.dat')[:,0]
            tens = np.loadtxt(output1+'_tensCls.dat')[:,3] 

	    '''
	    Claa = []
	    for ii in range(len(ell_array)):
	        Claa.append(Cl_aa_spline(ii,nb_FR[nn],B_G[gg]))
	        #print(ii)
	    Claa = np.array(Claa)

	    Claa1 = []
	    for ii in range(len(ell_array1)):
	        Claa1.append(Cl_aa_spline1(ii,nb_FR[nn],B_G[gg]))
	        #print(ii)
	    Claa1 = np.array(Claa1)

	    big_ell_array = np.append(ell_array, ell_array1)
	    big_Claa_array = np.append(Claa, Claa1)

	    l_rot, clBBrot = GimmeClBBRot(Cl_EE,big_ell_array,big_Claa_array,dl=10, n=1024, nwanted=500)        
	    '''
            mydic[bb,gg,nn]={}
	    #mydic[bb,gg,nn]['claa_FR'] = big_Claa_array
	    #mydic[bb,gg,nn]['clBB_FR'] = clBBrot
	    mydic[bb,gg,nn]['clBB_MC_vec'] = vec
	    mydic[bb,gg,nn]['clBB_MC_tens'] = tens
	    embed()
            print(bb,gg,nn)


#embed()
for bb in range(len(beta)):
    for gg in range(len(BnG))[85:len(BnG)]:
        for nn in range(len(nb_MAGCAMB)):

            config = configparser.ConfigParser()
	    config.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/params_mag_copy.ini')

	    output = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/final1/MCONLY1'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
	    config['output'] = {'output_root' : output}

	    config['cls'] = {'get_scalar_cls' :'F', 'get_vector_cls' : 'T','get_tensor_cls ': 'F', 'get_transfer': 'F'}

	    config['mag'] = {'magnetic_mode' : '1', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}

	    with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile:
    	        config.write(configfile)

	    os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
            #embed()
            ell = np.loadtxt(output+'_vecCls.dat')[:,0]
            vec = np.loadtxt(output+'_vecCls.dat')[:,3]            


            config1 = configparser.ConfigParser()
	    config1.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/params_mag_copy.ini')

	    output1 = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/final1/MCONLY2'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
	    config1['output'] = {'output_root' : output1}

	    config1['cls'] = {'get_scalar_cls' :'T', 'get_vector_cls' : 'F','get_tensor_cls ': 'T', 'get_transfer': 'F'}

	    config1['mag'] = {'magnetic_mode' : '2', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}

	    with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile1:
    	        config1.write(configfile1)

	    os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
            ell1 = np.loadtxt(output1+'_tensCls.dat')[:,0]
            tens = np.loadtxt(output1+'_tensCls.dat')[:,3] 

	    '''
	    Claa = []
	    for ii in range(len(ell_array)):
	        Claa.append(Cl_aa_spline(ii,nb_FR[nn],B_G[gg]))
	        #print(ii)
	    Claa = np.array(Claa)

	    Claa1 = []
	    for ii in range(len(ell_array1)):
	        Claa1.append(Cl_aa_spline1(ii,nb_FR[nn],B_G[gg]))
	        #print(ii)
	    Claa1 = np.array(Claa1)

	    big_ell_array = np.append(ell_array, ell_array1)
	    big_Claa_array = np.append(Claa, Claa1)

	    l_rot, clBBrot = GimmeClBBRot(Cl_EE,big_ell_array,big_Claa_array,dl=10, n=1024, nwanted=500)        
	    '''
            mydic[bb,gg,nn]={}
	    #mydic[bb,gg,nn]['claa_FR'] = big_Claa_array
	    #mydic[bb,gg,nn]['clBB_FR'] = clBBrot
	    mydic[bb,gg,nn]['clBB_MC_vec'] = vec
	    mydic[bb,gg,nn]['clBB_MC_tens'] = tens

            print(bb,gg,nn)
embed()

config = configparser.ConfigParser()
config.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/params_mag_copy.ini')

output = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/final1/MCONLY1'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
config['output'] = {'output_root' : output}

config['cls'] = {'get_scalar_cls' :'F', 'get_vector_cls' : 'T','get_tensor_cls ': 'F', 'get_transfer': 'F'}

config['mag'] = {'magnetic_mode' : '1', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}

with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile:
    config.write(configfile)

os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
            #embed()
ell = np.loadtxt(output+'_vecCls.dat')[:,0]
vec = np.loadtxt(output+'_vecCls.dat')[:,3]            


config1 = configparser.ConfigParser()
config1.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/params_mag_copy.ini')

output1 = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/final1/MCONLY2'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
config1['output'] = {'output_root' : output1}

config1['cls'] = {'get_scalar_cls' :'T', 'get_vector_cls' : 'F','get_tensor_cls ': 'T', 'get_transfer': 'F'}

config1['mag'] = {'magnetic_mode' : '2', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}

with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile1:
    config1.write(configfile1)

os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
ell1 = np.loadtxt(output1+'_tensCls.dat')[:,0]
tens = np.loadtxt(output1+'_tensCls.dat')[:,3] 

mydic[bb,gg,nn]={}
#mydic[bb,gg,nn]['claa_FR'] = big_Claa_array
#mydic[bb,gg,nn]['clBB_FR'] = clBBrot
mydic[bb,gg,nn]['clBB_MC_vec'] = vec
mydic[bb,gg,nn]['clBB_MC_tens'] = tens
print(bb,gg,nn)
