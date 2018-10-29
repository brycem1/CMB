import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
import matplotlib.colors as mcolors

import rotation as ro
import lensing as le

from IPython import embed

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad

cosmo = Cosmo()
cmbspec = cosmo.cmb_spectra(5000)
smspec_r = cosmo.cmb_spectra(3000, 'tensor')
#sigma(r=0)
def sig_r0(fsky,BBpowerspectrum, NlBB,lmin=20,lmax=200):
    ell = np.arange(BBpowerspectrum.shape[0])
    
    summand = ((2.*ell[lmin:lmax+1]+1)*fsky/2.)*(BBpowerspectrum[lmin:lmax+1]/NlBB[lmin:lmax+1])**2.
    return 1.0/(np.sqrt(np.sum(summand)))

def del_Cl_r(del_r,r0=0.1, lmax=3000):
    cmbspec_ru = Cosmo({'r':r0+del_r}).cmb_spectra(lmax, spec='tensor')
    cmbspec_rl = Cosmo({'r':r0-del_r}).cmb_spectra(lmax, spec='tensor')
    del_Clr = cmbspec_ru[:,2]-cmbspec_rl[:,2]
    return del_Clr

def dClr_dr(del_r,r0=0.1,l_max = 3000):
    quotient = del_Cl_r(del_r)/(2.*del_r)
    return quotient

#sigma(r>0)
def sig_r(fsky,del_r,Cl_BBr,Cl_BBlens,Cl_BBrot, noise_BB,r0=0.1,lmin=2,lmax=200):
    ell = np.arange(Cl_BBr.shape[0])
    Cl_BB_full = Cl_BBr[lmin:lmax+1]+Cl_BBlens[lmin:lmax+1]+Cl_BBrot[lmin:lmax+1]+noise_BB[lmin:lmax+1]
    sig_Cl_BB_full = np.sqrt(2./((2.*ell[lmin:lmax+1]+1)*fsky))*Cl_BB_full
    dCl_dr = dClr_dr(del_r)
    summand = (dCl_dr[lmin:lmax+1]/sig_Cl_BB_full)**2
    return 1.0/(np.sqrt(np.sum(summand)))


#base case 
ClBBr = Cosmo({'r':0.1}).cmb_spectra(3000, spec='tensor')[:,2]
ClBBlens = cosmo.cmb_spectra(5000)[:,2]

#rotation stuff

#def GimmeClBBRot(cmbspec, kind='default', A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=8, n=512, nwanted=200):
l_rot, ClBBRot = ro.GimmeClBBRot(cmbspec,dl=1,nwanted=3000)

#def nl_cmb(noise_uK_arcmin, fwhm_arcmin, lmax=3000, lknee=None, alpha=None):
nl0 = nl_cmb(9.8,1.3)
nl1 = nl_cmb(3.4,30)
nl2 = nl_cmb(11.8,3.5)
nl3 = nl_cmb(4.5,1.1)
nl_tot =((1./(nl0))+(1./(nl2))+(1./(nl3)))**(-1)

#def GimmeClBBRot(cmbspec, kind='default', A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=8, n=512, nwanted=200):
l_rot, ClBBrot = ro.GimmeClBBRot(cmbspec)

#def GimmeClBBRes(cmbspec, beam, noise, lnlpp=None, nlpp=None, dl=8, n=512, nwanted=200):
l_a0, ClBBares0 = ro.GimmeClBBRes(cmbspec,1.3,9.8)
l_a1, ClBBares1 = ro.GimmeClBBRes(cmbspec,30.0,3.4)
l_a2, ClBBares2 = ro.GimmeClBBRes(cmbspec,3.5,11.8)
l_a3, ClBBares3 = ro.GimmeClBBRes(cmbspec,1.1,4.5)

#GimmeNl(cmbspec, beam, noise, est='EB', dl=8, n=512, nwanted=200, f_delens=0., lmax_delens=5000, lknee=None, alpha=None):
ell,Nl_aa0 = ro.GimmeNl(cmbspec,1.3,9.8)
ell,Nl_aa1 = ro.GimmeNl(cmbspec,30,3.4)
ell,Nl_aa2 = ro.GimmeNl(cmbspec,3.5,11.8)
ell,Nl_aa3 = ro.GimmeNl(cmbspec,1.1,4.5)

nlaa_tot =np.nan_to_num(((1./(Nl_aa0[1:]))+(1./(Nl_aa2[1:]))+(1./(Nl_aa3[1:])))**(-1))

#lensing stuff

#loading Nl^PP from lensing stuff
N_lens = np.loadtxt('mynoise.dat')
l = N_lens[:,0];Nlens = N_lens[:,1]
N_lens1 = np.loadtxt('mynoise1.dat')
l1 = N_lens1[:,0];Nlens1 = N_lens1[:,1]
N_lens2 = np.loadtxt('mynoise2.dat')
l2 = N_lens2[:,0];Nlens2 = N_lens2[:,1]
N_lens3 = np.loadtxt('mynoise3.dat')
l3 = N_lens3[:,0];Nlens3 = N_lens3[:,1]


import matplotlib
tphi = lambda l: (l + 0.5)**4 / (2. * np.pi) # scaling to apply to cl_phiphi when plotting.
colors = lambda i: matplotlib.cm.jet(i * 60)
nl_tot =((1./(Nlens))+(1./(Nlens2))+(1./(Nlens3)))**(-1)
Nlens_adj = Nlens*tphi
Nlens1_adj = Nlens1*tphi
Nlens2_adj = Nlens2*tphi
Nlens3_adj = Nlens3*tphi

embed()



