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
cosmo = Cosmo()
cmbspec = cosmo.cmb_spectra(400)
cmbspec_r = Cosmo({'r':1.}).cmb_spectra(400,spec='tensor')[:,2]

#exps = [f_sky,noise,beam]

test = [0.5,9.8,1.3]

fsky= 0.5
noice = 9.8
fwhm = 1.3

def delclaa(Nlaa,fsky,dl,ell):
    return np.sqrt(2./((2.*ell+1.)*dl*fsky))*(Nlaa)

def GimmeClBBRot(cmbspec, claa, A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=21):
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


ell, Nl = ro.GimmeNl(cmbspec,fwhm,noice,dl=10, nwanted = 21)
claa = []
clBBrot = []
nlbb_tot = []
clbblens = cmbspec[:,2][20:210:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:210:10]
clbbr = cmbspec_r[20:210:10]
summand_clbblens = clbblens[0:19]
summand_nlbb = nlbb[0:19]
summand_clbbr = clbbr[0:19]
r_eff = []

for i in range(1000):
    sigclaa=delclaa(Nl,fsky,10,ell)
    mu = 0.0
    #embed()
    #ell from 20 to 2000 with dl=10
    sigclaa = sigclaa[2:21]
    claa_i = []
    for j in range(len(sigclaa)):
        claa_i.append(np.abs(np.random.normal(mu,sigclaa[j])))
    claa_i = np.array(claa_i)
    claa.append(claa_i)
    l_aa, clBBrot_i = GimmeClBBRot(cmbspec, claa_i, A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=21)
    #taking sum of l=20 to l=200 with dl=10
    summand_clBBrot_i = clBBrot_i[0:19]
    #embed()
    nlbb_tot_i = clbblens + clBBrot_i + nlbb
    nlbb_tot.append(nlbb_tot_i)
    summand_nlbb_tot = summand_clbblens+summand_clBBrot_i + summand_nlbb
    num_i = np.sum(((summand_clBBrot_i-summand_clbblens)*summand_clbbr)/(summand_nlbb_tot**2.))*10.
    denom_i = np.sum((summand_clbbr**2.)/(summand_nlbb_tot**2.))*10.
    r_i = num_i/denom_i
    #r_i = np.logspace(-10,0,num=1000)
    r_eff.append(r_i)
    print(i)
r_eff = np.array(r_eff)
r_eff_std = np.std(r_eff)
embed()

fsky= 0.01
noice = 3.4
fwhm = 30.0

ell, Nl = ro.GimmeNl(cmbspec,fwhm,noice,dl=10, nwanted = 21)
claa = []
clBBrot = []
nlbb_tot = []
clbblens = cmbspec[:,2][20:210:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:210:10]
clbbr = cmbspec_r[20:210:10]
summand_clbblens = clbblens[0:19]
summand_nlbb = nlbb[0:19]
summand_clbbr = clbbr[0:19]
r_eff1 = []

for i in range(1000):
    sigclaa=delclaa(Nl,fsky,10,ell)
    mu = 0.0
    #embed()
    #ell from 20 to 2000 with dl=10
    sigclaa=sigclaa[2:21]
    claa_i = []
    for j in range(len(sigclaa)):
        claa_i.append(np.abs(np.random.normal(mu,sigclaa[j])))
    claa_i = np.array(claa_i)
    claa.append(claa_i)
    l_aa, clBBrot_i = GimmeClBBRot(cmbspec, claa_i, A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=21)
    #taking sum of l=20 to l=200 with dl=10
    summand_clBBrot_i = clBBrot_i[0:19]
    #embed()
    nlbb_tot_i = clbblens + clBBrot_i + nlbb
    nlbb_tot.append(nlbb_tot_i)
    summand_nlbb_tot = summand_clbblens+summand_clBBrot_i + summand_nlbb
    num_i = np.sum(((summand_clBBrot_i-summand_clbblens)*summand_clbbr)/(summand_nlbb_tot**2.))*10.
    denom_i = np.sum((summand_clbbr**2.)/(summand_nlbb_tot**2.))*10.
    r_i = num_i/denom_i
    #r_i = np.logspace(-10,0,num=1000)
    r_eff1.append(r_i)
    print(i)
r_eff1 = np.array(r_eff1)
r_eff_std1 = np.std(r_eff1)

fsky= 0.65
noice = 11.8
fwhm = 3.5

ell, Nl = ro.GimmeNl(cmbspec,fwhm,noice,dl=10, nwanted = 21)
claa = []
clBBrot = []
nlbb_tot = []
clbblens = cmbspec[:,2][20:210:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:210:10]
clbbr = cmbspec_r[20:210:10]
summand_clbblens = clbblens[0:19]
summand_nlbb = nlbb[0:19]
summand_clbbr = clbbr[0:19]
r_eff2 = []

for i in range(1000):
    sigclaa=delclaa(Nl,fsky,10,ell)
    mu = 0.0
    #embed()
    #ell from 20 to 2000 with dl=10
    sigclaa=sigclaa[2:21]
    claa_i = []
    for j in range(len(sigclaa)):
        claa_i.append(np.abs(np.random.normal(mu,sigclaa[j])))
    claa_i = np.array(claa_i)
    claa.append(claa_i)
    l_aa, clBBrot_i = GimmeClBBRot(cmbspec, claa_i, A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=21)
    #taking sum of l=20 to l=200 with dl=10
    summand_clBBrot_i = clBBrot_i[0:19]
    #embed()
    nlbb_tot_i = clbblens + clBBrot_i + nlbb
    nlbb_tot.append(nlbb_tot_i)
    summand_nlbb_tot = summand_clbblens+summand_clBBrot_i + summand_nlbb
    num_i = np.sum(((summand_clBBrot_i-summand_clbblens)*summand_clbbr)/(summand_nlbb_tot**2.))*10.
    denom_i = np.sum((summand_clbbr**2.)/(summand_nlbb_tot**2.))*10.
    r_i = num_i/denom_i
    #r_i = np.logspace(-10,0,num=1000)
    r_eff2.append(r_i)
    print(i)
r_eff2 = np.array(r_eff2)
r_eff_std2 = np.std(r_eff2)
#chi2 = ((sum_clBBrot_i - r_i * sum_clBBrot_i)/(sum_nlbb_tot))**2.
#sigclaa[2:200]

ell, Nl = ro.GimmeNl(cmbspec,fwhm,noice,dl=10, nwanted = 21)
claa = []
clBBrot = []
nlbb_tot = []
clbblens = cmbspec[:,2][20:210:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:210:10]
clbbr = cmbspec_r[20:210:10]
summand_clbblens = clbblens[0:19]
summand_nlbb = nlbb[0:19]
summand_clbbr = clbbr[0:19]
r_eff3 = []

for i in range(1000):
    sigclaa=delclaa(Nl,fsky,10,ell)
    mu = 0.0
    #embed()
    #ell from 20 to 2000 with dl=10
    sigclaa=sigclaa[2:21]
    claa_i = []
    for j in range(len(sigclaa)):
        claa_i.append(np.abs(np.random.normal(mu,sigclaa[j])))
    claa_i = np.array(claa_i)
    claa.append(claa_i)
    l_aa, clBBrot_i = GimmeClBBRot(cmbspec, claa_i, A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=21)
    #taking sum of l=20 to l=200 with dl=10
    summand_clBBrot_i = clBBrot_i[0:19]
    #embed()
    nlbb_tot_i = clbblens + clBBrot_i + nlbb
    nlbb_tot.append(nlbb_tot_i)
    summand_nlbb_tot = summand_clbblens+summand_clBBrot_i + summand_nlbb
    num_i = np.sum(((summand_clBBrot_i-summand_clbblens)*summand_clbbr)/(summand_nlbb_tot**2.))*10.
    denom_i = np.sum((summand_clbbr**2.)/(summand_nlbb_tot**2.))*10.
    r_i = num_i/denom_i
    #r_i = np.logspace(-10,0,num=1000)
    r_eff3.append(r_i)
    print(i)
r_eff3 = np.array(r_eff3)
r_eff_std3 = np.std(r_eff3)

fsky= 0.5
noice = 1.5
fwhm = 3.0

ell, Nl = ro.GimmeNl(cmbspec,fwhm,noice,dl=10, nwanted = 21)
claa = []
clBBrot = []
nlbb_tot = []
clbblens = cmbspec[:,2][20:210:10]
nlbb = ro.nl_cmb(noice,fwhm)[20:210:10]
clbbr = cmbspec_r[20:210:10]
summand_clbblens = clbblens[0:19]
summand_nlbb = nlbb[0:19]
summand_clbbr = clbbr[0:19]
r_eff4 = []

for i in range(1000):
    sigclaa=delclaa(Nl,fsky,10,ell)
    mu = 0.0
    #embed()
    #ell from 20 to 2000 with dl=10
    sigclaa=sigclaa[2:21]
    claa_i = []
    for j in range(len(sigclaa)):
        claa_i.append(np.abs(np.random.normal(mu,sigclaa[j])))
    claa_i = np.array(claa_i)
    claa.append(claa_i)
    l_aa, clBBrot_i = GimmeClBBRot(cmbspec, claa_i, A_CB=1., nu=30., B=1., H_I=1., f_a=1., dl=10, n=512, nwanted=21)
    #taking sum of l=20 to l=200 with dl=10
    summand_clBBrot_i = clBBrot_i[0:19]
    #embed()
    nlbb_tot_i = clbblens + clBBrot_i + nlbb
    nlbb_tot.append(nlbb_tot_i)
    summand_nlbb_tot = summand_clbblens+summand_clBBrot_i + summand_nlbb
    num_i = np.sum(((summand_clBBrot_i-summand_clbblens)*summand_clbbr)/(summand_nlbb_tot**2.))*10.
    denom_i = np.sum((summand_clbbr**2.)/(summand_nlbb_tot**2.))*10.
    r_i = num_i/denom_i
    #r_i = np.logspace(-10,0,num=1000)
    r_eff4.append(r_i)
    print(i)
r_eff4 = np.array(r_eff4)
r_eff_std4 = np.std(r_eff4)
embed()
'''
n, bins, patches = plt.hist(r_eff)
plt.xlabel(r'$r_{eff}$',size=14)
plt.ylabel('Frequency')
plt.show()
'''
#data = {'r_eff' : r_eff, 'r_eff1' : r_eff1, 'r_eff2' : r_eff2, 'r_eff3' : r_eff3, 'r_eff4' : r_eff4}
#with open("r_xi_abs.pkl", "wb") as infile:
#    pickle.dump(data, infile)

