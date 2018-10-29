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
import pickle

data = pickle.load(open("r_xi_abs2.pkl","rb"))

cosmo = Cosmo()
cmbspec = cosmo.cmb_spectra(400)
cmbspec_r = Cosmo({'r':1.}).cmb_spectra(400,spec='tensor')[:,2]

clbblens = cmbspec[:,2][20:210:10]
clbbr = cmbspec_r[20:210:10]

r_eff  = data['r_eff']
r_eff1 = data['r_eff1']
r_eff2 = data['r_eff2']
r_eff3 = data['r_eff3']
r_eff4 = data['r_eff4']

claa=data['claa']
claa1=data['claa1']
claa2=data['claa2']
claa3=data['claa3']
claa4=data['claa4']

clBBrot=data['clBBrot']
clBBrot1=data['clBBrot1']
clBBrot2=data['clBBrot2']
clBBrot3=data['clBBrot3']
clBBrot4=data['clBBrot4']

del_clbb=data['del_clbb']
del_clbb1=data['del_clbb1']
del_clbb2=data['del_clbb2']
del_clbb3=data['del_clbb3']
del_clbb4=data['del_clbb4']

claa = np.array(claa)
claa1 = np.array(claa1)
claa2 = np.array(claa2)
claa3 = np.array(claa3)
claa4 = np.array(claa4)

clBBrot = np.array(clBBrot)
clBBrot1 = np.array(clBBrot1)
clBBrot2 = np.array(clBBrot2)
clBBrot3 = np.array(clBBrot3)
clBBrot4 = np.array(clBBrot4)

del_clbb = np.array(del_clbb)
del_clbb1 = np.array(del_clbb1)
del_clbb2 = np.array(del_clbb2)
del_clbb3 = np.array(del_clbb3)
del_clbb4 = np.array(del_clbb4)

r_eff_std  = np.std(r_eff)
r_eff_std1 = np.std(r_eff1)
r_eff_std2 = np.std(r_eff2)
r_eff_std3 = np.std(r_eff3)
r_eff_std4 = np.std(r_eff4)

r_eff_mean = np.mean(r_eff)
r_eff_mean1 = np.mean(r_eff1)
r_eff_mean2 = np.mean(r_eff2)
r_eff_mean3 = np.mean(r_eff3)
r_eff_mean4 = np.mean(r_eff4)



claa_tot = np.zeros(19)
claa_tot1 = np.zeros(19)
claa_tot2 = np.zeros(19)
claa_tot3 = np.zeros(19)
claa_tot4 = np.zeros(19)

clBBrot_tot = np.zeros(19)
clBBrot_tot1 = np.zeros(19)
clBBrot_tot2 = np.zeros(19)
clBBrot_tot3 = np.zeros(19)
clBBrot_tot4 = np.zeros(19)

del_clbb_tot = np.zeros(19)
del_clbb_tot1 = np.zeros(19)
del_clbb_tot2 = np.zeros(19)
del_clbb_tot3 = np.zeros(19)
del_clbb_tot4 = np.zeros(19)

ell = np.arange(20,210,10)

for i in range(19):
    claa_tot[i] = np.mean(claa[:,i])
    claa_tot1[i] = np.mean(claa1[:,i])
    claa_tot2[i] = np.mean(claa2[:,i])
    claa_tot3[i] = np.mean(claa3[:,i])
    claa_tot4[i] = np.mean(claa4[:,i])

for i in range(19):
    clBBrot_tot[i] = np.mean(clBBrot[:,i])
    clBBrot_tot1[i] = np.mean(clBBrot1[:,i])
    clBBrot_tot2[i] = np.mean(clBBrot2[:,i])
    clBBrot_tot3[i] = np.mean(clBBrot3[:,i])
    clBBrot_tot4[i] = np.mean(clBBrot4[:,i])

for i in range(19):
    del_clbb_tot[i] = np.mean(del_clbb[:,i])
    del_clbb_tot1[i] = np.mean(del_clbb1[:,i])
    del_clbb_tot2[i] = np.mean(del_clbb2[:,i])
    del_clbb_tot3[i] = np.mean(del_clbb3[:,i])
    del_clbb_tot4[i] = np.mean(del_clbb4[:,i])

embed()

plt.loglog(ell,claa_tot, label='AdvACT')
plt.loglog(ell,claa_tot1,label='BICEP3')
plt.loglog(ell,claa_tot2,label='SA')
plt.loglog(ell,claa_tot3,label='SPT3G')
plt.loglog(ell,claa_tot4,label='CMBS4')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{\alpha\alpha}$')
plt.show()

plt.loglog(ell,clBBrot_tot, label='AdvACT')
plt.loglog(ell,clBBrot_tot1,label='BICEP3')
plt.loglog(ell,clBBrot_tot2,label='SA')
plt.loglog(ell,clBBrot_tot3,label='SPT3G')
plt.loglog(ell,clBBrot_tot4,label='CMBS4')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB,rot}$')
plt.show()

plt.subplot(3,2,1)
plt.loglog(ell,clBBrot_tot, label='AdvACT')
plt.loglog(ell,r_eff_mean*clbbr, label='GW $r=1$')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')

plt.subplot(3,2,2)
plt.loglog(ell,clBBrot_tot1, label='BICEP3')
plt.loglog(ell,r_eff_mean1*clbbr, label='GW $r=1$')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')

plt.subplot(3,2,3)
plt.loglog(ell,clBBrot_tot2, label='SA')
plt.loglog(ell,r_eff_mean2*clbbr, label='GW $r=1$')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')

plt.subplot(3,2,4)
plt.loglog(ell,clBBrot_tot3, label='SPT3G')
plt.loglog(ell,r_eff_mean3*clbbr, label='GW $r=1$')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')

plt.subplot(3,2,5)
plt.loglog(ell,clBBrot_tot4, label='CMBS4')
plt.loglog(ell,r_eff_mean4*clbbr, label='GW $r=1$')
plt.legend(loc='best')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{BB}$')
plt.show()

plt.hist(r_eff1,bins=20,label=r'$\mu=%.6f,\sigma=%.6f$' %(r_eff_mean1, r_eff_std1))
plt.xlabel(r'$r_{\rm{eff}}$',size=14)
plt.ylabel('Frequency')
plt.title('BICEP3')
plt.legend(loc='best')
plt.show()

plt.hist(r_eff,bins=20,label=r'$\mu=%.6f,\sigma=%.6f$' %(r_eff_mean, r_eff_std))
plt.xlabel(r'$r_{eff}$',size=14)
plt.ylabel('Frequency')
plt.title('AdvACT')
plt.legend(loc='best')
plt.show()

plt.hist(r_eff2,bins=20,label=r'$\mu=%.6f,\sigma=%.6f$' %(r_eff_mean2, r_eff_std2))
plt.xlabel(r'$r_{eff}$',size=14)
plt.ylabel('Frequency')
plt.title('Simons Array')
plt.legend(loc='best')
plt.show()

plt.hist(r_eff3,bins=20,label=r'$\mu=%.6f,\sigma=%.6f$' %(r_eff_mean3, r_eff_std3))
plt.xlabel(r'$r_{eff}$',size=14)
plt.ylabel('Frequency')
plt.title('SPT-3G')
plt.legend(loc='best')
plt.show()

plt.hist(r_eff4,bins=20,label=r'$\mu=%.3e,\sigma=%.3e$' %(r_eff_mean4, r_eff_std4))
plt.xlabel(r'$r_{eff}$',size=14)
plt.ylabel('Frequency')
#plt.ylim(ymax=160)
plt.title('CMBS4')
plt.legend(loc='best')
plt.show()

plt.hist(r_eff1,bins=20,label=r'$r_{\rm{eff}}<%.3E$' %(mh_951) +r'$\;\;@ 95\%\;\; \rm{C.L.}$' )
plt.xlabel(r'$r_{\rm{eff}}$',size=14)
plt.ylabel('Frequency',size=14)
plt.title('BICEP3',size=14)
plt.legend(loc='best',fontsize=14)
plt.xlim(xmin=0)
plt.show()

plt.hist(r_eff,bins=20,label=r'$r_{\rm{eff}}<%.3E$' %(mh_95) +r'$\;\;@ 95\%\;\; \rm{C.L.}$')
plt.xlabel(r'$r_{\rm{eff}}$',size=14)
plt.ylabel('Frequency',size=14)
plt.title('AdvACT',size=14)
plt.legend(loc='best',fontsize=14)
plt.xlim(xmin=0)
plt.show()

plt.hist(r_eff2,bins=20,label=r'$r_{\rm{eff}}<%.3E$' %(mh_952) +r'$\;\;@ 95\%\;\; \rm{C.L.}$')
plt.xlabel(r'$r_{\rm{eff}}$',size=14)
plt.ylabel('Frequency')
plt.title('Simons Array')
plt.legend(loc='best',fontsize=14)
plt.xlim(xmin=0)
plt.show()

plt.hist(r_eff3,bins=20,label=r'$r_{\rm{eff}}<%.3E$' %(mh_953) +r'$\;\;@ 95\%\;\; \rm{C.L.}$')
plt.xlabel(r'$r_{\rm{eff}}$',size=14)
plt.ylabel('Frequency',size=14)
plt.title('SPT-3G',size=14)
plt.legend(loc='best',fontsize=14)
plt.xlim(xmin=0)
plt.show()

plt.hist(r_eff4,bins=20,label=r'$r_{\rm{eff}}<%.3E$' %(mh_954) +r'$\;\;@ 95\%\;\; \rm{C.L.}$')
plt.xlabel(r'$r_{\rm{eff}}$',size=14)
plt.ylabel('Frequency',size=14)
#plt.ylim(ymax=160)
plt.title('CMBS4',size=14)
plt.legend(loc='best',fontsize=14)
plt.xlim(xmin=0)
plt.show()




plt.hist(r_eff1,bins=20,label='BICEP3')
plt.hist(r_eff,bins=20, label='AdvACT')
plt.hist(r_eff2,bins=20,label='SA')
plt.hist(r_eff3,bins=20,label='SPT3G')
plt.hist(r_eff4,bins=20,label='CMBS4')
plt.xlabel(r'$r_{eff}$',size=14)
plt.ylabel('Frequency',size=14)
plt.legend(loc='best',fontsize=14)
plt.xlim(xmin=0)
plt.show()



