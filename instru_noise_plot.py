import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
import matplotlib.colors as mcolors

from rotation import *
from lensing import *

arcmin2rad = np.pi / 180. / 60. 
rad2arcmin = 1./arcmin2rad

cosmo = Cosmo()

cmbspec = cosmo.cmb_spectra(5000)
cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(3000, spec='tensor')

nl0 = nl_cmb(9.8,1.3)
nl1 = nl_cmb(3.4,30)
nl2 = nl_cmb(11.8,3.5)
nl3 = nl_cmb(4.5,1.1)

l = np.arange(nl0.shape[0])

nl_tot =((1./(nl0))+(1./(nl2))+(1./(nl3)))**(-1)

plt.loglog(l,nl0,label='ACTPol')
plt.loglog(l,nl1,label='BICEP3')
plt.loglog(l,nl2,label='SA')
plt.loglog(l,nl3,label='SPT-3G')
plt.loglog(l,nl_tot,label='ACT+SA+SPT3G')
plt.legend(title = 'Instrumental Noise', loc='upper left')
plt.xlabel(r'$\ell$',size=14)
plt.ylabel(r'$N_\ell^{BB}$',size=14)
#plt.savefig("/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/Figures/Nl.pdf")
plt.show()
