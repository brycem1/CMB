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
l_rot, ClBBRot = ro.GimmeClBBRot(cmbspec,dl=1,nwanted=3000)

def del_clbb(r):
    cmbspec_r = Cosmo({'r':r}).cmb_spectra(1999,spec='tensor')[:,2]
    ell=np.arange(2000)
    denom_r = np.sqrt(2./(2.*ell+1.))*(cmbspec_r+ClBBRot[0:2000])
    frac_r = (cmbspec_r/denom_r)**2.
    return np.sum(frac_r[20:200])-6.

embed()

from scipy.optimize import brentq
brentq(del_clbb,1.0E-8,1.0)
    

