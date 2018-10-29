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
cmbspec_r = Cosmo({'r':1.}).cmb_spectra(210,spec='tensor')[:,2]
l_rot, ClBBRot = ro.GimmeClBBRot(cmbspec,dl=1,nwanted=200)

r_eff = np.sum(ClBBRot[20:200])/np.sum(cmbspec_r[20:200])
embed()

