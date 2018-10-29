import numpy as np
from scipy import interpolate
import os
import matplotlib.pyplot as plt
from cosmojo.universe import Cosmo
from matplotlib.pyplot import cm  
import matplotlib.colors as mcolors
import pylab as py

import rotation as ro
import lensing as le
import numba

from IPython import embed

cls_sc_comp = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/mag1_scalCls.dat')
cls_vec_comp = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/mag1_vecCls.dat')
cls_tens_comp = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/mag1_tensCls.dat')
cls_sc_pass = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/mag2_scalCls.dat')
cls_tens_pass = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/mag2_tensCls.dat')
cls_tens_prim = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/testGR_tensCls.dat')

neg29_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg29_1_1nG_vecCls.dat')[:,3]
neg25_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg25_1_1nG_vecCls.dat')[:,3]
neg20_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg20_1_1nG_vecCls.dat')[:,3]
neg15_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg15_1_1nG_vecCls.dat')[:,3]
neg10_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg10_1_1nG_vecCls.dat')[:,3]
pos0_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos0_1_1nG_vecCls.dat')[:,3]
pos10_1_1nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos10_1_1nG_vecCls.dat')[:,3]

neg29_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg29_2_1nG_tensCls.dat')[:,3]
neg25_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg25_2_1nG_tensCls.dat')[:,3]
neg20_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg20_2_1nG_tensCls.dat')[:,3]
neg15_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg15_2_1nG_tensCls.dat')[:,3]
neg10_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg10_2_1nG_tensCls.dat')[:,3]
pos0_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos0_2_1nG_tensCls.dat')[:,3]
pos10_2_1nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos10_2_1nG_tensCls.dat')[:,3]

neg29_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg29_1_45nG_vecCls.dat')[:,3]
neg25_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg25_1_45nG_vecCls.dat')[:,3]
neg20_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg20_1_45nG_vecCls.dat')[:,3]
neg15_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg15_1_45nG_vecCls.dat')[:,3]
neg10_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg10_1_45nG_vecCls.dat')[:,3]
pos0_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos0_1_45nG_vecCls.dat')[:,3]
pos10_1_45nG_vecCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos10_1_45nG_vecCls.dat')[:,3]

neg29_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg29_2_45nG_tensCls.dat')[:,3]
neg25_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg25_2_45nG_tensCls.dat')[:,3]
neg20_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg20_2_45nG_tensCls.dat')[:,3]
neg15_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg15_2_45nG_tensCls.dat')[:,3]
neg10_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/neg10_2_45nG_tensCls.dat')[:,3]
pos0_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos0_2_45nG_tensCls.dat')[:,3]
pos10_2_45nG_tensCls = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos10_2_45nG_tensCls.dat')[:,3]

ells = np.loadtxt('/home/student.unimelb.edu.au/brycem1/MagCAMB/bryce/research17/pos10_2_45nG_tensCls.dat')[:,0]

plt.loglog(ells,neg29_1_1nG_vecCls,label = r'$n_B = -2.9$')
plt.loglog(ells,neg25_1_1nG_vecCls,label = r'$n_B = -2.5$')
plt.loglog(ells,neg20_1_1nG_vecCls,label = r'$n_B = -2.0$')
plt.loglog(ells,neg15_1_1nG_vecCls,label = r'$n_B = -1.5$')
plt.loglog(ells,neg10_1_1nG_vecCls,label = r'$n_B = -1.0$')
plt.loglog(ells,pos0_1_1nG_vecCls,label = r'$n_B = 0.0$')
plt.loglog(ells,pos10_1_1nG_vecCls,label = r'$n_B = 1.0$')

plt.loglog(ells,neg29_1_45nG_vecCls,label = r'$n_B = -2.9$')
plt.loglog(ells,neg25_1_45nG_vecCls,label = r'$n_B = -2.5$')
plt.loglog(ells,neg20_1_45nG_vecCls,label = r'$n_B = -2.0$')
plt.loglog(ells,neg15_1_45nG_vecCls,label = r'$n_B = -1.5$')
plt.loglog(ells,neg10_1_45nG_vecCls,label = r'$n_B = -1.0$')
plt.loglog(ells,pos0_1_45nG_vecCls,label = r'$n_B = 0.0$')
plt.loglog(ells,pos10_1_45nG_vecCls,label = r'$n_B = 1.0$')

plt.loglog(ells,neg29_2_1nG_tensCls,label = r'$n_B = -2.9$')
plt.loglog(ells,neg25_2_1nG_tensCls,label = r'$n_B = -2.5$')
plt.loglog(ells,neg20_2_1nG_tensCls,label = r'$n_B = -2.0$')
plt.loglog(ells,neg15_2_1nG_tensCls,label = r'$n_B = -1.5$')
plt.loglog(ells,neg10_2_1nG_tensCls,label = r'$n_B = -1.0$')
plt.loglog(ells,pos0_2_1nG_tensCls,label = r'$n_B = 0.0$')
plt.loglog(ells,pos10_2_1nG_tensCls,label = r'$n_B = 1.0$')
'''
plt.loglog(ells,neg29_2_45nG_tensCls,label = r'$n_B = -2.9$')
plt.loglog(ells,neg25_2_45nG_tensCls,label = r'$n_B = -2.5$')
plt.loglog(ells,neg20_2_45nG_tensCls,label = r'$n_B = -2.0$')
plt.loglog(ells,neg15_2_45nG_tensCls,label = r'$n_B = -1.5$')
plt.loglog(ells,neg10_2_45nG_tensCls,label = r'$n_B = -1.0$')
plt.loglog(ells,pos0_2_45nG_tensCls,label = r'$n_B = 0.0$')
plt.loglog(ells,pos10_2_45nG_tensCls,label = r'$n_B = 1.0$')
'''
embed()

D_r = cmbspec_r*np.arange(cmbspec_r.shape[0])*(np.arange(cmbspec_r.shape[0])+1.)/(2.*np.pi)
plt.loglog(D_r,label=r'GW: $r=0.1$')
plt.loglog(BB_lensedCL,label = 'Lensing')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell (\ell+1) C_{\ell}^{BB} / 2 \pi \quad [\mu {\rm K}^2]$')
plt.xlim(2,2500)
plt.title(r'Tensor passive modes at $B_{1Mpc}=1.0{nG}$')
plt.ylim(1.0E-7,1.0E4)
plt.legend(loc='upper left')
plt.show()


plt.loglog(ells, clBB_tens_prim*np.arange(cmbspec_r.shape[0])**2, label=r'Tens. Prim. $r=0.1$')
plt.loglog(cls_vec_comp[:,0], clBB_vec_comp, label='Vec. Comp.')
plt.loglog(cls_tens_comp[:,0], clBB_tens_comp, label='Tens. Comp.')
plt.loglog(cls_tens_pass[:,0], clBB_tens_pass,label='Tens. Pass.')
plt.loglog(cls_tens_prim[:,0], clBB_tens_primary, label='Tens. Prim.')
plt.xlim(2,2500)
plt.legend(loc='lower left')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell (\ell+1) C_{\ell}^{BB} / 2 \pi \quad [\mu {\rm K}^2]$')
plt.show()

#cosmo = Cosmo()
#cmbspec = cosmo.cmb_spectra(2500)
#cmbspec_r = Cosmo({'r':0.1}).cmb_spectra(2500,spec='tensor')

#plt.loglog(cmbspec[:,2]*np.arange(cmbspec.shape[0])**2, 'k', label='Lensing')
'''
clBB_tens_prim = cmbspec_r[:,2]
clBB_scal_prim = cmbspec[:,2]

clBB_vec_comp = cls_vec_comp[:,3]
clBB_tens_comp = cls_tens_comp[:,3]
clBB_tens_pass = cls_tens_pass[:,3]
clBB_tens_primary = cls_tens_prim[:,3]
ells = np.arange(0,2501)
'''
'''
#plt.loglog(cmbspec[:,2]*np.arange(cmbspec.shape[0])**2, 'k', label='Lensing')
plt.loglog(ells, clBB_tens_prim*np.arange(cmbspec_r.shape[0])**2, label=r'Tens. Prim. $r=0.1$')
plt.loglog(cls_vec_comp[:,0], clBB_vec_comp, label='Vec. Comp.')
plt.loglog(cls_tens_comp[:,0], clBB_tens_comp, label='Tens. Comp.')
plt.loglog(cls_tens_pass[:,0], clBB_tens_pass,label='Tens. Pass.')
plt.loglog(cls_tens_prim[:,0], clBB_tens_primary, label='Tens. Prim.')
plt.xlim(2,2500)
plt.legend(loc='lower left')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell (\ell+1) C_{\ell}^{BB} / 2 \pi \quad [\mu {\rm K}^2]$')
plt.show()
'''
config = configparser.ConfigParser()
config.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/param_mag_test.ini')

output = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/final/MCONLY1'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
config['output'] = {'output_root' : output}

config['cls'] = {'get_scalar_cls' :'F', 'get_vector_cls' : 'T','get_tensor_cls ': 'F', 'get_transfer': 'F'}

config['mag'] = {'magnetic_mode' : '1', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}

with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile:
    config.write(configfile)
'''
os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
ell = np.loadtxt(output+'_vecCls.dat')[:,0]
vec = np.loadtxt(output+'_vecCls.dat')[:,3]            


            config1 = configparser.ConfigParser()
	    config1.read('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/param_mag_test.ini')

	    output1 = '/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/final/MCONLY2'+'_'+str(bb)+'_'+str(gg)+'_'+str(nn)
	    config1['output'] = {'output_root' : output1}

	    config1['cls'] = {'get_scalar_cls' :'T', 'get_vector_cls' : 'F','get_tensor_cls ': 'T', 'get_transfer': 'F'}

	    config1['mag'] = {'magnetic_mode' : '2', 'magnetic_amp' : str(BnG[gg]), 'magnetic_ind' : str(nb_MAGCAMB[nn]), 'magnetic_lrat': str(beta[bb])}

	    with open('/home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini', 'w') as configfile1:
    	        config1.write(configfile1)

	    os.system(r'./camb /home/student.unimelb.edu.au/brycem1/cmb/Cosmic_Rotation/MagCAMB/python/test.ini')
            ell1 = np.loadtxt(output1+'_tensCls.dat')[:,0]
            tens = np.loadtxt(output1+'_tensCls.dat')[:,3] 
'''
