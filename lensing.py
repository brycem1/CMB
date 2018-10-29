import numpy as np
import numba
from utils import *
from IPython import embed

# ===========================
# ====== FILTERS ============
# ===========================

@numba.jit
def W_delens(l1, l2,  n,  dl ):

    i1 = ltoi(l1, n, dl)
    i2 = ltoi(l2, n, dl)

    bigell_x = l1[0] - l2[0]
    bigell_y = l1[1] - l2[1]
    bigell   = (bigell_x, bigell_y)
    phi_l1l2 = anglebetween(l1, l2)

    output =  dotprod(l2,bigell) * np.sin(2.0 * phi_l1l2) 

    return output

# ===========================
# ==== INTEGRAL at L ========
# ===========================

@numba.jit#(nopython=1)
def Cl_BB_res_integral(eepowerspec2d, pppowerspec2d, eetotallenspowerspec2d, pptotallenspowerspec2d, dl, n, l):
    runningtotal = 0.
    usedvals = 0

    il = ltoi(l, n, dl)

    for ipx in range(n): 
        for ipy in range(n): 
            lp = itol((ipx,ipy), n, dl)
            l_min_lp = (l[0] - lp[0], l[1] - lp[1])
            il_min_ilp = ltoi(l_min_lp, n, dl)
        
            if not(circlecheck(lp, 3000, 0)): continue
            if not(circlecheck(l_min_lp, 3000, 0)): continue
                
            if (ipx < 0 or ipx >= n or il_min_ilp[0] < 0 or il_min_ilp[0] >= n or ipy < 0 or ipy >= n or il_min_ilp[1] < 0 or il_min_ilp[1] >= n ):
                continue
            
            runningtotal += eepowerspec2d[ipx,ipy] * pppowerspec2d[il_min_ilp[0]][il_min_ilp[1]] * W_delens(l, lp,  n,  dl )**2 * (1. - (eepowerspec2d[ipx][ipy]/eetotallenspowerspec2d[ipx][ipy])*(pppowerspec2d[il_min_ilp[0]][il_min_ilp[1]]/pptotallenspowerspec2d[il_min_ilp[0]][il_min_ilp[1]]) )
            usedvals += 1
  
    runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
    
    if runningtotal == 0.:
        return 0.
    else:
        return runningtotal

@numba.jit
def Cl_BB_lens_integral(eepowerspec2d, pppowerspec2d, dl, n, l):
    runningtotal = 0.
    usedvals = 0

    il = ltoi(l, n, dl)

    for ipx in range(n): 
        for ipy in range(n): 
            lp = itol((ipx,ipy), n, dl)
            l_min_lp = (l[0] - lp[0], l[1] - lp[1])
            il_min_ilp = ltoi(l_min_lp, n, dl)
        
            if not(circlecheck(lp, 3000, 0)): continue
            if not(circlecheck(l_min_lp, 3000, 0)): continue
                
            if (ipx < 0 or ipx >= n or il_min_ilp[0] < 0 or il_min_ilp[0] >= n or ipy < 0 or ipy >= n or il_min_ilp[1] < 0 or il_min_ilp[1] >= n ):
                continue
            
            runningtotal += eepowerspec2d[ipx,ipy] * pppowerspec2d[il_min_ilp[0]][il_min_ilp[1]] * W_delens(l, lp,  n,  dl )**2
            usedvals += 1
  
    runningtotal *= dl * dl / (2*np.pi) / (2*np.pi)
    
    if runningtotal == 0.:
        return 0.
    else:
        return runningtotal

# ===========================
# ======= INTEGRALS =========
# ===========================

@numba.jit
def ClBB_res(eepowerspec2d, pppowerspec2d, eetotallenspowerspec2d, pptotallenspowerspec2d, dl, n, nwanted):
    output = np.zeros(nwanted)
  
    for iell in range(1,nwanted):
        ell = (dl * iell, 0)
        output[iell] = Cl_BB_res_integral(eepowerspec2d, pppowerspec2d, eetotallenspowerspec2d, pptotallenspowerspec2d, dl, n, ell)

    return output

#Returns the first-order lensed BB power spectrum
@numba.jit
def ClBB_lens(eepowerspec2d, pppowerspec2d, dl, n, nwanted):


    output = np.zeros(nwanted)
  
    for iell in range(1,nwanted):
        ell = (dl * iell, 0)
        output[iell] = Cl_BB_lens_integral(eepowerspec2d, pppowerspec2d, dl, n, ell)

    return output


# ===========================
# ======= HELPERS ===========
# ===========================

def GimmeClBBRes(cmbspec, beam, noise, lnlpp=None, nlpp=None, dl=8, n=512, nwanted=200):
    lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
    lgrid = np.sqrt(lxgrid**2 + lygrid**2)
    L     = np.arange(0,nwanted)*dl+2    
    ell   = np.arange(0,2001)

    nlee = nl_cmb(noise*np.sqrt(2), beam, cmbspec.shape[0]-1)
    
    clee = cmbspec[:,1].copy()
    
    eepowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
    nlee2d                 = np.interp(lgrid, np.arange(cmbspec.shape[0]), nlee) 
    eetotallenspowerspec2d = eepowerspec2d.copy() + nlee2d.copy()

    ell  = np.arange(cmbspec.shape[0])
    clkk = cmbspec[:,4].copy()
    clpp = cmbspec[:,4].copy()/(ell*(ell+1))**2*4
    if nlpp is None:
    	nlpp  = np.zeros_like(clpp)
    	lnlpp = np.arange(nlpp.size)
    else:
    	assert(lnlpp.size == nlpp.size)

    pppowerspec2d          = np.interp(lgrid, np.arange(cmbspec.shape[0]), clpp)
    nlpp2d                 = np.interp(lgrid, lnlpp, nlpp)
    #embed()
    pptotallenspowerspec2d = pppowerspec2d.copy() + nlpp2d.copy()
#     plt.loglog(clpp)
#     plt.loglog(clee)
#     plt.loglog(nlpp)
    clres = ClBB_res(eepowerspec2d, pppowerspec2d, eetotallenspowerspec2d, pptotallenspowerspec2d, dl, n, nwanted)
    
    return L, clres

def GimmeClBBLens(cmbspec, dl=8, n=512, nwanted=200):
    lxgrid, lygrid  = np.meshgrid( np.arange(-n/2.,n/2.)*dl, np.arange(-n/2.,n/2.)*dl )
    lgrid = np.sqrt(lxgrid**2 + lygrid**2)
    L     = np.arange(0,nwanted)*dl    
    
    clee = cmbspec[:,1].copy()
    
    eepowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), clee)
    
    ell  = np.arange(cmbspec.shape[0])
    clkk = cmbspec[:,4].copy()
    clpp = cmbspec[:,4].copy()/(ell*(ell+1))**2*4

    pppowerspec2d = np.interp(lgrid, np.arange(cmbspec.shape[0]), clpp)
    
    cllens = ClBB_lens(eepowerspec2d, pppowerspec2d, dl, n, nwanted)
    
    return L, cllens
