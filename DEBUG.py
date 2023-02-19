import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})

import numpy as np
import healpy as hp
import os
import config as conf
import common as c
import loginterp
from scipy.interpolate import interp1d 
from astropy.io import fits
from scipy.integrate import simps
from scipy.special import kn
from scipy.special import factorial

bin_centres = lambda bins : (bins[1:]+bins[:-1]) / 2
basic_conf_dir = c.get_basic_conf(conf)
estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(conf, exclude = False))+'/T_freq='+str(143)+'/'  
bin_width = 525.7008226572364
outdir = 'plots/analysis/planck_unwise_analysis_QUESTIONABLE/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Setting up maps and Cls')
print('   loading maps')
### Load maps (bin 0)
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
fsky = np.where(mask_map!=0)[0].size/mask_map.size
Tmap_100_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
lssmap_unwise = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
lssmap_gauss = hp.synfast(hp.anafast(lssmap_unwise), 2048)

beam_100   = hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6144)

beam_100[4001:] = beam_100[4000]

Tmap_100_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)

Tmap_100 = hp.alm2map(Tmap_100_alm_masked_debeamed, 2048)

### Load / compute Cls (bin 0)
# Temperature
print('   computing ClTT')
Tcl_100 = hp.alm2cl(Tmap_100_alm_masked_debeamed) / fsky
ClTT_100 = np.append(Tcl_100, Tcl_100[-1])
ClTT_100[:100] = 1e15

print('   loading and interpolating Clgg')
Cldd_gauss = hp.anafast(lssmap_gauss)
Cldd_gauss = np.append(Cldd_gauss, Cldd_gauss[-1])
Cltaudg_gauss = Cldd_gauss.copy()

print('Reconstructing')
from math import lgamma
def wigner_symbol(ell, ell_1,ell_2):
    if not ((np.abs(ell_1-ell_2) <= ell) and (ell <= ell_1+ell_2)):  
        return 0 
    J = ell +ell_1 +ell_2
    if J % 2 != 0:
        return 0
    else:
        g = int(J/2)*1.0
        w = (-1)**(g)*np.exp((lgamma(2.0*g-2.0*ell+1.0)+lgamma(2.0*g-2.0*ell_1+1.0)+lgamma(2.0*g-2.0*ell_2+1.0)\
                              -lgamma(2.0*g+1.0+1.0))/2.0 +lgamma(g+1.0)-lgamma(g-ell+1.0)-lgamma(g-ell_1+1.0)-lgamma(g-ell_2+1.0))
        return w

def Noise_vr_diag(lmax, alpha, gamma, ell, cltt, clgg_binned, cltaudg_binned):
    terms = 0
    for l2 in np.arange(lmax):
        for l1 in np.arange(np.abs(l2-ell),l2+ell+1):
            if l1 > lmax-1 or l1 <2:   #triangle rule
                continue
            gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell+1)/(4*np.pi))*wigner_symbol(ell, l1, l2)*cltaudg_binned[l2]
            term_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg_binned[l2]))
            if np.isfinite(term_entry):
                terms += term_entry
    return (2*ell+1) / terms

print('   filtering and plotting maps')
def combine(Tmap, lssmap, ClTT, Clgg, Noise):
    ClTT[:100] = 1e15
    Clgg[:100] = 1e15
    mask = mask_map.copy()
    dTlm = hp.map2alm(Tmap)
    dlm  = hp.map2alm(lssmap)
    cltaudg = Cltaudg_gauss.copy()
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
    dlm_zeta = hp.almxfl(dlm, np.divide(cltaudg, Clgg, out=np.zeros_like(cltaudg), where=Clgg!=0))
    Tmap_filtered = hp.alm2map(dTlm_xi, 2048) * mask
    lssmap_filtered = hp.alm2map(dlm_zeta, 2048) * mask
    outmap_filtered = Tmap_filtered*lssmap_filtered
    const_noise = np.median(Noise[10:100])
    outmap = outmap_filtered * const_noise * mask
    return Tmap_filtered * const_noise, lssmap_filtered, outmap

def twopt(outmap, theory_noise, plottitle, filename):
    fsky = np.where(mask_map!=0)[0].size / mask_map.size
    recon_noise = hp.anafast(outmap)
    plt.figure()
    plt.loglog(np.arange(2,6144), recon_noise[2:], label='Reconstruction')
    plt.loglog(np.arange(2,6144), theory_noise[2:] * fsky, label='Theory * fsky')
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

Noise_T100_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_gauss, Cltaudg_gauss)]*6144)

Tmap_100_filtered, lssmap_gauss_filtered, outmap_T100_ggauss = combine(Tmap_100, lssmap_gauss, ClTT_100, Cldd_gauss, Noise_T100_ggauss)

twopt(outmap_T100_ggauss, Noise_T100_ggauss, r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss')
'''





import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})

import numpy as np
import healpy as hp
import os
import config as conf
import common as c
import loginterp
from scipy.interpolate import interp1d 
from astropy.io import fits
from scipy.integrate import simps
from scipy.special import kn
from scipy.special import factorial

bin_centres = lambda bins : (bins[1:]+bins[:-1]) / 2
basic_conf_dir = c.get_basic_conf(conf)
estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(conf, exclude = False))+'/T_freq='+str(143)+'/'  
bin_width = 525.7008226572364
outdir = 'plots/analysis/planck_unwise_analysis_GOOD/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Setting up maps and Cls')
print('   loading maps')
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
fsky = np.where(mask_map!=0)[0].size/mask_map.size
Tmap_100_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
lssmap_unwise = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()


lssmap_gauss = hp.synfast(hp.anafast(lssmap_unwise), 2048)


beam_100   = hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6144)

beam_100[4001:] = beam_100[4000]

Tmap_100 = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100), 2048)

print('   computing ClTT')
Tcl_100 = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)) / fsky
ClTT_100 = np.append(Tcl_100, Tcl_100[-1])
ClTT_100[:100] = 1e15

print('   loading and interpolating Clgg')
Cldd = hp.anafast(lssmap_unwise)
Cldd = np.append(Cldd, Cldd[-1])
Cltaudd = Cldd / bin_width
Cldd[:100] = 1e15

print('Reconstructing')
from math import lgamma
def wigner_symbol(ell, ell_1,ell_2):
    if not ((np.abs(ell_1-ell_2) <= ell) and (ell <= ell_1+ell_2)):  
        return 0 
    J = ell +ell_1 +ell_2
    if J % 2 != 0:
        return 0
    else:
        g = int(J/2)*1.0
        w = (-1)**(g)*np.exp((lgamma(2.0*g-2.0*ell+1.0)+lgamma(2.0*g-2.0*ell_1+1.0)+lgamma(2.0*g-2.0*ell_2+1.0)\
                              -lgamma(2.0*g+1.0+1.0))/2.0 +lgamma(g+1.0)-lgamma(g-ell+1.0)-lgamma(g-ell_1+1.0)-lgamma(g-ell_2+1.0))
        return w

def Noise_vr_diag(lmax, alpha, gamma, ell, cltt, clgg_binned):
    cltaudg_binned = Cltaudd*bin_width
    terms = 0
    for l2 in np.arange(lmax):
        for l1 in np.arange(np.abs(l2-ell),l2+ell+1):
            if l1 > lmax-1 or l1 <2:   #triangle rule
                continue
            gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell+1)/(4*np.pi))*wigner_symbol(ell, l1, l2)*cltaudg_binned[l2]
            term_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg_binned[l2]))
            if np.isfinite(term_entry):
                terms += term_entry
    return (2*ell+1) / terms

print('   filtering and plotting maps')
def combine(Tmap, lssmap, ClTT, Noise):
    mask = mask_map.copy()
    dTlm = hp.map2alm(Tmap)
    dlm  = hp.map2alm(lssmap)
    cltaudg = Cltaudd.copy() * bin_width
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
    dlm_zeta = hp.almxfl(dlm, np.divide(cltaudg, Cldd, out=np.zeros_like(cltaudg), where=Cldd!=0))
    Tmap_filtered = hp.alm2map(dTlm_xi, 2048) * mask
    lssmap_filtered = hp.alm2map(dlm_zeta, 2048) * mask
    outmap_filtered = Tmap_filtered*lssmap_filtered
    const_noise = np.median(Noise[10:100])
    outmap = outmap_filtered * const_noise * mask
    return Tmap_filtered * const_noise, lssmap_filtered, outmap

def twopt(outmap, theory_noise, plottitle, filename):
    fsky = np.where(mask_map!=0)[0].size / mask_map.size
    recon_noise = hp.anafast(outmap)
    plt.figure()
    plt.loglog(np.arange(2,6144), recon_noise[2:], label='Reconstruction')
    plt.loglog(np.arange(2,6144), theory_noise[2:] * fsky, label='Theory * fsky')
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

Noise_T100_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd)]*6144)

Tmap_100_filtered,       lssmap_gauss_filtered,  outmap_T100_ggauss        = combine(Tmap_100, lssmap_gauss, ClTT_100, Noise_T100_gunwise)

twopt(outmap_T100_ggauss, Noise_T100_gunwise, r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss')

'''