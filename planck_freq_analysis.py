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
outdir = 'plots/analysis/Planck x unWISE Frequency Maps/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Setting up maps and Cls')
print('   loading maps')
### Load maps (bin 0)
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
Tmap_100_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_143_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_217_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
lssmap_unwise_inp = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
lssmap_gauss_inp = hp.synfast(hp.anafast(lssmap_unwise_inp), 2048)
vrmaps = np.zeros((8, hp.nside2npix(2048)))
for i in np.arange(8):
    vrmaps[i,:] = c.load(basic_conf_dir, 'vr_full_fine_2048_64_real=0_bin=0', dir_base=estim_dir+'sims')

# Need correct units
#vr_actual = np.mean(vrmaps,axis=0)
#Clvv_actual = hp.anafast(vr_actual)
lssmap_gauss  = lssmap_gauss_inp.copy()
lssmap_gauss_masked  = lssmap_gauss_inp * mask_map
lssmap_unwise = lssmap_unwise_inp * mask_map

fsky        = np.where(mask_map!=0)[0].size / mask_map.size

beam_100   = hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6144)
beam_143   = hp.gauss_beam(fwhm=np.radians(7.27/60), lmax=6144)
beam_217   = hp.gauss_beam(fwhm=np.radians(5.01/60), lmax=6144)

# Numerical trouble for healpix for beams that blow up at large ell. Sufficient to flatten out at high ell above where the 1/TT filter peaks
beam_100[4001:] = beam_100[4000]
beam_143[4001:] = beam_143[4000]
#beam_217[4001:] = beam_217[4000]

Tmap_100_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)
Tmap_143_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_143_inp*mask_map), 1/beam_143)
Tmap_217_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_217_inp*mask_map), 1/beam_217)

Tmap_100 = hp.alm2map(Tmap_100_alm_masked_debeamed, 2048)
Tmap_143 = hp.alm2map(Tmap_143_alm_masked_debeamed, 2048)
Tmap_217 = hp.alm2map(Tmap_217_alm_masked_debeamed, 2048)

### Load / compute Cls (bin 0)
# Temperature
print('   computing ClTT')
Tcl_100 = hp.alm2cl(Tmap_100_alm_masked_debeamed) / fsky
Tcl_143 = hp.alm2cl(Tmap_143_alm_masked_debeamed) / fsky
Tcl_217 = hp.alm2cl(Tmap_217_alm_masked_debeamed) / fsky
ClTT_100 = np.append(Tcl_100, Tcl_100[-1])
ClTT_143 = np.append(Tcl_143, Tcl_143[-1])
ClTT_217 = np.append(Tcl_217, Tcl_217[-1])
ClTT_100[:100] = ClTT_143[:100] = ClTT_217[:100] = 1e15

print('   loading and interpolating Clgg')
Cldd_gauss = hp.anafast(lssmap_gauss)
Cldd_gauss = np.append(Cldd_gauss, Cldd_gauss[-1])
Cltaudg_gauss = Cldd_gauss.copy()

Cldd_gauss_masked = hp.anafast(lssmap_gauss_masked) / fsky
Cldd_gauss_masked = np.append(Cldd_gauss_masked, Cldd_gauss_masked[-1])
Cltaudg_gauss_masked = Cldd_gauss_masked.copy()

Cldd_unwise = hp.anafast(lssmap_unwise) / fsky
Cldd_unwise = np.append(Cldd_unwise, Cldd_unwise[-1])
Cltaudg_unwise = Cldd_unwise.copy()

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

def combine(Tmap, lssmap, ClTT, Clgg, Noise):
    mask = mask_map.copy()
    dTlm = hp.map2alm(Tmap)
    dlm  = hp.map2alm(lssmap)
    cltaudg = Clgg.copy()
    ClTT_filter = ClTT.copy()
    Clgg_filter = Clgg.copy()
    ClTT_filter[:100] = 1e15
    Clgg_filter[:100] = 1e15
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
    dlm_zeta = hp.almxfl(dlm, np.divide(cltaudg, Clgg_filter, out=np.zeros_like(cltaudg), where=Clgg_filter!=0))
    Tmap_filtered = hp.alm2map(dTlm_xi, 2048) * mask
    lssmap_filtered = hp.alm2map(dlm_zeta, 2048) * mask
    outmap_filtered = Tmap_filtered*lssmap_filtered
    const_noise = np.median(Noise[10:100])
    outmap = outmap_filtered * const_noise * mask
    return Tmap_filtered * const_noise, lssmap_filtered, outmap

hp.mollview(Tmap_100, title='Planck 100GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_100')

hp.mollview(Tmap_143, title='Planck 143GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_143')

hp.mollview(Tmap_217, title='Planck 217GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_217')

hp.mollview(lssmap_gauss,title='Gaussian Realization of unWISE gg Spectrum')
plt.savefig(outdir + 'maps_lssmap_gauss')

hp.mollview(lssmap_unwise, title='unWISE Blue Sample')
plt.savefig(outdir + 'maps_lssmap_unwise')

plt.close('all')

def twopt(outmap, theory_noise, FSKY, plottitle, filename):
    recon_noise = hp.anafast(outmap)
    plt.figure()
    plt.loglog(np.arange(2,6144), recon_noise[2:], label='Reconstruction')
    plt.loglog(np.arange(2,6144), theory_noise[2:] * FSKY, label='Theory * fsky')
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

Noise_T100_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_gauss, Cltaudg_gauss)]*6144)
Noise_T100_ggauss_masklss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_gauss_masked, Cltaudg_gauss_masked)]*6144)
Noise_T143_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_143, Cldd_gauss, Cltaudg_gauss)]*6144)
Noise_T217_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd_gauss, Cltaudg_gauss)]*6144)

Tmap_100_filtered, lssmap_gauss_filtered,         outmap_T100_ggauss         = combine(Tmap_100, lssmap_gauss, ClTT_100, Cldd_gauss, Noise_T100_ggauss)
_                , lssmap_gauss_filtered_masklss, outmap_T100_ggauss_masklss = combine(Tmap_100, lssmap_gauss_masked, ClTT_100, Cldd_gauss_masked, Noise_T100_ggauss_masklss)
Tmap_143_filtered, _,                             outmap_T143_ggauss         = combine(Tmap_143, lssmap_gauss, ClTT_143, Cldd_gauss, Noise_T143_ggauss)
Tmap_217_filtered, _,                             outmap_T217_ggauss         = combine(Tmap_217, lssmap_gauss, ClTT_217, Cldd_gauss, Noise_T217_ggauss)

twopt(outmap_T100_ggauss, Noise_T100_ggauss, fsky, r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss')
twopt(outmap_T100_ggauss_masklss, Noise_T100_ggauss_masklss, fsky, r'Power Spectrum of T[100GHz] $\times$ masked lss[gauss]', 'twopt_T100_ggauss_masklss')
twopt(outmap_T143_ggauss, Noise_T143_ggauss, fsky, r'Power Spectrum of T[143GHz] $\times$ lss[gauss]', 'twopt_T143_ggauss')
twopt(outmap_T217_ggauss, Noise_T217_ggauss, fsky, r'Power Spectrum of T[217GHz] $\times$ lss[gauss]', 'twopt_T217_ggauss')


hp.mollview(outmap_T100_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T100_ggauss')

hp.mollview(outmap_T143_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T143_ggauss')

hp.mollview(outmap_T217_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T217_ggauss')

Noise_T100_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_unwise, Cltaudg_unwise)]*6144)
Noise_T143_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_143, Cldd_unwise, Cltaudg_unwise)]*6144)
Noise_T217_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd_unwise, Cltaudg_unwise)]*6144)

_, lssmap_unwise_filtered, outmap_T100_gunwise = combine(Tmap_100, lssmap_unwise, ClTT_100, Cldd_unwise, Noise_T100_gunwise)
_, _,                      outmap_T143_gunwise = combine(Tmap_143, lssmap_unwise, ClTT_143, Cldd_unwise, Noise_T143_gunwise)
_, _,                      outmap_T217_gunwise = combine(Tmap_217, lssmap_unwise, ClTT_217, Cldd_unwise, Noise_T217_gunwise)

twopt(outmap_T100_gunwise, Noise_T100_gunwise, fsky, r'Power Spectrum of T[100GHz] $\times$ lss[unWISE]', 'twopt_T100_gunwise')
twopt(outmap_T143_gunwise, Noise_T143_gunwise, fsky, r'Power Spectrum of T[143GHz] $\times$ lss[unWISE]', 'twopt_T143_gunwise')
twopt(outmap_T217_gunwise, Noise_T217_gunwise, fsky, r'Power Spectrum of T[217GHz] $\times$ lss[unWISE]', 'twopt_T217_gunwise')


hp.mollview(outmap_T100_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T100_gunwise')

hp.mollview(outmap_T143_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T143_gunwise')

hp.mollview(outmap_T217_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T217_gunwise')

