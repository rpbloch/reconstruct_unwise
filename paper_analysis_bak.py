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
import camb
from camb import model


sigma_T = 6.6524587e-29  # m^2
cambpars = camb.CAMBparams()
cambpars.set_cosmology(H0 = conf.H0, ombh2=conf.ombh2, \
                                    omch2=conf.omch2, mnu=conf.mnu , \
                                    omk=conf.Omega_K, tau=conf.tau,  \
                                    TCMB =2.725 )
cambpars.InitPower.set_params(As =conf.As*1e-9 ,ns=conf.ns, r=0)
cambpars.NonLinear = model.NonLinear_both
cambpars.max_eta_k = 14000.0*conf.ks_hm[-1]
cosmology_data = camb.get_background(cambpars)
zmax = conf.z_max
bin_centres = lambda bins : (bins[1:]+bins[:-1]) / 2
basic_conf_dir = c.get_basic_conf(conf)
estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(conf, exclude = False))+'/T_freq='+str(143)+'/'  
outdir = 'plots/analysis/Planck_unWISE/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def a(z):
    return 1/(1+z)

def ne0z(z):  # units 1/m^3
    H100_SI = 3.241e-18  # 1/s
    G_SI = 6.674e-11
    mProton_SI = 1.673e-27  # kg
    chi = 0.86
    me = 1.14
    gasfrac = 0.9
    omgh2 = gasfrac* 0.049*0.68**2
    ne0_SI = chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me                   
    return ne0_SI

mperMpc = 3.086e22  # metres per megaparsec
bin_width = cosmology_data.comoving_radial_distance(zmax) * mperMpc
K = -sigma_T * bin_width * a(zmax) * ne0z(zmax)  # conversion from delta to tau dot, unitless conversion

print('Setting up maps and Cls')
print('   loading maps')
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
Tmap_SMICA_inp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True)
Tmap_100_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_143_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_217_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_SMICAsub_100_inp = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten()
lssmap_unwise_inp = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
lssmap_gauss_inp = hp.synfast(hp.anafast(lssmap_unwise_inp), 2048)

Clvv = loginterp.log_interpolate_matrix(c.load(basic_conf_dir,'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(basic_conf_dir,'L_sample_lmax=6144', dir_base = 'Cls'))[:,0,0]

fsky        = np.where(mask_map!=0)[0].size / mask_map.size

print('   masking and debeaming T maps and computing spectra')
beam_SMICA = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6144)
beam_100   = hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6144)
beam_143   = hp.gauss_beam(fwhm=np.radians(7.27/60), lmax=6144)
beam_217   = hp.gauss_beam(fwhm=np.radians(5.01/60), lmax=6144)

# Numerical trouble for healpix for beams that blow up at large ell. Sufficient to flatten out at high ell above where the 1/TT filter peaks
beam_100[4001:] = beam_100[4000]

Tcl_SMICA_maxdata = hp.anafast(Tmap_SMICA_inp, lmax=2500)[-1]
faux_noisemap = hp.synfast([Tcl_SMICA_maxdata for i in np.arange(6144)], 2048)
Tmap_SMICA_noised = Tmap_SMICA_inp + faux_noisemap
Tmap_SMICA_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_SMICA_noised*mask_map), 1/beam_SMICA)
Tmap_100_alm_masked_debeamed   = hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)
Tmap_143_alm_masked_debeamed   = hp.almxfl(hp.map2alm(Tmap_143_inp*mask_map), 1/beam_143)
Tmap_217_alm_masked_debeamed   = hp.almxfl(hp.map2alm(Tmap_217_inp*mask_map), 1/beam_217)
Tmap_SMICAsub_100_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_SMICAsub_100_inp*mask_map), 1/beam_100)

Tmap_SMICA = hp.alm2map(Tmap_SMICA_alm_masked_debeamed, 2048)
Tmap_100   = hp.alm2map(Tmap_100_alm_masked_debeamed, 2048)
Tmap_143   = hp.alm2map(Tmap_143_alm_masked_debeamed, 2048)
Tmap_217   = hp.alm2map(Tmap_217_alm_masked_debeamed, 2048)
Tmap_SMICAsub_100 = hp.alm2map(Tmap_SMICAsub_100_alm_masked_debeamed, 2048)

### Load / compute Cls (bin 0)
# Temperature
Tcl_SMICA = hp.alm2cl(Tmap_SMICA_alm_masked_debeamed) / fsky
Tcl_100 = hp.alm2cl(Tmap_100_alm_masked_debeamed) / fsky
Tcl_143 = hp.alm2cl(Tmap_143_alm_masked_debeamed) / fsky
Tcl_217 = hp.alm2cl(Tmap_217_alm_masked_debeamed) / fsky
Tcl_SMICAsub_100 = hp.alm2cl(Tmap_SMICAsub_100_alm_masked_debeamed) / fsky
ClTT_SMICA = np.append(Tcl_SMICA, Tcl_SMICA[-1])
ClTT_100 = np.append(Tcl_100, Tcl_100[-1])
ClTT_143 = np.append(Tcl_143, Tcl_143[-1])
ClTT_217 = np.append(Tcl_217, Tcl_217[-1])
ClTT_SMICAsub_100 = np.append(Tcl_SMICAsub_100, Tcl_SMICAsub_100[-1])
ClTT_SMICA[:100] = ClTT_100[:100] = ClTT_143[:100] = ClTT_217[:100] = ClTT_SMICAsub_100[:100] =  1e15

print('   masking lss maps and computing spectra')
lssmap_gauss  = lssmap_gauss_inp.copy()
lssmap_gauss_masked  = lssmap_gauss_inp * mask_map
lssmap_unwise = lssmap_unwise_inp * mask_map

Cldd_gauss = hp.anafast(lssmap_gauss)
Cldd_gauss = np.append(Cldd_gauss, Cldd_gauss[-1])

Cldd_gauss_masked = hp.anafast(lssmap_gauss_masked) / fsky
Cldd_gauss_masked = np.append(Cldd_gauss_masked, Cldd_gauss_masked[-1])

Cldd_unwise = hp.anafast(lssmap_unwise) / fsky
Cldd_unwise = np.append(Cldd_unwise, Cldd_unwise[-1])

Cldd_unwise_unmasked = hp.anafast(lssmap_unwise_inp)
Cldd_unwise_unmasked = np.append(Cldd_unwise_unmasked, Cldd_unwise_unmasked[-1])

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
    terms = 0
    cltaudg_binned = K * clgg_binned
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
    cltaudg = K * Clgg.copy()
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

def twopt(outmap, theory_noise, FSKY, plottitle, filename, lmaxplot=700):
    #output map has units of K and noise has units of K^2
    recon_noise = hp.anafast(outmap / 2.725)
    plt.figure()
    plt.loglog(np.arange(2,lmaxplot), recon_noise[2:lmaxplot], label='Reconstruction')
    plt.loglog(np.arange(2,lmaxplot), theory_noise[2:lmaxplot] * FSKY / 2.725**2, label='Theory * fsky')
    plt.loglog(np.arange(2,10), Clvv[2:10], label='Theory Signal') 
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

print('Reconstructing')
print('   reconstructing T[freq] x lss[gauss]')
Noise_T100_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_gauss)]*6144)
Noise_T100_ggauss_masklss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_gauss_masked)]*6144)
Noise_T143_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_143, Cldd_gauss)]*6144)
Noise_T217_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd_gauss)]*6144)
Tmap_100_filtered, lssmap_gauss_filtered,         outmap_T100_ggauss         = combine(Tmap_100, lssmap_gauss, ClTT_100, Cldd_gauss, Noise_T100_ggauss)
_                , lssmap_gauss_filtered_masklss, outmap_T100_ggauss_masklss = combine(Tmap_100, lssmap_gauss_masked, ClTT_100, Cldd_gauss_masked, Noise_T100_ggauss_masklss)
Tmap_143_filtered, _,                             outmap_T143_ggauss         = combine(Tmap_143, lssmap_gauss, ClTT_143, Cldd_gauss, Noise_T143_ggauss)
Tmap_217_filtered, _,                             outmap_T217_ggauss         = combine(Tmap_217, lssmap_gauss, ClTT_217, Cldd_gauss, Noise_T217_ggauss)
print('      plotting...')
twopt(outmap_T100_ggauss, Noise_T100_ggauss, fsky, r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss')
twopt(outmap_T100_ggauss_masklss, Noise_T100_ggauss_masklss, fsky, r'Power Spectrum of T[100GHz] $\times$ masked lss[gauss]', 'twopt_T100_ggauss_masklss')
twopt(outmap_T143_ggauss, Noise_T143_ggauss, fsky, r'Power Spectrum of T[143GHz] $\times$ lss[gauss]', 'twopt_T143_ggauss')
twopt(outmap_T217_ggauss, Noise_T217_ggauss, fsky, r'Power Spectrum of T[217GHz] $\times$ lss[gauss]', 'twopt_T217_ggauss')

print('   reconstructing T[freq] x lss[unWISE]')
Noise_T100_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_unwise)]*6144)
Noise_T143_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_143, Cldd_unwise)]*6144)
Noise_T217_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd_unwise)]*6144)
_, lssmap_unwise_filtered, outmap_T100_gunwise = combine(Tmap_100, lssmap_unwise, ClTT_100, Cldd_unwise, Noise_T100_gunwise)
_, _,                      outmap_T143_gunwise = combine(Tmap_143, lssmap_unwise, ClTT_143, Cldd_unwise, Noise_T143_gunwise)
_, _,                      outmap_T217_gunwise = combine(Tmap_217, lssmap_unwise, ClTT_217, Cldd_unwise, Noise_T217_gunwise)
print('      plotting...')
twopt(outmap_T100_gunwise, Noise_T100_gunwise, fsky, r'Power Spectrum of T[100GHz] $\times$ lss[unWISE]', 'twopt_T100_gunwise')
twopt(outmap_T143_gunwise, Noise_T143_gunwise, fsky, r'Power Spectrum of T[143GHz] $\times$ lss[unWISE]', 'twopt_T143_gunwise')
twopt(outmap_T217_gunwise, Noise_T217_gunwise, fsky, r'Power Spectrum of T[217GHz] $\times$ lss[unWISE]', 'twopt_T217_gunwise')

print('   reconstructing T[freq] x unmasked lss[unWISE]')
Noise_T100_gunwise_unmaskedlss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd_unwise_unmasked)]*6144)
Noise_T143_gunwise_unmaskedlss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_143, Cldd_unwise_unmasked)]*6144)
Noise_T217_gunwise_unmaskedlss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd_unwise_unmasked)]*6144)
_, lssmap_unwise_filtered, outmap_T100_gunwise_unmaskedlss = combine(Tmap_100, lssmap_unwise_inp, ClTT_100, Cldd_unwise_unmasked, Noise_T100_gunwise_unmaskedlss)
_, _,                      outmap_T143_gunwise_unmaskedlss = combine(Tmap_143, lssmap_unwise_inp, ClTT_143, Cldd_unwise_unmasked, Noise_T143_gunwise_unmaskedlss)
_, _,                      outmap_T217_gunwise_unmaskedlss = combine(Tmap_217, lssmap_unwise_inp, ClTT_217, Cldd_unwise_unmasked, Noise_T217_gunwise_unmaskedlss)
print('      plotting...')
twopt(outmap_T100_gunwise_unmaskedlss, Noise_T100_gunwise_unmaskedlss, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[100GHz] $\times$ unmasked lss[unWISE]', 'twopt_T100_gunwise_unmaskedlss')
twopt(outmap_T143_gunwise_unmaskedlss, Noise_T143_gunwise_unmaskedlss, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[143GHz] $\times$ unmasked lss[unWISE]', 'twopt_T143_gunwise_unmaskedlss')
twopt(outmap_T217_gunwise_unmaskedlss, Noise_T217_gunwise_unmaskedlss, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[217GHz] $\times$ unmasked lss[unWISE]', 'twopt_T217_gunwise_unmaskedlss')

print('   reconstructing T[SMICA] x lss[gauss]')
Noise_TSMICA_ggauss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_SMICA, Cldd_gauss)]*6144)
Tmap_SMICA_filtered, _, outmap_TSMICA_ggauss = combine(Tmap_SMICA, lssmap_gauss, ClTT_SMICA, Cldd_gauss, Noise_TSMICA_ggauss)
print('      plotting...')
twopt(outmap_TSMICA_ggauss, Noise_TSMICA_ggauss, fsky, r'Power Spectrum of T[SMICA] $\times$ lss[gauss]', 'twopt_TSMICA_ggauss')

print('   reconstructing T[SMICA] x lss[unWISE]')
Noise_TSMICA_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_SMICA, Cldd_unwise)]*6144)
Noise_TSMICA_gunwise_unmaskedlss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_SMICA, Cldd_unwise_unmasked)]*6144)
_, _, outmap_TSMICA_gunwise = combine(Tmap_SMICA, lssmap_unwise, ClTT_SMICA, Cldd_unwise, Noise_TSMICA_gunwise)
_, _, outmap_TSMICA_gunwise_unmaskedlss = combine(Tmap_SMICA, lssmap_unwise_inp, ClTT_SMICA, Cldd_unwise_unmasked, Noise_TSMICA_gunwise_unmaskedlss)
print('      plotting...')
twopt(outmap_TSMICA_gunwise, Noise_TSMICA_gunwise, fsky, r'Power Spectrum of T[SMICA] $\times$ lss[unWISE]', 'twopt_TSMICA_gunwise')
twopt(outmap_TSMICA_gunwise_unmaskedlss, Noise_TSMICA_gunwise_unmaskedlss, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[SMICA] $\times$ unmasked lss[unWISE]', 'twopt_TSMICA_gunwise_unmaskedlss')

print('   reconstructing T[freq-SMICA] x masked lss[unWISE]')
Noise_TSMICAsub100_gunwise_maskedlss = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_SMICAsub_100, Cldd_unwise_unmasked)]*6144)
_, _, outmap_TSMICAsub100_gunwise_unmaskedlss = combine(Tmap_SMICAsub_100, lssmap_unwise_inp, ClTT_SMICAsub_100, Cldd_unwise_unmasked, Noise_TSMICAsub100_gunwise_maskedlss)
print('      plotting...')
twopt(outmap_TSMICAsub100_gunwise_unmaskedlss, Noise_TSMICAsub100_gunwise_maskedlss, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[100GHz-SMICA] $\times$ unmasked lss[unWISE]', 'twopt_TSMICAsub100_gunwise_unmaskedlss')





Tmap_gauss = hp.synfast(hp.anafast(Tmap_SMICA_noised),2048)
Tmap_gauss = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_gauss*mask_map), 1/beam_SMICA), 2048)
Tcl_gauss = hp.anafast(Tmap_gauss) / fsky
Tcl_gauss = np.append(Tcl_gauss, Tcl_gauss[-1])
A = np.array([Noise_vr_diag(6144, 0, 0, 5, Tcl_gauss, Cldd_unwise_unmasked)]*6144)
_, _, X = combine(Tmap_gauss, lssmap_unwise_inp, Tcl_gauss, Cldd_unwise_unmasked, A)
print('      plotting...')
twopt(X, A, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[gauss] $\times$ unmasked lss[unWISE]', 'twopt_Tgauss_gunwise_unmaskedlss')


### Component separated: Theory ClTTT include foregrounds and CMB to see variance budget
### Writeup of plots and their combinations
### As masking grows what is effect on reconstruction
### inpainting of the lss unwise map with masked pixel values drawn from Poisson of mean n_galaxies per pix (unmasked area)
### Integrate tau dot but still multiply by Delta Chi