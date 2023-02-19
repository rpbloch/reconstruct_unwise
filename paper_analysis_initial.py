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
from scipy.integrate import simps


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
zmin = conf.z_min
zmax = conf.z_max
bin_centres = lambda bins : (bins[1:]+bins[:-1]) / 2
basic_conf_dir = c.get_basic_conf(conf)
estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(conf, exclude = False))+'/T_freq='+str(143)+'/'  
outdir = 'plots/analysis/Planck_unWISE_continuation/'
if not os.path.exists(outdir):
    os.makedirs(outdir)


with open('data/unWISE/blue.txt', 'r') as FILE:
    lines = FILE.readlines()

zs_unwise = np.array([float(line.split(' ')[0]) for line in lines])
dn_unwise = np.array([float(line.split(' ')[1]) for line in lines])

maxval = np.abs(simps(zs_unwise, dn_unwise))

### Justifies considering only redshifts out to z=1.6, which encapsulates 99.3% of the galaxy window.
### Minimum nonzero data is at z=0.01 so this should set the cosmology for our setup.
plt.figure()
plt.plot(zs_unwise, dn_unwise)
ylim_store = plt.ylim()
for i in np.logspace(np.log10(1.3), np.log10(1.6), 5):
    zcap = np.where(zs_unwise<=i)[0][-1]
    plt.plot([i,i],ylim_store,ls='--',c='gray')
    plt.text(i,(ylim_store[-1]*.8)-(2*i)+2.5,r'  %.3f%% @ z$\leq$%.2f'%((np.abs(simps(zs_unwise[:zcap], dn_unwise[:zcap])) / maxval),zs_unwise[zcap]))

plt.xlim([0,2])
plt.xlabel(r'$z$')
plt.ylabel(r'$\mathrm{N}\left(z\right)$  (normalized)')
plt.title('unWISE Blue Sample Redshift Distribution')
plt.savefig(outdir + 'unwise_blue_dndz_windowcap')


def a(z):
    return 1/(1+z)

def ne0z(z):  # units 1/m^3, no actual z dependence?????
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
bin_width = (cosmology_data.comoving_radial_distance(zmax)-cosmology_data.comoving_radial_distance(zmin)) * mperMpc
zrange = np.where(np.logical_and(zs_unwise>=zmin,zs_unwise<=zmax))[0]
zmin_ind = zrange.min()
zmax_ind = zrange.max()
a_integrated = np.abs(simps(zs_unwise[zmin_ind:zmax_ind], a(zs_unwise[zmin_ind:zmax_ind])))
K = -sigma_T * bin_width * a_integrated * ne0z(zmax)  # conversion from delta to tau dot, unitless conversion

print('Setting up maps and Cls')
print('   loading maps')
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
Tmap_SMICA_inp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True)
Tmap_100_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_143_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_217_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_SMICAsub_100_inp = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten()
Tmap_SMICAsub_143_inp = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits')[1].data['INTENSITY'].flatten()
Tmap_SMICAsub_217_inp = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits')[1].data['INTENSITY'].flatten()
lssmap_unwise_inp = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
lssmap_gauss_inp = hp.synfast(hp.anafast(lssmap_unwise_inp), 2048)

Clvv = loginterp.log_interpolate_matrix(c.load(basic_conf_dir,'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(basic_conf_dir,'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

fsky        = np.where(mask_map!=0)[0].size / mask_map.size
fsky_unwise = np.where((mask_map*lssmap_unwise_inp)!=0)[0].size / mask_map.size

print('   masking and debeaming T maps and computing spectra')
beam_SMICA = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143)
beam_100   = hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6143)
beam_143   = hp.gauss_beam(fwhm=np.radians(7.27/60), lmax=6143)
beam_217   = hp.gauss_beam(fwhm=np.radians(5.01/60), lmax=6143)

beam_100[4001:] = beam_100[4000]  # Numerical trouble for healpix for beams that blow up at large ell. Sufficient to flatten out at high ell above where the 1/TT filter peaks

Tcl_SMICA_maxdata = hp.anafast(Tmap_SMICA_inp, lmax=2500)[-1]
faux_noisemap = hp.synfast([Tcl_SMICA_maxdata for i in np.arange(6144)], 2048)
Tmap_SMICA_noised = Tmap_SMICA_inp + faux_noisemap
Tmap_SMICA_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_SMICA_noised*mask_map), 1/beam_SMICA)
Tmap_SMICAgauss_alm_masked_debeamed = hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(Tmap_SMICA_noised),2048)*mask_map), 1/beam_SMICA)
Tmap_100_alm_masked_debeamed   = hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)
Tmap_143_alm_masked_debeamed   = hp.almxfl(hp.map2alm(Tmap_143_inp*mask_map), 1/beam_143)
Tmap_217_alm_masked_debeamed   = hp.almxfl(hp.map2alm(Tmap_217_inp*mask_map), 1/beam_217)
Tmap_SMICAsub_100_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_SMICAsub_100_inp*mask_map), 1/beam_100)
Tmap_SMICAsub_143_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_SMICAsub_143_inp*mask_map), 1/beam_143)
Tmap_SMICAsub_217_alm_masked_debeamed = hp.almxfl(hp.map2alm(Tmap_SMICAsub_217_inp*mask_map), 1/beam_217)
Tmap_100gauss_alm_masked_debeamed = hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(Tmap_100_inp),2048)*mask_map), 1/beam_100)
Tmap_143gauss_alm_masked_debeamed = hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(Tmap_143_inp),2048)*mask_map), 1/beam_143)
Tmap_217gauss_alm_masked_debeamed = hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(Tmap_217_inp),2048)*mask_map), 1/beam_217)

Tmap_SMICA = hp.alm2map(Tmap_SMICA_alm_masked_debeamed, 2048)
Tmap_SMICAgauss = hp.alm2map(Tmap_SMICAgauss_alm_masked_debeamed, 2048)
Tmap_100   = hp.alm2map(Tmap_100_alm_masked_debeamed, 2048)
Tmap_143   = hp.alm2map(Tmap_143_alm_masked_debeamed, 2048)
Tmap_217   = hp.alm2map(Tmap_217_alm_masked_debeamed, 2048)
Tmap_SMICAsub_100 = hp.alm2map(Tmap_SMICAsub_100_alm_masked_debeamed, 2048)
Tmap_SMICAsub_143 = hp.alm2map(Tmap_SMICAsub_143_alm_masked_debeamed, 2048)
Tmap_SMICAsub_217 = hp.alm2map(Tmap_SMICAsub_217_alm_masked_debeamed, 2048)
Tmap_100gauss = hp.alm2map(Tmap_100gauss_alm_masked_debeamed, 2048)
Tmap_143gauss = hp.alm2map(Tmap_143gauss_alm_masked_debeamed, 2048)
Tmap_217gauss = hp.alm2map(Tmap_217gauss_alm_masked_debeamed, 2048)

### Load / compute Cls (bin 0)
# Temperature
ClTT_SMICA = hp.alm2cl(Tmap_SMICA_alm_masked_debeamed) / fsky
ClTT_SMICAgauss = hp.alm2cl(Tmap_SMICAgauss_alm_masked_debeamed) / fsky
ClTT_100 = hp.alm2cl(Tmap_100_alm_masked_debeamed) / fsky
ClTT_143 = hp.alm2cl(Tmap_143_alm_masked_debeamed) / fsky
ClTT_217 = hp.alm2cl(Tmap_217_alm_masked_debeamed) / fsky
ClTT_SMICAsub_100 = hp.alm2cl(Tmap_SMICAsub_100_alm_masked_debeamed) / fsky
ClTT_SMICAsub_143 = hp.alm2cl(Tmap_SMICAsub_143_alm_masked_debeamed) / fsky
ClTT_SMICAsub_217 = hp.alm2cl(Tmap_SMICAsub_217_alm_masked_debeamed) / fsky
ClTT_100gauss = hp.alm2cl(Tmap_100gauss_alm_masked_debeamed) / fsky
ClTT_143gauss = hp.alm2cl(Tmap_143gauss_alm_masked_debeamed) / fsky
ClTT_217gauss = hp.alm2cl(Tmap_217gauss_alm_masked_debeamed) / fsky

print('   masking lss maps and computing spectra')
lssmap_gauss  = lssmap_gauss_inp.copy()
lssmap_gauss_masked  = lssmap_gauss_inp * mask_map
lssmap_unwise = lssmap_unwise_inp * mask_map

Cldd_gauss = hp.anafast(lssmap_gauss)
Cldd_gauss_masked = hp.anafast(lssmap_gauss_masked) / fsky
Cldd_unwise = hp.anafast(lssmap_unwise) / fsky
Cldd_unwise_unmasked = hp.anafast(lssmap_unwise_inp)

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

def combine(Tmap, lssmap, ClTT, Clgg, Noise, convert_K=True):
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
    if convert_K:  # output map has units of K
        outmap /= 2.725
    return Tmap_filtered * const_noise, lssmap_filtered, outmap, hp.anafast(outmap)

def twopt(recon_Cls, theory_noise, FSKY, plottitle, filename, lmaxplot=700, convert_K=True):
    plt.figure()
    plt.loglog(np.arange(2,lmaxplot), recon_Cls[2:lmaxplot], label='Reconstruction')
    if convert_K:
        plt.loglog(np.arange(2,lmaxplot), theory_noise[2:lmaxplot] * FSKY / 2.725**2, label='Theory * fsky')
    else:
        plt.loglog(np.arange(2,lmaxplot), theory_noise[2:lmaxplot] * FSKY, label='Theory * fsky')
    plt.loglog(np.arange(2,10), Clvv[2:10], label='Theory Signal') 
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')



print('Reconstructing')
print('   reconstructing T 100GHz x lss')
Noise_T100gauss_ggauss         = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100gauss, Cldd_gauss)]*6143)
Noise_T100gauss_ggauss_masklss = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100gauss, Cldd_gauss_masked)]*6143)
Noise_T100gauss_gunwise        = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100gauss, Cldd_unwise_unmasked)]*6143)
Noise_T100_ggauss              = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100, Cldd_gauss)]*6143)
Noise_T100_gunwise             = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100, Cldd_unwise_unmasked)]*6143)

Tmap_100gauss_filtered, lssmap_gauss_filtered, outmap_T100gauss_ggauss, Cl_T100gauss_ggauss = combine(Tmap_100gauss, lssmap_gauss, ClTT_100gauss, Cldd_gauss, Noise_T100gauss_ggauss)
_, lssmap_gauss_masked_filtered, outmap_T100gauss_ggauss_masklss, Cl_T100gauss_ggauss_masklss = combine(Tmap_100gauss, lssmap_gauss_masked, ClTT_100gauss, Cldd_gauss_masked, Noise_T100gauss_ggauss_masklss)
_, lssmap_unwise_filtered, outmap_T100gauss_gunwise, Cl_T100gauss_gunwise = combine(Tmap_100gauss, lssmap_unwise_inp, ClTT_100gauss, Cldd_unwise_unmasked, Noise_T100gauss_gunwise)
Tmap_100_filtered, _, outmap_T100_ggauss, Cl_T100_ggauss = combine(Tmap_100, lssmap_gauss, ClTT_100, Cldd_gauss, Noise_T100_ggauss)
_, _, outmap_T100_gunwise, Cl_T100_gunwise = combine(Tmap_100, lssmap_unwise_inp, ClTT_100, Cldd_unwise_unmasked, Noise_T100_gunwise)

twopt(Cl_T100gauss_ggauss, Noise_T100gauss_ggauss, fsky, r'Power Spectrum of gauss(T[100GHz]) $\times$ lss[gauss]', 'twopt_T100gauss_ggauss')
twopt(Cl_T100gauss_ggauss_masklss, Noise_T100gauss_ggauss_masklss, fsky, r'Power Spectrum of gauss(T[100GHz]) $\times$ masked lss[gauss]', 'twopt_T100gauss_ggauss_masklss')
twopt(Cl_T100gauss_gunwise, Noise_T100gauss_gunwise, fsky_unwise, r'Power Spectrum of gauss(T[100GHz]) $\times$ lss[unWISE]', 'twopt_T100gauss_gunwise')
twopt(Cl_T100_ggauss, Noise_T100_ggauss, fsky, r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss')
twopt(Cl_T100_gunwise, Noise_T100_gunwise, fsky_unwise, r'Power Spectrum of T[100GHz] $\times$ lss[unWISE]', 'twopt_T100_gunwise')

print('   reconstructing T 143GHz x lss')
Noise_T143gauss_ggauss         = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143gauss, Cldd_gauss)]*6143)
Noise_T143gauss_gunwise        = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143gauss, Cldd_unwise_unmasked)]*6143)
Noise_T143_ggauss              = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143, Cldd_gauss)]*6143)
Noise_T143_gunwise             = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143, Cldd_unwise_unmasked)]*6143)

Tmap_143gauss_filtered, lssmap_gauss_filtered, outmap_T143gauss_ggauss, Cl_T143gauss_ggauss = combine(Tmap_143gauss, lssmap_gauss, ClTT_143gauss, Cldd_gauss, Noise_T143gauss_ggauss)
_, lssmap_unwise_filtered, outmap_T143gauss_gunwise, Cl_T143gauss_gunwise = combine(Tmap_143gauss, lssmap_unwise_inp, ClTT_143gauss, Cldd_unwise_unmasked, Noise_T143gauss_gunwise)
Tmap_143_filtered, _, outmap_T143_ggauss, Cl_T143_ggauss = combine(Tmap_143, lssmap_gauss, ClTT_143, Cldd_gauss, Noise_T143_ggauss)
_, _, outmap_T143_gunwise, Cl_T143_gunwise = combine(Tmap_143, lssmap_unwise_inp, ClTT_143, Cldd_unwise_unmasked, Noise_T143_gunwise)

twopt(Cl_T143gauss_ggauss, Noise_T143gauss_ggauss, fsky, r'Power Spectrum of gauss(T[143GHz]) $\times$ lss[gauss]', 'twopt_T143gauss_ggauss')
twopt(Cl_T143gauss_gunwise, Noise_T143gauss_gunwise, fsky_unwise, r'Power Spectrum of gauss(T[143GHz]) $\times$ lss[unWISE]', 'twopt_T143gauss_gunwise')
twopt(Cl_T143_ggauss, Noise_T143_ggauss, fsky, r'Power Spectrum of T[143GHz] $\times$ lss[gauss]', 'twopt_T143_ggauss')
twopt(Cl_T143_gunwise, Noise_T143_gunwise, fsky_unwise, r'Power Spectrum of T[143GHz] $\times$ lss[unWISE]', 'twopt_T143_gunwise')

print('   reconstructing T 217GHz x lss')
Noise_T217gauss_ggauss         = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217gauss, Cldd_gauss)]*6143)
Noise_T217gauss_gunwise        = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217gauss, Cldd_unwise_unmasked)]*6143)
Noise_T217_ggauss              = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217, Cldd_gauss)]*6143)
Noise_T217_gunwise             = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217, Cldd_unwise_unmasked)]*6143)

Tmap_217gauss_filtered, lssmap_gauss_filtered, outmap_T217gauss_ggauss, Cl_T217gauss_ggauss = combine(Tmap_217gauss, lssmap_gauss, ClTT_217gauss, Cldd_gauss, Noise_T217gauss_ggauss)
_, lssmap_unwise_filtered, outmap_T217gauss_gunwise, Cl_T217gauss_gunwise = combine(Tmap_217gauss, lssmap_unwise_inp, ClTT_217gauss, Cldd_unwise_unmasked, Noise_T217gauss_gunwise)
Tmap_217_filtered, _, outmap_T217_ggauss, Cl_T217_ggauss = combine(Tmap_217, lssmap_gauss, ClTT_217, Cldd_gauss, Noise_T217_ggauss)
_, _, outmap_T217_gunwise, Cl_T217_gunwise = combine(Tmap_217, lssmap_unwise_inp, ClTT_217, Cldd_unwise_unmasked, Noise_T217_gunwise)

twopt(Cl_T217gauss_ggauss, Noise_T217gauss_ggauss, fsky, r'Power Spectrum of gauss(T[217GHz]) $\times$ lss[gauss]', 'twopt_T217gauss_ggauss')
twopt(Cl_T217gauss_gunwise, Noise_T217gauss_gunwise, fsky_unwise, r'Power Spectrum of gauss(T[217GHz]) $\times$ lss[unWISE]', 'twopt_T217gauss_gunwise')
twopt(Cl_T217_ggauss, Noise_T217_ggauss, fsky, r'Power Spectrum of T[217GHz] $\times$ lss[gauss]', 'twopt_T217_ggauss')
twopt(Cl_T217_gunwise, Noise_T217_gunwise, fsky_unwise, r'Power Spectrum of T[217GHz] $\times$ lss[unWISE]', 'twopt_T217_gunwise')

print('   reconstructing SMICA x lss')
Noise_TSMICAgauss_ggauss         = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_gauss)]*6143)
Noise_TSMICAgauss_gunwise        = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_unwise_unmasked)]*6143)
Noise_TSMICA_ggauss              = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICA, Cldd_gauss)]*6143)
Noise_TSMICA_gunwise             = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICA, Cldd_unwise_unmasked)]*6143)

Tmap_SMICAgauss_filtered, lssmap_gauss_filtered, outmap_TSMICAgauss_ggauss, Cl_TSMICAgauss_ggauss = combine(Tmap_SMICAgauss, lssmap_gauss, ClTT_SMICAgauss, Cldd_gauss, Noise_TSMICAgauss_ggauss)
_, lssmap_unwise_filtered, outmap_TSMICAgauss_gunwise, Cl_TSMICAgauss_gunwise = combine(Tmap_SMICAgauss, lssmap_unwise_inp, ClTT_SMICAgauss, Cldd_unwise_unmasked, Noise_TSMICAgauss_gunwise)
Tmap_SMICA_filtered, _, outmap_TSMICA_ggauss, Cl_TSMICA_ggauss = combine(Tmap_SMICA, lssmap_gauss, ClTT_SMICA, Cldd_gauss, Noise_TSMICA_ggauss)
_, _, outmap_TSMICA_gunwise, Cl_TSMICA_gunwise = combine(Tmap_SMICA, lssmap_unwise_inp, ClTT_SMICA, Cldd_unwise_unmasked, Noise_TSMICA_gunwise)

twopt(Cl_TSMICAgauss_ggauss, Noise_TSMICAgauss_ggauss, fsky, r'Power Spectrum of gauss(T[SMICA]) $\times$ lss[gauss]', 'twopt_TSMICAgauss_ggauss')
twopt(Cl_TSMICAgauss_gunwise, Noise_TSMICAgauss_gunwise, fsky_unwise, r'Power Spectrum of gauss(T[SMICA]) $\times$ lss[unWISE]', 'twopt_TSMICAgauss_gunwise')
twopt(Cl_TSMICA_ggauss, Noise_TSMICA_ggauss, fsky, r'Power Spectrum of T[SMICA] $\times$ lss[gauss]', 'twopt_TSMICA_ggauss')
twopt(Cl_TSMICA_gunwise, Noise_TSMICA_gunwise, fsky_unwise, r'Power Spectrum of T[SMICA] $\times$ lss[unWISE]', 'twopt_TSMICA_gunwise')

# ### Component separated: Theory ClTTT include foregrounds and CMB to see variance budget
print('   reconstructing T[CMB-subtracted] x lss[unWISE]')
Noise_TSMICAsub100_gunwise = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100, Cldd_unwise_unmasked)]*6143)
Noise_TSMICAsub143_gunwise = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143, Cldd_unwise_unmasked)]*6143)
Noise_TSMICAsub217_gunwise = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217, Cldd_unwise_unmasked)]*6143)

_, _, outmap_TSMICAsub100_gunwise, Cl_TSMICAsub100_gunwise = combine(Tmap_SMICAsub_100, lssmap_unwise_inp, ClTT_100, Cldd_unwise_unmasked, Noise_TSMICAsub100_gunwise)
_, _, outmap_TSMICAsub143_gunwise, Cl_TSMICAsub143_gunwise = combine(Tmap_SMICAsub_143, lssmap_unwise_inp, ClTT_143, Cldd_unwise_unmasked, Noise_TSMICAsub143_gunwise)
_, _, outmap_TSMICAsub217_gunwise, Cl_TSMICAsub217_gunwise = combine(Tmap_SMICAsub_217, lssmap_unwise_inp, ClTT_217, Cldd_unwise_unmasked, Noise_TSMICAsub217_gunwise)

twopt(Cl_TSMICAsub100_gunwise, Noise_TSMICAsub100_gunwise, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[100GHz-SMICA] $\times$ lss[unWISE]', 'twopt_TSMICAsub100_gunwise')
twopt(Cl_TSMICAsub143_gunwise, Noise_TSMICAsub143_gunwise, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[143GHz-SMICA] $\times$ lss[unWISE]', 'twopt_TSMICAsub143_gunwise')
twopt(Cl_TSMICAsub217_gunwise, Noise_TSMICAsub217_gunwise, fsky*(np.where(lssmap_unwise_inp!=0)[0].size/mask_map.size), r'Power Spectrum of T[217GHz-SMICA] $\times$ lss[unWISE]', 'twopt_TSMICAsub217_gunwise')

### inpainting of the lss unwise map with masked pixel values drawn from Poisson of mean n_galaxies per pix (unmasked area)
ngal_per_pix = lssmap_unwise_inp.sum() / np.where(lssmap_unwise_inp!=0)[0].size
ngal_fill = np.random.poisson(lam=ngal_per_pix, size=np.where(lssmap_unwise_inp==0)[0].size)
lssmap_unwise_inpainted = lssmap_unwise_inp.copy()
lssmap_unwise_inpainted[np.where(lssmap_unwise_inp==0)] = ngal_fill

Cldd_unwise_inpainted = hp.anafast(lssmap_unwise_inpainted)

hp.mollview(lssmap_unwise_inpainted, title='Inpainted unWISE Map')
plt.savefig(outdir + 'unWISE_inpainted')

plt.figure()
plt.loglog(Cldd_unwise_unmasked, label='Original Map')
plt.loglog(Cldd_unwise_inpainted, label='Inpainted Map')
plt.ylabel(r'$C_\ell^{\mathrm{gg}}$')
plt.xlabel(r'$\ell$')
plt.legend()
plt.savefig(outdir+'Inpainting Clgg')

print('Original unWISE map:  %.1f%% zero-valued pixels' % (143*np.where(lssmap_unwise_inp==0)[0].size/lssmap_unwise_inp.size))
print('Inpainted unWISE map: %.1f%% zero-valued pixels' % (143*np.where(lssmap_unwise_inpainted==0)[0].size/lssmap_unwise_inp.size))

print('   reconstructing T x inpainted unWISE map')
#Noise_T100gauss_gunwise_inpainted = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100gauss, Cldd_unwise_inpainted)]*6143)
#Noise_T100_gunwise_inpainted      = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_100, Cldd_unwise_inpainted)]*6143)
#Noise_T143gauss_gunwise_inpainted = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143gauss, Cldd_unwise_inpainted)]*6143)
#Noise_T143_gunwise_inpainted      = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_143, Cldd_unwise_inpainted)]*6143)
#Noise_T217gauss_gunwise_inpainted = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217gauss, Cldd_unwise_inpainted)]*6143)
#Noise_T217_gunwise_inpainted      = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_217, Cldd_unwise_inpainted)]*6143)
Noise_TSMICAgauss_gunwise_inpainted = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_unwise_inpainted)]*6143)
Noise_TSMICA_gunwise_inpainted      = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICA, Cldd_unwise_inpainted)]*6143)

#_, lssmap_unwise_filtered_inpainted, outmap_T100gauss_gunwise_inpainted, Cl_T100gauss_gunwise_inpainted = combine(Tmap_100gauss, lssmap_unwise_inpainted, ClTT_100gauss, Cldd_unwise_inpainted, Noise_T100gauss_gunwise_inpainted)
#_, _, outmap_T100_gunwise_inpainted, Cl_T100_gunwise_inpainted = combine(Tmap_100, lssmap_unwise_inpainted, ClTT_100, Cldd_unwise_inpainted, Noise_T100_gunwise_inpainted)
#_, _, outmap_T143gauss_gunwise_inpainted, Cl_T143gauss_gunwise_inpainted = combine(Tmap_143gauss, lssmap_unwise_inpainted, ClTT_143gauss, Cldd_unwise_inpainted, Noise_T143gauss_gunwise_inpainted)
#_, _, outmap_T143_gunwise_inpainted, Cl_T143_gunwise_inpainted = combine(Tmap_143, lssmap_unwise_inpainted, ClTT_143, Cldd_unwise_inpainted, Noise_T143_gunwise_inpainted)
#_, _, outmap_T217gauss_gunwise_inpainted, Cl_T217gauss_gunwise_inpainted = combine(Tmap_217gauss, lssmap_unwise_inpainted, ClTT_217gauss, Cldd_unwise_inpainted, Noise_T217gauss_gunwise_inpainted)
#_, _, outmap_T217_gunwise_inpainted, Cl_T217_gunwise_inpainted = combine(Tmap_217, lssmap_unwise_inpainted, ClTT_217, Cldd_unwise_inpainted, Noise_T217_gunwise_inpainted)
_, _, outmap_TSMICAgauss_gunwise_inpainted, Cl_TSMICAgauss_gunwise_inpainted = combine(Tmap_SMICAgauss, lssmap_unwise_inpainted, ClTT_SMICAgauss, Cldd_unwise_inpainted, Noise_TSMICAgauss_gunwise_inpainted)
_, _, outmap_TSMICA_gunwise_inpainted, Cl_TSMICA_gunwise_inpainted = combine(Tmap_SMICA, lssmap_unwise_inpainted, ClTT_SMICA, Cldd_unwise_inpainted, Noise_TSMICA_gunwise_inpainted)

#twopt(Cl_T100gauss_gunwise_inpainted, Noise_T100gauss_gunwise_inpainted, fsky, r'Power Spectrum of gauss(T[100GHz]) $\times$ inpainted lss[unWISE]', 'twopt_T100gauss_gunwise_inpainted')
#twopt(Cl_T100_gunwise_inpainted, Noise_T100_gunwise_inpainted, fsky, r'Power Spectrum of T[100GHz] $\times$ inpainted lss[unWISE]', 'twopt_T100_gunwise_inpainted')
#twopt(Cl_T143gauss_gunwise_inpainted, Noise_T143gauss_gunwise_inpainted, fsky, r'Power Spectrum of gauss(T[143GHz]) $\times$ inpainted lss[unWISE]', 'twopt_T143gauss_gunwise_inpainted')
#twopt(Cl_T143_gunwise_inpainted, Noise_T143_gunwise_inpainted, fsky, r'Power Spectrum of T[143GHz] $\times$ inpainted lss[unWISE]', 'twopt_T143_gunwise_inpainted')
#twopt(Cl_T217gauss_gunwise_inpainted, Noise_T217gauss_gunwise_inpainted, fsky, r'Power Spectrum of gauss(T[217GHz]) $\times$ inpainted lss[unWISE]', 'twopt_T217gauss_gunwise_inpainted')
#twopt(Cl_T217_gunwise_inpainted, Noise_T217_gunwise_inpainted, fsky, r'Power Spectrum of T[217GHz] $\times$ inpainted lss[unWISE]', 'twopt_T217_gunwise_inpainted')
twopt(Cl_TSMICAgauss_gunwise_inpainted, Noise_TSMICAgauss_gunwise_inpainted, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ inpainted lss[unWISE]', 'twopt_TSMICAgauss_gunwise_inpainted')
twopt(Cl_TSMICA_gunwise_inpainted, Noise_TSMICA_gunwise_inpainted, fsky, r'Power Spectrum of T[SMICAGHz] $\times$ inpainted lss[unWISE]', 'twopt_TSMICA_gunwise_inpainted')

## What if we inpainted inside the whole mask
ngal_fill = np.random.poisson(lam=ngal_per_pix, size=np.where((mask_map)==0)[0].size)
lssmap_unwise_inpainted_large = lssmap_unwise_inp.copy()
lssmap_unwise_inpainted_large[np.where((mask_map)==0)] = ngal_fill

Cldd_unwise_inpainted_large = hp.anafast(lssmap_unwise_inpainted_large)

hp.mollview(lssmap_unwise_inpainted_large, title='Inpainted unWISE Map')
plt.savefig(outdir + 'unWISE_inpainted_maskedregion')

plt.figure()
plt.loglog(Cldd_unwise_unmasked, label='Original Map')
plt.loglog(Cldd_unwise_inpainted, label='Inpainted on zero-valued')
plt.loglog(Cldd_unwise_inpainted_large, label='Inpainted on zero-valued+mask')
plt.ylabel(r'$C_\ell^{\mathrm{gg}}$')
plt.xlabel(r'$\ell$')
plt.legend()
plt.savefig(outdir+'Inpainting Clgg (masked region)')

print('Original unWISE map:  %.1f%% zero-valued pixels' % (143*np.where(lssmap_unwise_inp==0)[0].size/lssmap_unwise_inp.size))
print('Inpainted unWISE map: %.1f%% zero-valued pixels' % (143*np.where(lssmap_unwise_inpainted_large==0)[0].size/lssmap_unwise_inp.size))

print('   reconstructing T x inpainted unWISE map')
Noise_TSMICAgauss_gunwise_inpainted_large = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_unwise_inpainted_large)]*6143)

_, lssmap_unwise_filtered_inpainted_large, outmap_TSMICAgauss_gunwise_inpainted_large, Cl_TSMICAgauss_gunwise_inpainted_large = combine(Tmap_SMICAgauss, lssmap_unwise_inpainted_large, ClTT_SMICAgauss, Cldd_unwise_inpainted_large, Noise_TSMICAgauss_gunwise_inpainted_large)

twopt(Cl_TSMICAgauss_gunwise_inpainted_large, Noise_TSMICAgauss_gunwise_inpainted_large, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ mask-region-inpainted lss[unWISE]', 'twopt_TSMICAgauss_gunwise_inpainted_large')


### recon actual, recon inpainted. Same inpainted Clgg for both to get same prefactor.
Noise_TSMICAgauss_gunwise_inpainted_large = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_unwise_inpainted_large)]*6143)

_, x, out_x, Cl_x = combine(Tmap_SMICAgauss, lssmap_unwise_inp, ClTT_SMICAgauss, Cldd_unwise_inpainted_large, Noise_TSMICAgauss_gunwise_inpainted_large)

twopt(Cl_TSMICAgauss_gunwise_inpainted_large, Noise_TSMICAgauss_gunwise_inpainted_large, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ mask-region-inpainted lss[unWISE]', 'twopt_TSMICAgauss_gunwise_inpainted_large')

### Would be nice to do multiple realizations to see the cosmic variance error bars
### Might be too big a project, but we could consider what happens if we inpaint without destroying correlations
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 1) Consider SMICA x unWISE. (If it's about as good as what we see below, we can consider what might improve things. Such as an actual tau filter)
### 2) Clgg = Pmm * bias(z) * bias(z) + shotnoise
### 3) Cltaug = Pmm * K * bias(z)
### Instead of the green input Clgg from the case plots, use these spectra instead. This should be better like Alex's.
plt.figure()
plt.loglog(np.arange(2,lmaxplot), Cl_TSMICAgauss_gunwise_inpainted_large[2:lmaxplot], label='Reconstruction (inpaint)')
plt.loglog(np.arange(2,lmaxplot), Cl_x[2:lmaxplot], label='Reconstruction (input)')
plt.loglog(np.arange(2,lmaxplot), Noise_TSMICAgauss_gunwise_inpainted_large[2:lmaxplot] * FSKY / 2.725**2, label='Theory * fsky')
plt.loglog(np.arange(2,10), Clvv[2:10], label='Theory Signal') 
plt.xlabel(r'$\ell$')
plt.ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$')
plt.legend()
plt.savefig(outdir + 'compare')
plt.close('all')


### inpaint pixels with galcounts greater than 3 standard deviations from the mean

inverted_mask_map = 1-mask_map.copy()
hp.mollview(inverted_mask_map)
plt.savefig(outdir + 'inverted_mask_map')

test_fill = np.random.poisson(lam=ngal_per_pix, size=np.where((inverted_mask_map * lssmap_unwise_inp) > (3*np.std(lssmap_unwise_inp)))[0].size)
test_lssmap = lssmap_unwise_inp.copy()
test_lssmap[np.where((inverted_mask_map * lssmap_unwise_inp) > (3*np.std(lssmap_unwise_inp)))] = test_fill
hp.mollview(test_lssmap)
plt.savefig(outdir+'masked overage')
Cldd_test = hp.anafast(test_lssmap)

Noise_test = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_test)]*6143)

_, _, outmap_test, Cl_test = combine(Tmap_SMICAgauss, test_lssmap, ClTT_SMICAgauss, Cldd_test, Noise_test)

twopt(Cl_test, Noise_test, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ mask-region-inpainted 3std+ lss[unWISE]', 'twopt_test')

### inpaint pixels with galcounts greater than 1 standard deviation from the mean
test2_fill = np.random.poisson(lam=ngal_per_pix, size=np.where((inverted_mask_map * lssmap_unwise_inp) > (np.std(lssmap_unwise_inp)))[0].size)
test2_lssmap = lssmap_unwise_inp.copy()
test2_lssmap[np.where((inverted_mask_map * lssmap_unwise_inp) > (np.std(lssmap_unwise_inp)))] = test2_fill
hp.mollview(test2_lssmap)
plt.savefig(outdir+'masked overage2')
Cldd_test2 = hp.anafast(test2_lssmap)

Noise_test2 = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_test2)]*6143)

_, _, outmap_test2, Cl_test2 = combine(Tmap_SMICAgauss, test2_lssmap, ClTT_SMICAgauss, Cldd_test2, Noise_test2)

twopt(Cl_test2, Noise_test2, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ mask-region-inpainted std+ lss[unWISE]', 'twopt_test2')


### inpaint pixels in the 70% planck galactic cut only
mask_planck70=hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL070'],n2r=True)
test3_fill = np.random.poisson(lam=ngal_per_pix, size=np.where(mask_planck70==0)[0].size)
test3_lssmap = lssmap_unwise_inp.copy()
test3_lssmap[np.where(mask_planck70==0)] = test3_fill
hp.mollview(test3_lssmap)
plt.savefig(outdir+'masked overage3')
Cldd_test3 = hp.anafast(test3_lssmap)

Noise_test3 = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_test3)]*6143)

_, _, outmap_test3, Cl_test3 = combine(Tmap_SMICAgauss, test3_lssmap, ClTT_SMICAgauss, Cldd_test3, Noise_test3)

twopt(Cl_test3, Noise_test3, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ mask-galplane-inpainted lss[unWISE]', 'twopt_test3')








### inpaint pixels with galcounts closer to the mean than 1 standard deviation
test4_fill = np.random.poisson(lam=ngal_per_pix, size=np.where((inverted_mask_map * lssmap_unwise_inp) < (np.std(lssmap_unwise_inp)))[0].size)
test4_lssmap = lssmap_unwise_inp.copy()
test4_lssmap[np.where((inverted_mask_map * lssmap_unwise_inp) < (np.std(lssmap_unwise_inp)))] = test4_fill
hp.mollview(test4_lssmap)
plt.savefig(outdir+'masked overage2')
Cldd_test4 = hp.anafast(test4_lssmap)

Noise_test4 = np.array([Noise_vr_diag(6143, 0, 0, 5, ClTT_SMICAgauss, Cldd_test4)]*6143)

_, _, outmap_test4, Cl_test4 = combine(Tmap_SMICAgauss, test4_lssmap, ClTT_SMICAgauss, Cldd_test4, Noise_test4)

twopt(Cl_test4, Noise_test4, fsky, r'Power Spectrum of gauss(T[SMICAGHz]) $\times$ mask-region-inpainted std+ lss[unWISE]', 'twopt_test4')



##
with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
    lines = FILE.readlines()

alex_ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float')
alex_clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')
lssspec = interp1d(alex_ells,alex_clgg, bounds_error=False,fill_value='extrapolate')(np.arange(6144))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))
for ax in (ax1, ax2):
    ax.loglog(Cldd_unwise_unmasked, label='Original Map')
    ax.loglog(Cldd_unwise_inpainted, label='Inpainted on zero-valued')
    ax.loglog(Cldd_unwise_inpainted_large, label='Inpainted on zero-valued+mask')
    ax.loglog(Cldd_test2, label='Inpaint masked std or greater')
    ax.loglog(Cldd_test, label='Inpaint masked 3std or greater')
    ax.loglog(Cldd_test3, label='Inpaint only galplane mask')
    ax.loglog(lssspec*(lssmap_unwise_inp.sum()/mask_map.size)**2, label='Alex\'s Clgg')
    ax.set_xlabel(r'$\ell$')
    ax.legend()

ax2.set_ylim([5e-7,5e-6])
ax2.set_xlim([1e3, 7500])
ax1.set_ylabel(r'$C_\ell^{\mathrm{gg}}$')
plt.savefig(outdir+'Inpainting Clgg (all cases)')




### As masking grows what is effect on reconstruction




### In the prefactor just use the green curve: inpainting of all masked regions, take power spectrum, that is Clgg
### Don't mask it


### Does inpainting change the shape in a favourable way? Inpainting all masked regions.
### Prefactor ONLY has the non-inpainted spectrum (alex's line)
### Mask convolves in filter. Tau should be modeled cuz delta != delta_e