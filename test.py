import numpy as np
import healpy as hp
from astropy.io import fits
from math import lgamma
from matplotlib import pyplot as plt; plt.switch_backend('agg')

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

print('Reading maps')
input_T353 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_353-psb_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.7255
input_T353_noCMB_COMMANDER = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-commander-353_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
input_T353_thermaldust = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-353_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.7255

mask_planck = hp.reorder(fits.open('data/planck_data_testing/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')[1].data['TMASK'],n2r=True)
mask_GAL060 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL060'],n2r=True)
mask_freq = mask_planck * mask_GAL060.astype(np.float32)

mask_unwise = np.load('data/mask_unWISE_thres_v10.npy')
mask_recon = mask_unwise * mask_planck

fsky = np.where(mask_recon!=0)[0].size / mask_recon.size
fsky_freq = np.where(mask_freq!=0)[0].size / mask_freq.size

input_unWISE = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
N_tot = np.sum(mask_recon)
n_av = N_tot / np.sum(mask_recon)
gdensitymap = (input_unWISE - n_av) / n_av

T353beam = hp.gauss_beam(fwhm=np.radians(4.94/60), lmax=6143)

print('Computing ClTT and Clgg')
ClTT_353 = hp.alm2cl(hp.almxfl(hp.map2alm(input_T353 * mask_freq, lmax=4000), 1/T353beam[:4001])) / fsky_freq
Clgg = hp.anafast(gdensitymap * mask_recon, lmax=4000) / fsky

cltaug_fiducial = np.load('cltaug_fiducial.npy') * 2644.412530692116

ClTT_filter = ClTT_353.copy()[:4001]
Clgg_filter = Clgg.copy()[:4001]
ClTT_filter[:100] = 1e15
Clgg_filter[:100] = 1e15
recon_noise = Noise_vr_diag(lmax=4000, alpha=0, gamma=0, ell=2, cltt=ClTT_353, clgg_binned=Clgg, cltaudg_binned=cltaug_fiducial)

print('Reconstructions and plotting')
Tlm_353 = hp.map2alm(input_T353, lmax=4000)
Tlm_353_noCOMMANDER = hp.map2alm(input_T353_noCMB_COMMANDER, lmax=4000)
Tlm_353_thermaldust = hp.map2alm(input_T353_thermaldust, lmax=4000)

filter_Tlm = lambda Tlm : hp.almxfl(Tlm, np.where(np.isfinite(1/ClTT_filter), 1/ClTT_filter, 0))
Tlm_353_xi = filter_Tlm(Tlm_353)
Tlm_353_noCOMMANDER_xi = filter_Tlm(Tlm_353_noCOMMANDER)
Tlm_353_thermaldust_xi = filter_Tlm(Tlm_353_thermaldust)

dlm = hp.map2alm(gdensitymap, lmax=4000)
dlm_zeta = hp.almxfl(dlm, np.where(np.isfinite(cltaug_fiducial/Clgg_filter), cltaug_fiducial/Clgg_filter, 0))

Tmap_353_filtered = hp.alm2map(Tlm_353_xi, lmax=4000, nside=2048)
Tmap_353_noCOMMANDER_filtered = hp.alm2map(Tlm_353_noCOMMANDER_xi, lmax=4000, nside=2048)
Tmap_353_thermaldust_filtered = hp.alm2map(Tlm_353_thermaldust_xi, lmax=4000, nside=2048)

lssmap_filtered = hp.alm2map(dlm_zeta, lmax=4000, nside=2048)

outmap_353 = hp.ma(-Tmap_353_filtered * lssmap_filtered * recon_noise)
outmap_353_noCOMMANDER = hp.ma(-Tmap_353_noCOMMANDER_filtered * lssmap_filtered * recon_noise)
outmap_353_thermaldust = hp.ma(-Tmap_353_thermaldust_filtered * lssmap_filtered * recon_noise)

outmap_353.mask = np.logical_not(mask_recon)
outmap_353_noCOMMANDER.mask = np.logical_not(mask_recon)
outmap_353_thermaldust.mask = np.logical_not(mask_recon)

recon_353 = hp.remove_dipole(outmap_353)
recon_353_noCOMMANDER = hp.remove_dipole(outmap_353_noCOMMANDER)
recon_353_thermaldust = hp.remove_dipole(outmap_353_thermaldust)

recon_353_Cls = hp.anafast(recon_353, lmax=100) / fsky
recon_353_noCOMMANDER_Cls = hp.anafast(recon_353_noCOMMANDER, lmax=100) / fsky
recon_353_thermaldust_Cls = hp.anafast(recon_353_thermaldust, lmax=100) / fsky

plt.figure()
plt.semilogy(np.arange(101)[2:100], recon_353_Cls[2:100], label='353 GHz')
plt.semilogy(np.arange(101)[2:100], recon_353_noCOMMANDER_Cls[2:100], label='353 GHz (no COMMANDER)')
plt.semilogy(np.arange(101)[2:100], recon_353_thermaldust_Cls[2:100], label='353 GHz (thermal dust)')
plt.semilogy(np.repeat(recon_noise,101)[2:100],ls='--',c='k')
plt.legend()
plt.savefig('test_353_thermal_contribution')
