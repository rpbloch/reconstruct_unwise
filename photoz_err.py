#### Get frequency maps to match theory noise which we take as correct
#### Scale window velocity, your cltaug is just at zbar so it's times that equation thing and W(v) ^2. W(v) is not normalized to 1, normalization is fixed to choice of zbar

import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as tick
from matplotlib import colors as mc
import colorsys
plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})
import numpy as np
import camb
from camb import model
import scipy
from paper_analysis import Estimator, Cosmology
from astropy.io import fits
import healpy as hp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.special import kn
from scipy.special import spherical_jn as jn

# Setup
redshifts = np.linspace(0.0,2.5,100)
zbar_index = 30
spectra_lmax = 4000
ls = np.unique(np.append(np.geomspace(1,spectra_lmax-1,200).astype(int), spectra_lmax-1))

#### File Handling
outdir = 'plots/paper/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# K_CMB units are removed from all temperature maps except frequency maps for 100 GHz.
# Arithmetic operations on that map or any copies thereof generate numerical garbage.
# Instead we carry units of K_CMB through and remove them from the reconstruction itself.
print('Loading maps...')
SMICAinp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725  # Remove K_CMB units
unWISEmap = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
T100inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
T143inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725
T217inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725
Tmap_noCMB_100 = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten()
Tmap_noCMB_143 = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.725
Tmap_noCMB_217 = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.725
thermaldust_100 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-100_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()
thermaldust_143 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-143_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.725
thermaldust_217 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-217_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.725

planckmask = hp.reorder(fits.open('data/planck_data_testing/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')[1].data['TMASK'],n2r=True)
unwise_mask = np.load('data/mask_unWISE_thres_v10.npy')
gauss_unwisemask = np.load('data/gauss_reals/sims_mask_unWISE_reconstructions.npz')
huge_mask = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL020'],n2r=True)
total_mask = unwise_mask * planckmask
hugemask_unwise = huge_mask.astype(np.float32) * unwise_mask * planckmask

####
# Map-based values
fsky = np.where(total_mask!=0)[0].size / total_mask.size
fsky_cltt = np.where(planckmask!=0)[0].size / planckmask.size
fsky_huge = np.where(hugemask_unwise!=0)[0].size / hugemask_unwise.size
ngbar = unWISEmap.sum() / unWISEmap.size  # Needed to scale delta to galaxy

####
# beams
SMICAbeam = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143)
T100beam = hp.gauss_beam(fwhm=np.radians(9.68/60), lmax=6143)
T143beam = hp.gauss_beam(fwhm=np.radians(7.30/60), lmax=6143)
T217beam = hp.gauss_beam(fwhm=np.radians(5.02/60), lmax=6143)
T100beam[spectra_lmax:] = T100beam[spectra_lmax]  # Extremely high beam for 100GHz at high ell ruins the map

###
# Masking / masking + debeaming maps
# Separate step to compute alms for maps which will have power spectra computed to save time
print('Masking/debeaming maps...')
unWISEmap_alms = hp.map2alm(unWISEmap, lmax=spectra_lmax)
SMICAmap_alms = hp.almxfl(hp.map2alm(SMICAinp, lmax=spectra_lmax), 1/SMICAbeam[:spectra_lmax+1])
SMICAmap_real_alms_masked_debeamed = hp.almxfl(hp.map2alm(SMICAinp), 1/SMICAbeam)
T100_alms_masked_debeamed = hp.almxfl(hp.map2alm(T100inp*total_mask), 1/T100beam)
T143_alms_masked_debeamed = hp.almxfl(hp.map2alm(T143inp*total_mask), 1/T143beam)
T217_alms_masked_debeamed = hp.almxfl(hp.map2alm(T217inp*total_mask), 1/T217beam)
T100_alms_masked_debeamed_hugemask = hp.almxfl(hp.map2alm(T100inp*hugemask_unwise), 1/T100beam)
T143_alms_masked_debeamed_hugemask = hp.almxfl(hp.map2alm(T143inp*hugemask_unwise), 1/T143beam)
T217_alms_masked_debeamed_hugemask = hp.almxfl(hp.map2alm(T217inp*hugemask_unwise), 1/T217beam)
Tmap_noCMB_100_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_100*total_mask), 1/T100beam)
Tmap_noCMB_143_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_143*total_mask), 1/T143beam)
Tmap_noCMB_217_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_217*total_mask), 1/T217beam)
Tmap_noCMB_100_masked_debeamed_alms_hugemask = hp.almxfl(hp.map2alm(Tmap_noCMB_100*hugemask_unwise), 1/T100beam)
Tmap_noCMB_143_masked_debeamed_alms_hugemask = hp.almxfl(hp.map2alm(Tmap_noCMB_143*hugemask_unwise), 1/T143beam)
Tmap_noCMB_217_masked_debeamed_alms_hugemask = hp.almxfl(hp.map2alm(Tmap_noCMB_217*hugemask_unwise), 1/T217beam)

SMICAmap_real = hp.alm2map(SMICAmap_real_alms_masked_debeamed, 2048)  # De-beamed unitless SMICA map
T100map_real = hp.alm2map(T100_alms_masked_debeamed, nside=2048)
T143map_real = hp.alm2map(T143_alms_masked_debeamed, nside=2048)
T217map_real = hp.alm2map(T217_alms_masked_debeamed, nside=2048)
T100map_real_hugemask = hp.alm2map(T100_alms_masked_debeamed_hugemask, nside=2048)
T143map_real_hugemask = hp.alm2map(T143_alms_masked_debeamed_hugemask, nside=2048)
T217map_real_hugemask = hp.alm2map(T217_alms_masked_debeamed_hugemask, nside=2048)
Tmap_noCMB_100_masked_debeamed = hp.alm2map(Tmap_noCMB_100_masked_debeamed_alms, 2048)
Tmap_noCMB_143_masked_debeamed = hp.alm2map(Tmap_noCMB_143_masked_debeamed_alms, 2048)
Tmap_noCMB_217_masked_debeamed = hp.alm2map(Tmap_noCMB_217_masked_debeamed_alms, 2048)
Tmap_noCMB_100_masked_debeamed_hugemask = hp.alm2map(Tmap_noCMB_217_masked_debeamed_alms_hugemask, 2048)
Tmap_noCMB_143_masked_debeamed_hugemask = hp.alm2map(Tmap_noCMB_143_masked_debeamed_alms_hugemask, 2048)
Tmap_noCMB_217_masked_debeamed_hugemask = hp.alm2map(Tmap_noCMB_217_masked_debeamed_alms_hugemask, 2048)
Tmap_thermaldust_100 = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_100*total_mask), 1/T100beam), 2048)
Tmap_thermaldust_143 = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_143*total_mask), 1/T143beam), 2048)
Tmap_thermaldust_217 = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_217*total_mask), 1/T217beam), 2048)
Tmap_thermaldust_100_hugemask = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_100*hugemask_unwise), 1/T100beam), 2048)
Tmap_thermaldust_143_hugemask = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_143*hugemask_unwise), 1/T143beam), 2048)
Tmap_thermaldust_217_hugemask = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_217*hugemask_unwise), 1/T217beam), 2048)

# Power spectra
ClTT = hp.alm2cl(hp.almxfl(hp.map2alm(SMICAinp*planckmask), 1/SMICAbeam)) / fsky_cltt
ClTT_100 = hp.alm2cl(T100_alms_masked_debeamed) / fsky
ClTT_143 = hp.alm2cl(T143_alms_masked_debeamed) / fsky
ClTT_217 = hp.alm2cl(T217_alms_masked_debeamed) / fsky
ClTT_100_huge = hp.alm2cl(T100_alms_masked_debeamed_hugemask) / fsky_huge
ClTT_143_huge = hp.alm2cl(T143_alms_masked_debeamed_hugemask) / fsky_huge
ClTT_217_huge = hp.alm2cl(T217_alms_masked_debeamed_hugemask) / fsky_huge
ClTT_100foregrounds = hp.alm2cl(Tmap_noCMB_100_masked_debeamed_alms) / fsky
ClTT_143foregrounds = hp.alm2cl(Tmap_noCMB_143_masked_debeamed_alms) / fsky
ClTT_217foregrounds = hp.alm2cl(Tmap_noCMB_217_masked_debeamed_alms) / fsky
# Zero Cls above our lmax
ClTT[ls.max()+1:] = 0.  
ClTT_100[ls.max()+1:] = 0.
ClTT_143[ls.max()+1:] = 0.
ClTT_217[ls.max()+1:] = 0.
ClTT_100_huge[ls.max()+1:] = 0.
ClTT_143_huge[ls.max()+1:] = 0.
ClTT_217_huge[ls.max()+1:] = 0.
ClTT_100foregrounds[ls.max()+1:] = 0.
ClTT_143foregrounds[ls.max()+1:] = 0.
ClTT_217foregrounds[ls.max()+1:] = 0.

### Cosmology
print('Setting up cosmology')
nks = 2000
ks = np.logspace(-4,1,nks)
# Set up the growth function by computing ratios of linear matter power spectra 
# and compute the radial comoving distance functions
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(redshifts=redshifts, kmax=2.0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)

### Spectra calculations and estimator
# Ensure csm object has same cosmology as what we used in our velocity calculation earlier
chis = results.comoving_radial_distance(redshifts)
csm = Cosmology(nbin=1,zmin=redshifts.min(), zmax=redshifts.max(), redshifts=redshifts, ks=ks, zerrs=True)  # Set up cosmology
estim = Estimator(nbin=1)  # Set up estimator
csm.cambpars = pars
csm.cosmology_data = camb.get_background(pars)
csm.bin_width = chis[-1] - chis[0]
fullspectrum_ls = np.unique(np.append(np.geomspace(1,6144-1,200).astype(int), 6144-1))

### realization-independent parameters
Pmms = np.zeros((chis.size,fullspectrum_ls.size))
Pmm_full = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=100., zmax=redshifts.max())
for l, ell in enumerate(fullspectrum_ls):  # Do limber approximation: P(z,k) -> P(z, (ell+0.5)/chi )
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))

Pem_bin1_chi = np.zeros((fullspectrum_ls.size))
for l, ell in enumerate(fullspectrum_ls):
	Pem_bin1_chi[l] = Pmms[zbar_index,l] * np.diagonal(np.flip(csm.bias_e2(csm.chi_to_z(chis), (ell+0.5)/chis[::-1]), axis=1))[zbar_index]  # Convert Pmm to Pem

taud1_window  = csm.get_limber_window('taud', chis, avg=False)[zbar_index]  # Units of 1/Mpc
h = results.h_of_z(redshifts)
chibar = csm.z_to_chi(redshifts[zbar_index])

print('Computing true velocity...')
velocity_compute_ells = np.append(np.unique(np.geomspace(1,30,10).astype(int)),100)
clv = np.zeros((velocity_compute_ells.shape[0],redshifts.shape[0],redshifts.shape[0]))
PKv = camb.get_matter_power_interpolator(pars,hubble_units=False, k_hunit=False, var1='v_newtonian_cdm',var2='v_newtonian_cdm')
for l in range(velocity_compute_ells.shape[0]):
	print('    @ l = %d' % velocity_compute_ells[l])
	for z1 in range(redshifts.shape[0]):
		for z2 in range(redshifts.shape[0]):
			integrand_k = scipy.special.spherical_jn(velocity_compute_ells[l],ks*chis[z1])*scipy.special.spherical_jn(velocity_compute_ells[l],ks*chis[z2]) * (h[z1]/(1+redshifts[z1]))*(h[z2]/(1+redshifts[z2])) * np.sqrt(PKv.P(redshifts[z1],ks)*PKv.P(redshifts[z2],ks))
			clv[l,z1,z2] = (2./np.pi)*np.trapz(integrand_k,ks)

# Compute inital reconstruction to get binning scheme to avoid saving full maps
print('Beginning reconstructions')
print('    Primary reconstruction: Planck x unWISE')
lowpass = lambda MAP : hp.alm2map(hp.almxfl(hp.map2alm(MAP), [0 if l > 20 else 1 for l in np.arange(6144)]), 2048)
csm.compute_Cls(ngbar=ngbar,gtag='g')
galaxy_window_binned = csm.get_limber_window('g', chis, avg=False)[zbar_index]  # Units of 1/Mpc
Cltaug_at_zbar = interp1d(fullspectrum_ls, (Pem_bin1_chi * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
noise_recon = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_SMICA = estim.combine(SMICAmap_real, unWISEmap, total_mask, ClTT, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon,6144), convert_K=False)
lowpass_output = lowpass(outmap_SMICA)
if not os.path.exists('data/unWISE/blue_window_alex_spectra_reconstructions.npz'):
	nbin_hist = 50
else:
	dataload = np.load('data/unWISE/blue_window_alex_spectra_reconstructions.npz')
	nbin_hist = dataload['nbin_hist']

n, bins = np.histogram(lowpass_output[np.where(total_mask!=0)], bins=nbin_hist)
print('    Secondary reconstruction: 100GHz x unWISE')
noise_100 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_100.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_100 = estim.combine(T100map_real, unWISEmap, total_mask, ClTT_100, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_100foregrounds = estim.combine(Tmap_noCMB_100_masked_debeamed, unWISEmap, total_mask, ClTT_100, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_thermaldust_100 = estim.combine(Tmap_thermaldust_100, unWISEmap, total_mask, ClTT_100, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
recon_Cls_100 = hp.anafast(outmap_100)
recon_Cls_100_foregrounds = hp.anafast(outmap_100foregrounds)
recon_Cls_100_thermaldust = hp.anafast(outmap_thermaldust_100)
lowpass_output_100 = lowpass(outmap_100)
lowpass_output_100foregrounds = lowpass(outmap_100foregrounds)
lowpass_output_thermaldust_100 = lowpass(outmap_thermaldust_100)
n_100, bins_100 = np.histogram(lowpass_output_100[np.where(total_mask!=0)], bins=nbin_hist)
n_100_foregrounds, bins_100_foregrounds = np.histogram(lowpass_output_100foregrounds[np.where(total_mask!=0)], bins=nbin_hist)
n_100_thermaldust, bins_100_thermaldust = np.histogram(lowpass_output_thermaldust_100[np.where(total_mask!=0)], bins=nbin_hist)

print('    Secondary reconstruction: 143GHz x unWISE')
noise_143 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_143.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_143 = estim.combine(T143map_real, unWISEmap, total_mask, ClTT_143, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_143foregrounds = estim.combine(Tmap_noCMB_143_masked_debeamed, unWISEmap, total_mask, ClTT_143, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_thermaldust_143 = estim.combine(Tmap_thermaldust_143, unWISEmap, total_mask, ClTT_143, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
recon_Cls_143 = hp.anafast(outmap_143)
recon_Cls_143_foregrounds = hp.anafast(outmap_143foregrounds)
recon_Cls_143_thermaldust = hp.anafast(outmap_thermaldust_143)
lowpass_output_143 = lowpass(outmap_143)
lowpass_output_143foregrounds = lowpass(outmap_143foregrounds)
lowpass_output_thermaldust_143 = lowpass(outmap_thermaldust_143)
n_143, bins_143 = np.histogram(lowpass_output_143[np.where(total_mask!=0)], bins=nbin_hist)
n_143_foregrounds, bins_143_foregrounds = np.histogram(lowpass_output_143foregrounds[np.where(total_mask!=0)], bins=nbin_hist)
n_143_thermaldust, bins_143_thermaldust = np.histogram(lowpass_output_thermaldust_143[np.where(total_mask!=0)], bins=nbin_hist)

print('    Secondary reconstruction: 217GHz x unWISE')
noise_217 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_217 = estim.combine(T217map_real, unWISEmap, total_mask, ClTT_217, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_217foregrounds = estim.combine(Tmap_noCMB_217_masked_debeamed, unWISEmap, total_mask, ClTT_217, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_thermaldust_217 = estim.combine(Tmap_thermaldust_217, unWISEmap, total_mask, ClTT_217, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
recon_Cls_217 = hp.anafast(outmap_217)
recon_Cls_217_foregrounds = hp.anafast(outmap_217foregrounds)
recon_Cls_217_thermaldust = hp.anafast(outmap_thermaldust_217)
lowpass_output_217 = lowpass(outmap_217)
lowpass_output_217foregrounds = lowpass(outmap_217foregrounds)
lowpass_output_thermaldust_217 = lowpass(outmap_thermaldust_217)
n_217, bins_217 = np.histogram(lowpass_output_217[np.where(total_mask!=0)], bins=nbin_hist)
n_217_foregrounds, bins_217_foregrounds = np.histogram(lowpass_output_217foregrounds[np.where(total_mask!=0)], bins=nbin_hist)
n_217_thermaldust, bins_217_thermaldust = np.histogram(lowpass_output_thermaldust_217[np.where(total_mask!=0)], bins=nbin_hist)

print('    Secondary reconstruction: 100GHz x unWISE with huge mask')
noise_100_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_100_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_100_huge = estim.combine(T100map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_100_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_100_huge,6144), convert_K=True)
_, _, outmap_100foregrounds_huge = estim.combine(Tmap_noCMB_100_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_100_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_100_huge,6144), convert_K=True)
_, _, outmap_thermaldust_100_huge = estim.combine(Tmap_thermaldust_100_hugemask, unWISEmap, hugemask_unwise, ClTT_100_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_100_huge,6144), convert_K=True)
recon_Cls_100_huge = hp.anafast(outmap_100_huge)
recon_Cls_100_foregrounds_huge = hp.anafast(outmap_100foregrounds_huge)
recon_Cls_100_thermaldust_huge = hp.anafast(outmap_thermaldust_100_huge)
lowpass_output_100_huge = lowpass(outmap_100_huge)
lowpass_output_100foregrounds_huge = lowpass(outmap_100foregrounds_huge)
lowpass_output_thermaldust_100_huge = lowpass(outmap_thermaldust_100_huge)
n_100_huge, bins_100_huge = np.histogram(lowpass_output_100_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_100_foregrounds_huge, bins_100_foregrounds_huge = np.histogram(lowpass_output_100foregrounds_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_100_thermaldust_huge, bins_100_thermaldust_huge = np.histogram(lowpass_output_thermaldust_100_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)

print('    Secondary reconstruction: 143GHz x unWISE with huge mask')
noise_143_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_143_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_143_huge = estim.combine(T143map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_143_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_143_huge,6144), convert_K=False)
_, _, outmap_143foregrounds_huge = estim.combine(Tmap_noCMB_143_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_143_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_143_huge,6144), convert_K=False)
_, _, outmap_thermaldust_143_huge = estim.combine(Tmap_thermaldust_143_hugemask, unWISEmap, hugemask_unwise, ClTT_143_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_143_huge,6144), convert_K=False)
recon_Cls_143_huge = hp.anafast(outmap_143_huge)
recon_Cls_143_foregrounds_huge = hp.anafast(outmap_143foregrounds_huge)
recon_Cls_143_thermaldust_huge = hp.anafast(outmap_thermaldust_143_huge)
lowpass_output_143_huge = lowpass(outmap_143_huge)
lowpass_output_143foregrounds_huge = lowpass(outmap_143foregrounds_huge)
lowpass_output_thermaldust_143_huge = lowpass(outmap_thermaldust_143_huge)
n_143_huge, bins_143_huge = np.histogram(lowpass_output_143_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_143_foregrounds_huge, bins_143_foregrounds_huge = np.histogram(lowpass_output_143foregrounds_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_143_thermaldust_huge, bins_143_thermaldust_huge = np.histogram(lowpass_output_thermaldust_143_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)

print('    Secondary reconstruction: 217GHz x unWISE with huge mask')
noise_217_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_217_huge = estim.combine(T217map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_217_huge,6144), convert_K=False)
_, _, outmap_217foregrounds_huge = estim.combine(Tmap_noCMB_217_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_217_huge,6144), convert_K=False)
_, _, outmap_thermaldust_217_huge = estim.combine(Tmap_thermaldust_217_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_217_huge,6144), convert_K=False)
recon_Cls_217_huge = hp.anafast(outmap_217_huge)
recon_Cls_217_foregrounds_huge = hp.anafast(outmap_217foregrounds_huge)
recon_Cls_217_thermaldust_huge = hp.anafast(outmap_thermaldust_217_huge)
lowpass_output_217_huge = lowpass(outmap_217_huge)
lowpass_output_217foregrounds_huge = lowpass(outmap_217foregrounds_huge)
lowpass_output_thermaldust_217_huge = lowpass(outmap_thermaldust_217_huge)
n_217_huge, bins_217_huge = np.histogram(lowpass_output_217_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_217_foregrounds_huge, bins_217_foregrounds_huge = np.histogram(lowpass_output_217foregrounds_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_217_thermaldust_huge, bins_217_thermaldust_huge = np.histogram(lowpass_output_thermaldust_217_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)

# Load or compute spectra for each realization of dN/dz
if not os.path.exists('data/unWISE/blue_window_alex_spectra_result_full_AB.npz'):
	print('Cached dN/dz realizations not found, computing power spectra...')
	Clgg_store = np.zeros((100, 6144))
	Cltaug_store = np.zeros((100, 6144))
	Clv_windowed_store = np.zeros((100, velocity_compute_ells.size))
	Clv_windowed_mm_me_store = np.zeros((100,velocity_compute_ells.size))
	for dndz_iter_ind in np.arange(100):
		print('    dN/dz iteration %d' % (dndz_iter_ind+1))
		csm.compute_Cls(ngbar=ngbar, gtag='gerr_alex_%d.txt' % dndz_iter_ind)  # These are the integrated Cls over the entire bin
		Clgg_store[dndz_iter_ind] = csm.Clgg[0,0,:].copy()
		galaxy_window_binned = csm.get_limber_window('gerr_alex_%d.txt' % dndz_iter_ind, chis, avg=False)[zbar_index]  # Units of 1/Mpc
		Cltaug_at_zbar = interp1d(fullspectrum_ls, (Pem_bin1_chi * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
		Cltaug_store[dndz_iter_ind] = Cltaug_at_zbar.copy()
		######## VELOCITY
		window_g = csm.get_limber_window('gerr_alex_%d.txt' % dndz_iter_ind, chis, avg=False)
		ell_const = 2
		terms = 0
		terms_with_me_entry = np.zeros(chis.size)
		terms_with_mm_me_entry = np.zeros(chis.size)
		for l2 in np.arange(spectra_lmax):
			Pmm_at_ellprime = np.diagonal(np.flip(Pmm_full.P(csm.cosmology_data.redshift_at_comoving_radial_distance(chis), (l2+0.5)/chis[::-1], grid=True), axis=1))
			Pem_at_ellprime = Pmm_at_ellprime * np.diagonal(np.flip(csm.bias_e2(csm.chi_to_z(chis), (l2+0.5)/chis[::-1]), axis=1))  # Convert Pmm to Pem
			Pem_at_ellprime_at_zbar = Pem_at_ellprime[zbar_index]
			for l1 in np.arange(np.abs(l2-ell_const),l2+ell_const+1):
				if l1 > spectra_lmax-1 or l1 <2:   #triangle rule
					continue
				gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell_const+1)/(4*np.pi))*estim.wigner_symbol(ell_const, l1, l2)*Cltaug_at_zbar[l2]
				term_with_me_entry = (gamma_ksz*gamma_ksz/(ClTT[l1]*csm.Clgg[0,0,:][l2])) * (Pem_at_ellprime/Pem_at_ellprime_at_zbar)
				term_with_mm_me_entry = (gamma_ksz*gamma_ksz/(ClTT[l1]*csm.Clgg[0,0,:][l2])) * (Pmm_at_ellprime/Pem_at_ellprime_at_zbar)
				term_entry = (gamma_ksz*gamma_ksz/(ClTT[l1]*csm.Clgg[0,0,:][l2]))
				if np.isfinite(term_entry):
					terms += term_entry
					terms_with_me_entry += term_with_me_entry
					terms_with_mm_me_entry += term_with_mm_me_entry
		ratio_me_me = terms_with_me_entry / terms
		ratio_mm_me = terms_with_mm_me_entry / terms
		window_v_chi1 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
		window_v_chi2 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
		window_v_mm_me_chi1 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )
		window_v_mm_me_chi2 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )
		clv_windowed = np.zeros(velocity_compute_ells.size)
		clv_windowed_mm_me = np.zeros(velocity_compute_ells.size)
		for i in np.arange(velocity_compute_ells.size):
			clv_windowed[i] = np.trapz(window_v_chi1*np.trapz(window_v_chi2*clv[i,:,:], chis,axis=1), chis)
			clv_windowed_mm_me[i] = np.trapz(window_v_mm_me_chi1*np.trapz(window_v_mm_me_chi2*clv[i,:,:], chis,axis=1), chis)
		Clv_windowed_store[dndz_iter_ind] = clv_windowed.copy()
		Clv_windowed_mm_me_store[dndz_iter_ind] = clv_windowed_mm_me.copy()
	np.savez('data/unWISE/blue_window_alex_spectra_result_full_AB.npz',Clgg_store=Clgg_store,Cltaug_store=Cltaug_store,Clv_windowed_store=Clv_windowed_store,Clv_windowed_mm_me_store=Clv_windowed_mm_me_store)
else:
	print('Cached dN/dz realizations found')
	dataload = np.load('data/unWISE/blue_window_alex_spectra_result_full_AB.npz')
	Clgg_store = dataload['Clgg_store']
	Cltaug_store = dataload['Cltaug_store']
	Clv_windowed_store = dataload['Clv_windowed_store']
	Clv_windowed_mm_me_store = dataload['Clv_windowed_mm_me_store']

#########################
# Load or compute reconstructions for each realization of dN/dz
if not os.path.exists('data/unWISE/blue_window_alex_spectra_reconstructions_full_AB.npz'):
	print('Cached dN/dz reconstructions not found, computing reconstructions...')
	nbin_hist = 50
	noises_dndz = np.zeros((100,6144))
	recon_Cls_dndz = np.zeros((100,6144))
	n_dndz = np.zeros((100,nbin_hist))
	bins_dndz = np.zeros((100,nbin_hist+1))
	for dndz_iter_ind in np.arange(100):
		print("    dN/dz iteration %d" % (dndz_iter_ind+1))
		noises_dndz[dndz_iter_ind] = np.repeat(estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=Clgg_store[dndz_iter_ind].copy(), cltaudg_binned=Cltaug_store[dndz_iter_ind].copy()), 6144)
		_, _, outmap_SMICA = estim.combine_alm(SMICAmap_alms, unWISEmap_alms, total_mask,  ClTT, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz[dndz_iter_ind], lmax=spectra_lmax, convert_K=False)
		recon_Cls_dndz[dndz_iter_ind] = interp1d(np.arange(101), hp.anafast(outmap_SMICA, lmax=100), bounds_error=False, fill_value=0.)(np.arange(6144))
		lowpass_output = lowpass(outmap_SMICA)
		n_dndz[dndz_iter_ind], bins_dndz[dndz_iter_ind] = np.histogram(lowpass_output[np.where(total_mask!=0)], bins=bins)
	np.savez('data/unWISE/blue_window_alex_spectra_reconstructions_full_AB.npz',
		nbin_hist=nbin_hist,
		recon_Cls_dndz=recon_Cls_dndz,
		n_dndz=n_dndz,bins_dndz=bins_dndz,
		noises_dndz=noises_dndz)
else:
	print('Cached dN/dz reconstructions found')
	dataload = np.load('data/unWISE/blue_window_alex_spectra_reconstructions_full_AB.npz')
	nbin_hist = dataload['nbin_hist']
	recon_Cls_dndz = dataload['recon_Cls_dndz']
	n_dndz = dataload['n_dndz']
	bins_dndz = dataload['bins_dndz']
	noises_dndz = dataload['noises_dndz']

# Load or compute frequency reconstructions for each realization of dN/dz
for frequency in [100, 143, 217]:
	if not os.path.exists('data/unWISE/blue_window_alex_spectra_reconstructions_%d.npz' % frequency):
		print('Cached dN/dz reconstructions not found for %d GHz, computing reconstructions...' % frequency)
		if frequency == 100:
			convert_units = True
		else:
			convert_units = False
		ClTT_freq = {100 : ClTT_100, 143 : ClTT_143, 217 : ClTT_217}[frequency]
		bins_freq = {100 : bins_100, 143 : bins_143, 217 : bins_217}[frequency]
		bins_freq_foregrounds = {100 : bins_100_foregrounds, 143 : bins_143_foregrounds, 217 : bins_217_foregrounds}[frequency]
		bins_freq_thermaldust = {100 : bins_100_thermaldust, 143 : bins_143_thermaldust, 217 : bins_217_thermaldust}[frequency]
		Tmap_freq = {100 : T100map_real, 143: T143map_real, 217 : T217map_real}[frequency]
		Tmap_freq_foregrounds = {100 : Tmap_noCMB_100_masked_debeamed, 143 : Tmap_noCMB_143_masked_debeamed, 217 : Tmap_noCMB_217_masked_debeamed}[frequency]
		Tmap_freq_thermaldust = {100 : Tmap_thermaldust_100, 143 : Tmap_thermaldust_143, 217 : Tmap_thermaldust_217}[frequency]
		noises_dndz_freq = np.zeros((100,6144))
		recon_Cls_dndz_freq = np.zeros((100,6144))
		recon_Cls_dndz_freq_foregrounds = np.zeros((100,6144))
		recon_Cls_dndz_freq_thermaldust = np.zeros((100,6144))
		n_dndz_freq = np.zeros((100,nbin_hist))
		n_dndz_freq_foregrounds = np.zeros((100,nbin_hist))
		n_dndz_freq_thermaldust = np.zeros((100,nbin_hist))
		for dndz_iter_ind in np.arange(100):
			print("    dN/dz iteration %d" % (dndz_iter_ind+1))
			noises_dndz_freq[dndz_iter_ind] = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_freq.copy(), clgg_binned=Clgg_store[dndz_iter_ind].copy(), cltaudg_binned=Cltaug_store[dndz_iter_ind].copy())
			_, _, outmap_freq = estim.combine(Tmap_freq, unWISEmap, total_mask, ClTT_freq, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz_freq[dndz_iter_ind], convert_K=convert_units)
			_, _, outmap_freq_foregrounds = estim.combine(Tmap_freq_foregrounds, unWISEmap, total_mask, ClTT_freq, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz_freq[dndz_iter_ind], convert_K=convert_units)
			_, _, outmap_thermaldust_freq = estim.combine(Tmap_freq_thermaldust, unWISEmap, total_mask, ClTT_freq, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz_freq[dndz_iter_ind], convert_K=convert_units)
			recon_Cls_dndz_freq[dndz_iter_ind] = hp.anafast(outmap_freq)
			recon_Cls_dndz_freq_foregrounds[dndz_iter_ind] = hp.anafast(outmap_freq_foregrounds)
			recon_Cls_dndz_freq_thermaldust[dndz_iter_ind] = hp.anafast(outmap_thermaldust_freq)
			lowpass_output_freq = lowpass(outmap_freq)
			lowpass_output_freq_foregrounds = lowpass(outmap_freq_foregrounds)
			lowpass_output_freq_thermaldust = lowpass(outmap_thermaldust_freq)
			n_dndz_freq[dndz_iter_ind], _ = np.histogram(lowpass_output_freq[np.where(total_mask!=0)], bins=bins_freq)
			n_dndz_freq_foregrounds[dndz_iter_ind], _ = np.histogram(lowpass_output_freq_foregrounds[np.where(total_mask!=0)], bins=bins_freq_foregrounds)
			n_dndz_freq_thermaldust[dndz_iter_ind], _ = np.histogram(lowpass_output_freq_thermaldust[np.where(total_mask!=0)], bins=bins_freq_thermaldust)
		np.savez('data/unWISE/blue_window_alex_spectra_reconstructions_%d.npz' % frequency,recon_Cls_dndz=recon_Cls_dndz_freq,recon_Cls_dndz_noCMB=recon_Cls_dndz_freq_foregrounds,recon_Cls_dndz_thermaldust=recon_Cls_dndz_freq_thermaldust,n_dndz=n_dndz_freq,n_dndz_noCMB=n_dndz_freq_foregrounds,n_dndz_thermaldust=n_dndz_freq_thermaldust,bins_dndz=bins_freq, bins_dndz_noCMB=bins_freq_foregrounds, bins_dndz_thermaldust=bins_freq_thermaldust)
	else:
		print('Cached dN/dz reconstructions found for %d GHz' % frequency)

frequency = 217
if not os.path.exists('data/unWISE/blue_window_alex_spectra_reconstructions_%d_hugemask.npz' % frequency):
	print('Cached dN/dz reconstructions not found for %d GHz with huge mask, computing reconstructions...' % frequency)
	noises_dndz_217_huge = np.zeros((100,6144))
	recon_Cls_dndz_217_huge = np.zeros((100,6144))
	recon_Cls_dndz_217_huge_foregrounds = np.zeros((100,6144))
	recon_Cls_dndz_217_huge_thermaldust = np.zeros((100,6144))
	n_dndz_217_huge = np.zeros((100,nbin_hist))
	n_dndz_217_huge_foregrounds = np.zeros((100,nbin_hist))
	n_dndz_217_huge_thermaldust = np.zeros((100,nbin_hist))
	for dndz_iter_ind in np.arange(100):
		print("    dN/dz iteration %d" % (dndz_iter_ind+1))
		noises_dndz_217_huge[dndz_iter_ind] = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217_huge.copy(), clgg_binned=Clgg_store[dndz_iter_ind].copy(), cltaudg_binned=Cltaug_store[dndz_iter_ind].copy())
		_, _, outmap_freq = estim.combine(T217map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz_217_huge[dndz_iter_ind], convert_K=False)
		_, _, outmap_freq_foregrounds = estim.combine(Tmap_noCMB_217_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz_217_huge[dndz_iter_ind], convert_K=False)
		_, _, outmap_thermaldust_freq = estim.combine(Tmap_thermaldust_217_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, Clgg_store[dndz_iter_ind].copy(), Cltaug_store[dndz_iter_ind].copy(), noises_dndz_217_huge[dndz_iter_ind], convert_K=False)
		recon_Cls_dndz_217_huge[dndz_iter_ind] = hp.anafast(outmap_freq)
		recon_Cls_dndz_217_huge_foregrounds[dndz_iter_ind] = hp.anafast(outmap_freq_foregrounds)
		recon_Cls_dndz_217_huge_thermaldust[dndz_iter_ind] = hp.anafast(outmap_thermaldust_freq)
		lowpass_output_freq = lowpass(outmap_freq)
		lowpass_output_freq_foregrounds = lowpass(outmap_freq_foregrounds)
		lowpass_output_freq_thermaldust = lowpass(outmap_thermaldust_freq)
		n_dndz_217_huge[dndz_iter_ind], _ = np.histogram(lowpass_output_freq[np.where(hugemask_unwise!=0)], bins=bins_217_huge)
		n_dndz_217_huge_foregrounds[dndz_iter_ind], _ = np.histogram(lowpass_output_freq_foregrounds[np.where(hugemask_unwise!=0)], bins=bins_217_foregrounds_huge)
		n_dndz_217_huge_thermaldust[dndz_iter_ind], _ = np.histogram(lowpass_output_freq_thermaldust[np.where(hugemask_unwise!=0)], bins=bins_217_thermaldust_huge)
	np.savez('data/unWISE/blue_window_alex_spectra_reconstructions_%d_hugemask.npz' % frequency,recon_Cls_dndz_217_huge=recon_Cls_dndz_217_huge,recon_Cls_dndz_217_huge_foregrounds=recon_Cls_dndz_217_huge_foregrounds,recon_Cls_dndz_217_huge_thermaldust=recon_Cls_dndz_217_huge_thermaldust,n_dndz_217_huge=n_dndz_217_huge,n_dndz_217_huge_foregrounds=n_dndz_217_huge_foregrounds,n_dndz_217_huge_thermaldust=n_dndz_217_huge_thermaldust,bins_217_huge=bins_217_huge,bins_217_foregrounds_huge=bins_217_foregrounds_huge,bins_217_thermaldust_huge=bins_217_thermaldust_huge)
else:
	print('Cached dN/dz reconstructions found for %d GHz with huge mask' % frequency)

recon_Cls_dndz_freq = {100 : {}, 143 : {}, 217 : {}}
n_dndz_freq = {100 : {}, 143 : {}, 217 : {}}
bins_dndz_freq = {100 : {}, 143 : {}, 217 : {}}
for frequency in [100, 143, 217]:
	dataload = np.load('data/unWISE/blue_window_alex_spectra_reconstructions_%d.npz' % frequency)
	recon_Cls_dndz_freq[frequency]['full'] = dataload['recon_Cls_dndz']
	recon_Cls_dndz_freq[frequency]['noCMB'] = dataload['recon_Cls_dndz_noCMB']
	recon_Cls_dndz_freq[frequency]['thermaldust'] = dataload['recon_Cls_dndz_thermaldust']
	n_dndz_freq[frequency]['full'] = dataload['n_dndz']
	n_dndz_freq[frequency]['noCMB'] = dataload['n_dndz_noCMB']
	n_dndz_freq[frequency]['thermaldust'] = dataload['n_dndz_thermaldust']
	bins_dndz_freq[frequency]['full'] = dataload['bins_dndz']
	bins_dndz_freq[frequency]['noCMB'] = dataload['bins_dndz_noCMB']
	bins_dndz_freq[frequency]['thermaldust'] = dataload['bins_dndz_thermaldust']

dataload = np.load('data/unWISE/blue_window_alex_spectra_reconstructions_217_hugemask.npz')
recon_Cls_dndz_217_huge = dataload['recon_Cls_dndz_217_huge']
recon_Cls_dndz_217_huge_foregrounds = dataload['recon_Cls_dndz_217_huge_foregrounds']
recon_Cls_dndz_217_huge_thermaldust = dataload['recon_Cls_dndz_217_huge_thermaldust']

#### Plots and recompute at Clgg, cltaug for main values
csm.compute_Cls(ngbar=ngbar,gtag='g')
galaxy_window_binned = csm.get_limber_window('g', chis, avg=False)[zbar_index]  # Units of 1/Mpc
Cltaug_at_zbar = interp1d(fullspectrum_ls, (Pem_bin1_chi * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
window_g = csm.get_limber_window('g', chis, avg=False)
ell_const = 2
print('evaluating L = %d' % ell_const)
terms = 0
terms_with_me_entry = np.zeros(chis.size)
terms_with_mm_me_entry = np.zeros(chis.size)
for l2 in np.arange(spectra_lmax):
	Pmm_at_ellprime = np.diagonal(np.flip(Pmm_full.P(csm.cosmology_data.redshift_at_comoving_radial_distance(chis), (l2+0.5)/chis[::-1], grid=True), axis=1))
	Pem_at_ellprime = Pmm_at_ellprime * np.diagonal(np.flip(csm.bias_e2(csm.chi_to_z(chis), (l2+0.5)/chis[::-1]), axis=1))  # Convert Pmm to Pem
	Pem_at_ellprime_at_zbar = Pem_at_ellprime[zbar_index]
	for l1 in np.arange(np.abs(l2-ell_const),l2+ell_const+1):
		if l1 > spectra_lmax-1 or l1 <2:   #triangle rule
			continue
		gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell_const+1)/(4*np.pi))*estim.wigner_symbol(ell_const, l1, l2)*Cltaug_at_zbar[l2]
		term_with_me_entry = (gamma_ksz*gamma_ksz/(ClTT[l1]*csm.Clgg[0,0,:][l2])) * (Pem_at_ellprime/Pem_at_ellprime_at_zbar)
		term_with_mm_me_entry = (gamma_ksz*gamma_ksz/(ClTT[l1]*csm.Clgg[0,0,:][l2])) * (Pmm_at_ellprime/Pem_at_ellprime_at_zbar)
		term_entry = (gamma_ksz*gamma_ksz/(ClTT[l1]*csm.Clgg[0,0,:][l2]))
		if np.isfinite(term_entry):
			terms += term_entry
			terms_with_me_entry += term_with_me_entry
			terms_with_mm_me_entry += term_with_mm_me_entry

ratio_me_me = terms_with_me_entry / terms
ratio_mm_me = terms_with_mm_me_entry / terms
window_v_chi1 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
window_v_chi2 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
window_v_mm_me_chi1 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )
window_v_mm_me_chi2 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )
clv_windowed = np.zeros(velocity_compute_ells.size)
clv_windowed_mm_me = np.zeros(velocity_compute_ells.size)
for i in np.arange(velocity_compute_ells.size):
	clv_windowed[i] = np.trapz(window_v_chi1*np.trapz(window_v_chi2*clv[i,:,:], chis,axis=1), chis)
	clv_windowed_mm_me[i] = np.trapz(window_v_mm_me_chi1*np.trapz(window_v_mm_me_chi2*clv[i,:,:], chis,axis=1), chis)

print("Reconstructing for actual case")
noise_recon = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=hp.alm2cl(unWISEmap_alms), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_SMICA = estim.combine(SMICAmap_real, unWISEmap, total_mask, ClTT, np.append(hp.alm2cl(unWISEmap_alms),np.zeros(6144-hp.alm2cl(unWISEmap_alms).size)), Cltaug_at_zbar.copy(), np.repeat(noise_recon,6144), convert_K=False)
recon_Cls = hp.anafast(outmap_SMICA)

####
# Test to see what Gaussian reconstruction + signal V realization looks like
if False:
	print('Computing velocity + noise for dndz reals')	
	draw_Cl_SMICA = hp.anafast(SMICAinp* total_mask, lmax=spectra_lmax) / fsky
	draw_Cl_T100  = hp.anafast(T100inp * total_mask, lmax=spectra_lmax) / fsky
	draw_Cl_T143  = hp.anafast(T143inp * total_mask, lmax=spectra_lmax) / fsky
	draw_Cl_T217  = hp.anafast(T217inp * total_mask, lmax=spectra_lmax) / fsky
	draw_Cl_T217huge = hp.anafast(T217inp * hugemask_unwise, lmax=spectra_lmax) / fsky_huge
	Cltaug_at_zbar_mm_me = interp1d(fullspectrum_ls, (Pmms[zbar_index,:] * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
	clv_windowed_interp = interp1d(velocity_compute_ells,clv_windowed,fill_value=0.,bounds_error=False)(np.arange(6144))
	clv_windowed_interp_mm_me = interp1d(velocity_compute_ells,clv_windowed_mm_me,fill_value=0.,bounds_error=False)(np.arange(6144))
	noise_SMICA_mm_me = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_mm_me.copy())
	noise_T100_mm_me = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_100.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_mm_me.copy())
	noise_T143_mm_me = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_143.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_mm_me.copy())
	noise_T217_mm_me = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_mm_me.copy())
	noise_T217huge_mm_me = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_mm_me.copy())	
	for big_i in np.arange(1):
		velocity_noise_Cls = []
		velocity_noise_T100_Cls = []
		velocity_noise_T143_Cls = []
		velocity_noise_T217_Cls = []
		velocity_noise_T217huge_Cls = []
		velocity_noise_Cls_mm_me = []
		velocity_noise_T100_Cls_mm_me = []
		velocity_noise_T143_Cls_mm_me = []
		velocity_noise_T217_Cls_mm_me = []
		velocity_noise_T217huge_Cls_mm_me = []
		for i in np.arange(20):
			print('big_i = %d, i=%d'%(big_i, i), end='   ')
			SMICA_gauss_realization = hp.almxfl(hp.synalm(draw_Cl_SMICA, lmax=spectra_lmax), 1/SMICAbeam[:spectra_lmax+1])
			#T100_gauss_realization = hp.almxfl(hp.synalm(draw_Cl_T100, lmax=spectra_lmax), 1/T100beam[:spectra_lmax+1])
			#T143_gauss_realization = hp.almxfl(hp.synalm(draw_Cl_T143, lmax=spectra_lmax), 1/T143beam[:spectra_lmax+1])
			#T217_gauss_realization = hp.almxfl(hp.synalm(draw_Cl_T217, lmax=spectra_lmax), 1/T217beam[:spectra_lmax+1])
			#T217_huge_gauss_realization = hp.almxfl(hp.synalm(draw_Cl_T217huge, lmax=spectra_lmax), 1/T217beam[:spectra_lmax+1])
			print('performing reconstructions for P_me', end='   ')
			noise_matching = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=hp.alm2cl(SMICA_gauss_realization), clgg_binned=hp.alm2cl(unWISEmap_alms), cltaudg_binned=Cltaug_at_zbar.copy())
			#_, _, outmap_gauss_realization = estim.combine_alm(SMICA_gauss_realization, unWISEmap_alms, total_mask,  ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_recon,6144), lmax=spectra_lmax, convert_K=False)
			_, _, outmap_gauss_realization = estim.combine_alm(SMICA_gauss_realization, unWISEmap_alms, total_mask,  ClTT, hp.alm2cl(unWISEmap_alms), Cltaug_at_zbar.copy(), np.repeat(noise_matching,6144), lmax=spectra_lmax, convert_K=False)
			#_, _, outmap_T100_gauss_realization = estim.combine_alm(T100_gauss_realization, unWISEmap_alms, total_mask,  ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), lmax=spectra_lmax, convert_K=True)
			#_, _, outmap_T143_gauss_realization = estim.combine_alm(T143_gauss_realization, unWISEmap_alms, total_mask,  ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), lmax=spectra_lmax, convert_K=False)
			#_, _, outmap_T217_gauss_realization = estim.combine_alm(T217_gauss_realization, unWISEmap_alms, total_mask,  ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), lmax=spectra_lmax, convert_K=False)
			#_, _, outmap_T217huge_gauss_realization = estim.combine_alm(T217_huge_gauss_realization, unWISEmap_alms, hugemask_unwise,  ClTT_217_huge, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217_huge,6144), lmax=spectra_lmax, convert_K=False)
			print('performing reconstructions for P_mm', end='   ')
			#_, _, outmap_gauss_realization_mm_me = estim.combine_alm(SMICA_gauss_realization, unWISEmap_alms, total_mask,  ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar_mm_me.copy(), np.repeat(noise_SMICA_mm_me,6144), lmax=spectra_lmax, convert_K=False)
			#_, _, outmap_T100_gauss_realization_mm_me = estim.combine_alm(T100_gauss_realization, unWISEmap_alms, total_mask,  ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar_mm_me.copy(), np.repeat(noise_T100_mm_me,6144), lmax=spectra_lmax, convert_K=True)
			#_, _, outmap_T143_gauss_realization_mm_me = estim.combine_alm(T143_gauss_realization, unWISEmap_alms, total_mask,  ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar_mm_me.copy(), np.repeat(noise_T143_mm_me,6144), lmax=spectra_lmax, convert_K=False)
			#_, _, outmap_T217_gauss_realization_mm_me = estim.combine_alm(T217_gauss_realization, unWISEmap_alms, total_mask,  ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar_mm_me.copy(), np.repeat(noise_T217_mm_me,6144), lmax=spectra_lmax, convert_K=False)
			#_, _, outmap_T217huge_gauss_realization_mm_me = estim.combine_alm(T217_huge_gauss_realization, unWISEmap_alms, hugemask_unwise,  ClTT_217_huge, csm.Clgg[0,0,:], Cltaug_at_zbar_mm_me.copy(), np.repeat(noise_T217huge_mm_me,6144), lmax=spectra_lmax, convert_K=False)
			print('constructing signal+noise maps and Cls')
			v_realization = hp.synfast(clv_windowed_interp, 2048)
			#v_realization_mm_me = hp.synfast(clv_windowed_interp_mm_me, 2048)
			#constructed_signalmap = v_realization*total_mask + outmap_gauss_realization
			constructed_signalmap = outmap_gauss_realization
			#constructed_signalmap_T100 = v_realization*total_mask + outmap_T100_gauss_realization
			#constructed_signalmap_T143 = v_realization*total_mask + outmap_T143_gauss_realization
			#constructed_signalmap_T217 = v_realization*total_mask + outmap_T217_gauss_realization
			#constructed_signalmap_T217huge = v_realization*total_mask + outmap_T217huge_gauss_realization
			#constructed_signalmap_mm_me = v_realization_mm_me*total_mask + outmap_gauss_realization_mm_me
			#constructed_signalmap_T100_mm_me = v_realization_mm_me*total_mask + outmap_T100_gauss_realization_mm_me
			#constructed_signalmap_T143_mm_me = v_realization_mm_me*total_mask + outmap_T143_gauss_realization_mm_me
			#constructed_signalmap_T217_mm_me = v_realization_mm_me*total_mask + outmap_T217_gauss_realization_mm_me
			#constructed_signalmap_T217huge_mm_me = v_realization_mm_me*total_mask + outmap_T217huge_gauss_realization_mm_me
			velocity_noise_Cls.append(hp.anafast(constructed_signalmap, lmax=1000)/fsky)
			#velocity_noise_T100_Cls.append(hp.anafast(constructed_signalmap_T100, lmax=100)/fsky)
			#velocity_noise_T143_Cls.append(hp.anafast(constructed_signalmap_T143, lmax=100)/fsky)
			#velocity_noise_T217_Cls.append(hp.anafast(constructed_signalmap_T217, lmax=100)/fsky)
			#velocity_noise_T217huge_Cls.append(hp.anafast(constructed_signalmap_T217huge, lmax=100)/fsky_huge)
			#velocity_noise_Cls_mm_me.append(hp.anafast(constructed_signalmap_mm_me, lmax=100)/fsky)
			#velocity_noise_T100_Cls_mm_me.append(hp.anafast(constructed_signalmap_T100_mm_me, lmax=100)/fsky)
			#velocity_noise_T143_Cls_mm_me.append(hp.anafast(constructed_signalmap_T143_mm_me, lmax=100)/fsky)
			#velocity_noise_T217_Cls_mm_me.append(hp.anafast(constructed_signalmap_T217_mm_me, lmax=100)/fsky)
			#velocity_noise_T217huge_Cls_mm_me.append(hp.anafast(constructed_signalmap_T217huge_mm_me, lmax=100)/fsky_huge)
		velocity_noise_Cls = np.array(velocity_noise_Cls)
		#velocity_noise_T100_Cls = np.array(velocity_noise_T100_Cls)
		#velocity_noise_T143_Cls = np.array(velocity_noise_T143_Cls)
		#velocity_noise_T217_Cls = np.array(velocity_noise_T217_Cls)
		#velocity_noise_T217huge_Cls = np.array(velocity_noise_T217huge_Cls)
		#velocity_noise_Cls_mm_me = np.array(velocity_noise_Cls_mm_me)
		#velocity_noise_T100_Cls_mm_me = np.array(velocity_noise_T100_Cls_mm_me)
		#velocity_noise_T143_Cls_mm_me = np.array(velocity_noise_T143_Cls_mm_me)
		#velocity_noise_T217_Cls_mm_me = np.array(velocity_noise_T217_Cls_mm_me)
		#velocity_noise_T217huge_Cls_mm_me = np.array(velocity_noise_T217huge_Cls_mm_me)
		plt.figure()
		plt.xlim([1,1000])
		plt.loglog([0,velocity_noise_Cls.size+1],[noise_recon,noise_recon],ls='--',color='k',lw=2)
		plt.loglog(np.mean(velocity_noise_Cls,axis=0))
		plt.savefig(outdir+'test')
		#np.savez("data/gauss_reals/noise_plus_velocity_full_%d-%d.npz"%(big_i*100,(big_i+1)*100-1),
		#	v_plus_n=velocity_noise_Cls,v_mm_me_plus_n_mm_me=velocity_noise_Cls_mm_me,
		#	v_plus_n_100=velocity_noise_T100_Cls,     v_mm_me_plus_n_mm_me_100=velocity_noise_T100_Cls_mm_me,
		#	v_plus_n_143=velocity_noise_T143_Cls,     v_mm_me_plus_n_mm_me_143=velocity_noise_T143_Cls_mm_me,
		#	v_plus_n_217=velocity_noise_T217_Cls,     v_mm_me_plus_n_mm_me_217=velocity_noise_T217_Cls_mm_me,
		#	v_plus_n_217huge=velocity_noise_T217huge_Cls, v_mm_me_plus_n_mm_me_217huge=velocity_noise_T217huge_Cls_mm_me)
else:
	filelist = [f for f in os.listdir('data/gauss_reals') if f.startswith('noise_plus_velocity') and f.endswith('.npz') and 'full' not in f]
	velocity_noise_Cls = np.zeros((len(filelist)*100, 6143))
	velocity_noise_Cls_mm_me = np.zeros((len(filelist)*100, 6143))
	for big_i in np.arange(len(filelist)):
		velocity_noise_Cls[big_i*100:(big_i+1)*100,:] = np.load('data/gauss_reals/'+filelist[big_i])['v_plus_n'][:,1:]
		velocity_noise_Cls_mm_me[big_i*100:(big_i+1)*100,:] = np.load('data/gauss_reals/'+filelist[big_i])['v_mm_me_plus_n_mm_me'][:,1:]











nells_bands = 5  # One that gives us an integer ell
bandpowers_shape = (ls.max() // nells_bands, nells_bands)
bandpowers = lambda spectrum : np.reshape(spectrum[1:ls.max()-3], bandpowers_shape).mean(axis=1)

ells_plot = np.arange(ls.max()+1)
linecolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure()
plt.semilogy(velocity_compute_ells[:20],clv[:20,zbar_index,zbar_index], label='True velocity')
plt.fill_between(velocity_compute_ells[:20], clv_windowed[:20]-np.std(Clv_windowed_store,axis=0)[:20], clv_windowed[:20]+np.std(Clv_windowed_store,axis=0)[:20], label='Windowed velocity', alpha=0.5, color=linecolors[1])
plt.fill_between(velocity_compute_ells[:20], clv_windowed_mm_me[:20]-np.std(Clv_windowed_mm_me_store,axis=0)[:20], clv_windowed_mm_me[:20]+np.std(Clv_windowed_mm_me_store,axis=0)[:20], label=r'Windowed velocity if $P_{\mathrm{em}}=P_{\mathrm{mm}}$', alpha=0.5,color=linecolors[2])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$', rotation=0.)
plt.legend()
plt.title('Velocity')
plt.savefig(outdir+'dndz_err_velocities')

vv_mm_me_err_const = clv_windowed_mm_me - clv_windowed
vv_dndz_err =interp1d(velocity_compute_ells, np.std(Clv_windowed_store,axis=0),bounds_error=False,fill_value=0.)(np.arange(ells_plot.size))
vv_mm_me_upper_err = interp1d(velocity_compute_ells, np.std(Clv_windowed_mm_me_store,axis=0) + vv_mm_me_err_const, bounds_error=False,fill_value=0.)(np.arange(ells_plot.size))

plt.figure()
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls[:ells_plot.size] / fsky),label='Planck x unWISE Reconstruction', ls='None', marker='x', zorder=101,color=linecolors[0])
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls[:ells_plot.size] / fsky), yerr=bandpowers(np.std(recon_Cls_dndz[:ells_plot.size]/fsky,axis=0)),label='+ dN/dz error', ls='None', marker='None', zorder=100,color=linecolors[1])
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls[:ells_plot.size] / fsky), yerr=bandpowers(np.std(recon_Cls_dndz[:ells_plot.size]/fsky,axis=0)+np.std(gauss_unwisemask['Cl_Tgauss_greal'],axis=0) / fsky),label='+ dN/dz + stat error', ls='None', marker='None', zorder=99,capsize=3,color=linecolors[2])
#plt.errorbar(np.arange(1,ls.max()), (recon_Cls[:ells_plot.size] / fsky)[1:ls.max()], c=linecolors[0], ls='--', alpha=0.5)
plt.errorbar(ells_plot[1:], np.repeat(noise_recon,ells_plot.size-1), c='k',label='Predicted Noise', ls='--', zorder=10, lw=1)
plt.xlim([0, 40])
plt.yscale('log')
y1,y2=plt.ylim()
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(interp1d(velocity_compute_ells, clv_windowed,bounds_error=False,fill_value=0.)(np.arange(ells_plot.size))[:ells_plot.size] / fsky), label='Velocity Signal', ls='None', marker='x', zorder=101,color=linecolors[3])
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(interp1d(velocity_compute_ells, clv_windowed,bounds_error=False,fill_value=0.)(np.arange(ells_plot.size))[:ells_plot.size] / fsky), yerr=bandpowers(vv_dndz_err[:ells_plot.size]/fsky),label='+ dN/dz error', ls='None', marker='None', zorder=100,color=linecolors[1])
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(interp1d(velocity_compute_ells, clv_windowed,bounds_error=False,fill_value=0.)(np.arange(ells_plot.size))[:ells_plot.size] / fsky), yerr=np.array([bandpowers(vv_dndz_err[:ells_plot.size]/fsky), bandpowers(vv_mm_me_upper_err[:ells_plot.size]/fsky)]), label='+ dN/dz + P_me model error', ls='None', marker="None", zorder=99, capsize=3,color=linecolors[4])
plt.ylim([y1,y2])
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
plt.title('Planck x unWISE Reconstruction')
plt.tight_layout()
plt.savefig(outdir+'dndz_err_signal_noise_gauss.png')


ells = np.arange(1,ls.max())

premask_fudge = 2.433764429900113
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,6))
# 100 GHz
ax1.errorbar(ells, (recon_Cls_100[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[100]['full'],axis=0)[ells]/fsky/premask_fudge, label='Full Sky', marker='x', capsize=3)
ax1.errorbar(ells, (recon_Cls_100_foregrounds[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[100]['noCMB'],axis=0)[ells]/fsky/premask_fudge, label='SMICA Subtracted Sky', marker='x', capsize=3)
ax1.errorbar(ells, (recon_Cls_100_thermaldust[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[100]['thermaldust'],axis=0)[ells]/fsky/premask_fudge, label='Thermal Dust Sim', marker='x', capsize=3)
# 143 GHz
ax2.errorbar(ells, (recon_Cls_143[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[143]['full'],axis=0)[ells]/fsky/premask_fudge, label='Full Sky', marker='x', capsize=3)
ax2.errorbar(ells, (recon_Cls_143_foregrounds[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[143]['noCMB'],axis=0)[ells]/fsky/premask_fudge, label='SMICA Subtracted Sky', marker='x', capsize=3)
ax2.errorbar(ells, (recon_Cls_143_thermaldust[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[143]['thermaldust'],axis=0)[ells]/fsky/premask_fudge, label='Thermal Dust Sim', marker='x', capsize=3)
# 217 GHz
ax3.errorbar(ells, (recon_Cls_217[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[217]['full'],axis=0)[ells]/fsky/premask_fudge, label='Full Sky', marker='x', capsize=3)
ax3.errorbar(ells, (recon_Cls_217_foregrounds[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[217]['noCMB'],axis=0)[ells]/fsky/premask_fudge, label='SMICA Subtracted Sky', marker='x', capsize=3)
ax3.errorbar(ells, (recon_Cls_217_thermaldust[ells] / fsky) / premask_fudge, yerr=np.std(recon_Cls_dndz_freq[217]['thermaldust'],axis=0)[ells]/fsky/premask_fudge, label='Thermal Dust Sim', marker='x', capsize=3)

for freq, ax in zip((100,143,217),(ax1, ax2,ax3)):
	_ = ax.set_xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
	ax.set_xlim([0, 50])
	ax.set_ylim([5e-11, 5e-7])
	ax.set_yscale('log')
	ax.set_title('%dGHz'%freq)
	ax.set_ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
	ax.set_xlabel(r'$\ell$',fontsize=16)
	ax.legend()

plt.savefig(outdir+'dndz_Foreground_Contributions.png')


### Read in realizations to compute statistical error
filelist = [f for f in os.listdir('data/gauss_reals') if f.startswith('noise_plus_velocity_full') and f.endswith('.npz')]
velocity_noise_Cls = np.zeros((len(filelist)*100, 6143))
velocity_noise_Cls_mm_me = np.zeros((len(filelist)*100, 6143))
for big_i in np.arange(len(filelist)):
	velocity_noise_Cls[big_i*100:(big_i+1)*100,:] = np.array([interp1d(np.arange(101), np.load('data/gauss_reals/'+filelist[big_i])['v_plus_n'][i,:], bounds_error=False, fill_value=0.)(np.arange(6144))[1:] for i in np.arange(100)])
	velocity_noise_Cls_mm_me[big_i*100:(big_i+1)*100,:] = np.array([interp1d(np.arange(101), np.load('data/gauss_reals/'+filelist[big_i])['v_mm_me_plus_n_mm_me'][i,:], bounds_error=False, fill_value=0.)(np.arange(6144))[1:] for i in np.arange(100)])

# Reconstruct with Cltaug based on Pmm instead of Pme
Cltaug_at_zbar_Pmm = interp1d(fullspectrum_ls, (Pmms[zbar_index] * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
noise_recon_Pmm = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_Pmm.copy())
_, _, outmap_SMICA_Pmm = estim.combine(SMICAmap_real, unWISEmap, total_mask, ClTT, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_recon_Pmm,6144), convert_K=False)
recon_Cls_Pmm = hp.anafast(outmap_SMICA_Pmm)

error_mm_me_recon = recon_Cls - recon_Cls_Pmm
recon_errs = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz,axis=0)[:ells_plot.size]**2+error_mm_me_recon[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz,axis=0)[:ells_plot.size]/fsky)])
Clv_windowed_store_interp = np.array([interp1d(velocity_compute_ells, Clv_windowed_store[i], bounds_error=False, fill_value=0.)(np.arange(1,6144)) for i in np.arange(Clv_windowed_store.shape[0])])
Clv_windowed_mm_me_store_interp = np.array([interp1d(velocity_compute_ells, Clv_windowed_mm_me_store[i], bounds_error=False, fill_value=0.)(np.arange(1,6144)) for i in np.arange(Clv_windowed_mm_me_store.shape[0])])
clv_windowed_interp = interp1d(velocity_compute_ells,clv_windowed, bounds_error=False,fill_value=0.)(np.arange(1,6144))
clv_windowed_mm_me_interp = interp1d(velocity_compute_ells, clv_windowed_mm_me, bounds_error=False,fill_value=0.)(np.arange(1,6144))

Cltaug_at_zbar_mm_me_store = np.zeros((100,6144))
noises_dndz_mm_me = np.zeros((100,6144))
for dndz_iter_ind in np.arange(100):
	galaxy_window_dndz_ind = csm.get_limber_window('gerr_alex_%d.txt' % dndz_iter_ind, chis, avg=False)[zbar_index]
	Cltaug_at_zbar_mm_me_store[dndz_iter_ind] = interp1d(fullspectrum_ls, (Pmms[zbar_index] * galaxy_window_dndz_ind * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
	noises_dndz_mm_me[dndz_iter_ind] = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=Clgg_store[dndz_iter_ind].copy(), cltaudg_binned=Cltaug_at_zbar_mm_me_store[dndz_iter_ind].copy())
	



noise_recon = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=hp.alm2cl(unWISEmap_alms), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_SMICA = estim.combine(SMICAmap_real, unWISEmap, total_mask, ClTT, np.append(hp.alm2cl(unWISEmap_alms),np.zeros(6144-hp.alm2cl(unWISEmap_alms).size)), Cltaug_at_zbar.copy(), np.repeat(noise_recon,6144), convert_K=False)
recon_Cls = hp.anafast(outmap_SMICA)

### TEMP COMMANDER CHECK
COMMANDERinp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-commander_2048_R3.00_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725  # Remove K_CMB units
COMMANDERmap_real_alms_masked_debeamed = hp.almxfl(hp.map2alm(COMMANDERinp, lmax=spectra_lmax), 1/SMICAbeam)
COMMANDERmap_real = hp.alm2map(COMMANDERmap_real_alms_masked_debeamed, 2048)  # De-beamed unitless COMMANDER map
ClTT_COMMANDER = hp.alm2cl(hp.almxfl(hp.map2alm(COMMANDERinp * planckmask), 1/SMICAbeam)) / fsky_cltt
ClTT_COMMANDER[ls.max()+1:] = 0.

noise_recon_COMMANDER = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_COMMANDER.copy(), clgg_binned=hp.alm2cl(unWISEmap_alms), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_COMMANDER = estim.combine(COMMANDERmap_real, unWISEmap, total_mask, ClTT_COMMANDER, np.append(hp.alm2cl(unWISEmap_alms),np.zeros(6144-hp.alm2cl(unWISEmap_alms).size)), Cltaug_at_zbar.copy(), np.repeat(noise_recon_COMMANDER,6144), convert_K=False)
recon_Cls_COMMANDER = hp.anafast(outmap_COMMANDER)
### TEMP COMMANDER CHECK END



### Do a Gaussian SMICA reconstruction and compare to noise on full sky reconstruction.
### Do a line with a galactic plane cut / fsky
### Do a line with planckmask / fsky_cltt
### Do a line with total_mask / fsky
### NAMASTER?
plt.figure()
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls[:ells_plot.size] / fsky), yerr=recon_errs,label='SMICA x unWISE Reconstruction', ls='None', marker='x', zorder=100,color=linecolors[0], capsize=3)
plt.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_COMMANDER[:ells_plot.size] / fsky),label='COMMANDER x unWISE Reconstruction', ls='None', marker='x', zorder=100,color=linecolors[3], capsize=3)
plt.semilogy(ells, np.repeat(noise_recon,ells.size), c='k',label='SMICA Theory Noise', ls='--', zorder=10, lw=2)
plt.semilogy(ells, np.repeat(noise_recon_COMMANDER,ells.size), c='gray',label='COMMANDER Theory Noise', ls='--', zorder=10, lw=2)
plt.xlim([0, 250])
#plt.semilogy(velocity_compute_ells, clv_windowed+noise_recon, color=linecolors[1],lw=2,label='Windowed velocity')
#plt.fill_between(ells, (clv_windowed_interp-np.std(Clv_windowed_store_interp,axis=0)+noise_recon-np.std(noises_dndz,axis=0)[:6143])[:ells.size], (clv_windowed_mm_me_interp+np.std(Clv_windowed_mm_me_store_interp,axis=0)+noise_recon_Pmm+np.std(noises_dndz_mm_me,axis=0)[:6143])[:ells.size], alpha=0.5, color=linecolors[1])
#plt.fill_between(ells, (np.mean(velocity_noise_Cls,axis=0)-np.sqrt(np.std(velocity_noise_Cls,axis=0)**2+(np.std(Clv_windowed_store_interp,axis=0)+np.std(noises_dndz[:,1:],axis=0))**2))[:ells.size], (np.mean(velocity_noise_Cls_mm_me,axis=0)+np.sqrt(np.std(velocity_noise_Cls_mm_me,axis=0)**2+(np.std(Clv_windowed_mm_me_store_interp,axis=0)+np.std(noises_dndz_mm_me[:,1:],axis=0))**2))[:ells.size],alpha=0.35,color=linecolors[1])
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
plt.title('Planck x unWISE Reconstruction')
plt.tight_layout()
plt.savefig(outdir+'signal_noise_gauss.png')



## Compute frequency reconstructions for Pmm case
noise_100_Pmm = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_100.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_Pmm.copy())
noise_143_Pmm = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_143.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_Pmm.copy())
noise_217_Pmm = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_Pmm.copy())
noise_217_Pmm_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar_Pmm.copy())
_, _, outmap_100_Pmm = estim.combine(T100map_real, unWISEmap, total_mask, ClTT_100, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_100_Pmm,6144), convert_K=True)
_, _, outmap_100_foregrounds_Pmm = estim.combine(Tmap_noCMB_100_masked_debeamed, unWISEmap, total_mask, ClTT_100, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_100_Pmm,6144), convert_K=True)
_, _, outmap_100_thermaldust_Pmm = estim.combine(Tmap_thermaldust_100, unWISEmap, total_mask, ClTT_100, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_100_Pmm,6144), convert_K=True)
_, _, outmap_143_Pmm = estim.combine(T143map_real, unWISEmap, total_mask, ClTT_143, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_143_Pmm,6144), convert_K=False)
_, _, outmap_143_foregrounds_Pmm = estim.combine(Tmap_noCMB_143_masked_debeamed, unWISEmap, total_mask, ClTT_143, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_143_Pmm,6144), convert_K=False)
_, _, outmap_143_thermaldust_Pmm = estim.combine(Tmap_thermaldust_143, unWISEmap, total_mask, ClTT_143, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_143_Pmm,6144), convert_K=False)
_, _, outmap_217_Pmm = estim.combine(T217map_real, unWISEmap, total_mask, ClTT_217, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_217_Pmm,6144), convert_K=False)
_, _, outmap_217_foregrounds_Pmm = estim.combine(Tmap_noCMB_217_masked_debeamed, unWISEmap, total_mask, ClTT_217, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_217_Pmm,6144), convert_K=False)
_, _, outmap_217_thermaldust_Pmm = estim.combine(Tmap_thermaldust_217, unWISEmap, total_mask, ClTT_217, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_217_Pmm,6144), convert_K=False)
_, _, outmap_217_Pmm_huge = estim.combine(T217map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_217_Pmm_huge,6144), convert_K=False)
_, _, outmap_217_foregrounds_Pmm_huge = estim.combine(Tmap_noCMB_217_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_217_Pmm_huge,6144), convert_K=False)
_, _, outmap_217_thermaldust_Pmm_huge = estim.combine(Tmap_thermaldust_217_hugemask, unWISEmap, hugemask_unwise, ClTT_217_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar_Pmm.copy(), np.repeat(noise_217_Pmm_huge,6144), convert_K=False)
recon_Cls_100_Pmm = hp.anafast(outmap_100_Pmm)
recon_Cls_100_foregrounds_Pmm = hp.anafast(outmap_100_foregrounds_Pmm)
recon_Cls_100_thermaldust_Pmm = hp.anafast(outmap_100_thermaldust_Pmm)
recon_Cls_143_Pmm = hp.anafast(outmap_143_Pmm)
recon_Cls_143_foregrounds_Pmm = hp.anafast(outmap_143_foregrounds_Pmm)
recon_Cls_143_thermaldust_Pmm = hp.anafast(outmap_143_thermaldust_Pmm)
recon_Cls_217_Pmm = hp.anafast(outmap_217_Pmm)
recon_Cls_217_foregrounds_Pmm = hp.anafast(outmap_217_foregrounds_Pmm)
recon_Cls_217_thermaldust_Pmm = hp.anafast(outmap_217_thermaldust_Pmm)
recon_Cls_217_Pmm_huge = hp.anafast(outmap_217_Pmm_huge)
recon_Cls_217_foregrounds_Pmm_huge = hp.anafast(outmap_217_foregrounds_Pmm_huge)
recon_Cls_217_thermaldust_Pmm_huge = hp.anafast(outmap_217_thermaldust_Pmm_huge)
error_mm_me_recon_100 = recon_Cls_100 - recon_Cls_100_Pmm
error_mm_me_recon_100_foregrounds = recon_Cls_100_foregrounds - recon_Cls_100_foregrounds_Pmm
error_mm_me_recon_100_thermaldust = recon_Cls_100_thermaldust - recon_Cls_100_thermaldust_Pmm
error_mm_me_recon_143 = recon_Cls_143 - recon_Cls_143_Pmm
error_mm_me_recon_143_foregrounds = recon_Cls_143_foregrounds - recon_Cls_143_foregrounds_Pmm
error_mm_me_recon_143_thermaldust = recon_Cls_143_thermaldust - recon_Cls_143_thermaldust_Pmm
error_mm_me_recon_217 = recon_Cls_217 - recon_Cls_217_Pmm
error_mm_me_recon_217_foregrounds = recon_Cls_217_foregrounds - recon_Cls_217_foregrounds_Pmm
error_mm_me_recon_217_thermaldust = recon_Cls_217_thermaldust - recon_Cls_217_thermaldust_Pmm
error_mm_me_recon_217_huge = recon_Cls_217_huge - recon_Cls_217_Pmm_huge
error_mm_me_recon_217_foregrounds_huge = recon_Cls_217_foregrounds_huge - recon_Cls_217_foregrounds_Pmm_huge
error_mm_me_recon_217_thermaldust_huge = recon_Cls_217_thermaldust_huge - recon_Cls_217_thermaldust_Pmm_huge

## Combine dN/dz and Pme/Pmm errors in quadrature for measurement error bars
recon_100_yerr = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[100]['full'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_100[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[100]['full'],axis=0)[:ells_plot.size]/fsky)])
recon_100_yerr_foregrounds = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[100]['noCMB'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_100_foregrounds[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[100]['noCMB'],axis=0)[:ells_plot.size]/fsky)])
recon_100_yerr_thermaldust = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[100]['thermaldust'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_100_thermaldust[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[100]['thermaldust'],axis=0)[:ells_plot.size]/fsky)])
recon_143_yerr = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[143]['full'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_143[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[143]['full'],axis=0)[:ells_plot.size]/fsky)])
recon_143_yerr_foregrounds = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[143]['noCMB'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_143_foregrounds[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[143]['noCMB'],axis=0)[:ells_plot.size]/fsky)])
recon_143_yerr_thermaldust = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[143]['thermaldust'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_143_thermaldust[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[143]['thermaldust'],axis=0)[:ells_plot.size]/fsky)])
recon_217_yerr = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[217]['full'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_217[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[217]['full'],axis=0)[:ells_plot.size]/fsky)])
recon_217_yerr_foregrounds = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[217]['noCMB'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_217_foregrounds[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[217]['noCMB'],axis=0)[:ells_plot.size]/fsky)])
recon_217_yerr_thermaldust = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[217]['thermaldust'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_217_thermaldust[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[217]['thermaldust'],axis=0)[:ells_plot.size]/fsky)])
recon_217_yerr_huge = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_217_huge,axis=0)[:ells_plot.size]**2+error_mm_me_recon_217_huge[:ells_plot.size]**2)/fsky_huge),bandpowers(np.std(recon_Cls_dndz_217_huge,axis=0)[:ells_plot.size]/fsky_huge)])
recon_217_yerr_foregrounds_huge = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_217_huge_foregrounds,axis=0)[:ells_plot.size]**2+error_mm_me_recon_217_foregrounds_huge[:ells_plot.size]**2)/fsky_huge),bandpowers(np.std(recon_Cls_dndz_217_huge_foregrounds,axis=0)[:ells_plot.size]/fsky_huge)])
recon_217_yerr_thermaldust_huge = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_217_huge_thermaldust,axis=0)[:ells_plot.size]**2+error_mm_me_recon_217_thermaldust_huge[:ells_plot.size]**2)/fsky_huge),bandpowers(np.std(recon_Cls_dndz_217_huge_thermaldust,axis=0)[:ells_plot.size]/fsky_huge)])

fig, ((ax100, ax143), (ax217, ax217huge)) = plt.subplots(2,2,figsize=(12,12))

ax100.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100[:ells_plot.size] / fsky), yerr=recon_100_yerr,label='100 GHz x unWISE', ls='None', marker='x', zorder=100,color=linecolors[2], capsize=3)
ax100.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100_foregrounds[:ells_plot.size] / fsky), yerr=recon_100_yerr_foregrounds,label='(100 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=100,color=linecolors[3], capsize=3)
ax100.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100_thermaldust[:ells_plot.size] / fsky), yerr=recon_100_yerr_thermaldust,label='Thermal dust x unWISE', ls='None', marker='x', zorder=100,color=linecolors[4], capsize=3)
ax143.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143[:ells_plot.size] / fsky), yerr=recon_143_yerr,label='143 GHz x unWISE', ls='None', marker='x', zorder=100,color=linecolors[2], capsize=3)
ax143.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143_foregrounds[:ells_plot.size] / fsky), yerr=recon_143_yerr_foregrounds,label='(143 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=143,color=linecolors[3], capsize=3)
ax143.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143_thermaldust[:ells_plot.size] / fsky), yerr=recon_143_yerr_thermaldust,label='Thermal dust x unWISE', ls='None', marker='x', zorder=143,color=linecolors[4], capsize=3)
ax217.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217[:ells_plot.size] / fsky), yerr=recon_217_yerr,label='217 GHz x unWISE', ls='None', marker='x', zorder=100,color=linecolors[2], capsize=3)
ax217.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217_foregrounds[:ells_plot.size] / fsky), yerr=recon_217_yerr_foregrounds,label='(217 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=217,color=linecolors[3], capsize=3)
ax217.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217_thermaldust[:ells_plot.size] / fsky), yerr=recon_217_yerr_thermaldust,label='Thermal dust x unWISE', ls='None', marker='x', zorder=217,color=linecolors[4], capsize=3)
ax217huge.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217_huge[:ells_plot.size] / fsky_huge), yerr=recon_217_yerr_huge,label='217 GHz x unWISE', ls='None', marker='x', zorder=100,color=linecolors[2], capsize=3)
ax217huge.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217_foregrounds_huge[:ells_plot.size] / fsky_huge), yerr=recon_217_yerr_foregrounds_huge,label='(217 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=100,color=linecolors[3], capsize=3)
ax217huge.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217_thermaldust_huge[:ells_plot.size] / fsky_huge), yerr=recon_217_yerr_thermaldust_huge,label='Thermal dust x unWISE', ls='None', marker='x', zorder=100,color=linecolors[4], capsize=3)


for ax in [ax100, ax143, ax217, ax217huge]:
	ax.set_xlim([0, 25])
	#ax.set_ylim([1e-10,3e-6])
	ax.set_yscale('log')
	#ax.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls[:ells_plot.size] / fsky), yerr=recon_errs,label='Fiducial reconstruction', ls='None', marker='x', zorder=100,color=linecolors[0], capsize=3,alpha=0.75)
	ax.errorbar(velocity_compute_ells, clv_windowed+noise_recon, color=linecolors[1],lw=2,label='Windowed velocity',alpha=1*0.8)
	ax.fill_between(velocity_compute_ells[:20], clv_windowed[:20]-np.std(Clv_windowed_store,axis=0)[:20]+noise_recon, clv_windowed_mm_me[:20]+np.std(Clv_windowed_mm_me_store,axis=0)[:20]+noise_recon, alpha=0.5*0.8, color=linecolors[1])
	ax.fill_between(ells, (np.mean(velocity_noise_Cls,axis=0)-np.std(velocity_noise_Cls,axis=0))[:ells.size], (np.mean(velocity_noise_Cls,axis=0)+np.std(velocity_noise_Cls,axis=0))[:ells.size],alpha=0.35*0.8,color=linecolors[1])
	handles, labels = ax.get_legend_handles_labels()
	order = [1,2,3,0]
	ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
	ax.legend()
	ax.set_xlabel(r'$\ell$')
	ax.set_ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
	ax.set_title('Reconstructions at %d GHz, fsky = %.2f' % {ax100 : (100,fsky), ax143 : (143,fsky), ax217 : (217,fsky), ax217huge : (217,fsky_huge)}[ax])

plt.tight_layout()
plt.savefig(outdir+'Foreground_Contributions')


lowpass_SMICA_plot = hp.alm2map(hp.almxfl(hp.map2alm(outmap_SMICA), [0 if l > 25 else 1 for l in np.arange(6144)]), 2048)

plt.figure()
hp.mollview(lowpass_SMICA_plot*total_mask, title=r'SMICA x unWISE Reconstruction $\left(\ell_{\mathrm{max}}\leq 25\right)$',unit=r'$\frac{v}{c}$')
plt.savefig(outdir+'SMICA_out')

c = mpl.colors.ListedColormap(['darkred', 'gold'])
plt.figure()
hp.mollview(total_mask,cmap=c,cbar=False,title=r'Fiducial Reconstruction Mask $\left(f_{\mathrm{sky}}=%.2f\right)$' % fsky)
plt.savefig(outdir+'Mask input')

plt.figure()
hp.mollview(hugemask_unwise,cmap=c,cbar=False,title=r'Large Reconstruction Mask $\left(f_{\mathrm{sky}}=%.2f\right)$' % fsky_huge)
plt.savefig(outdir+'hugemask_unwise')

with open('data/unWISE/blue.txt', 'r') as FILE:
    x = FILE.readlines()

z = np.array([float(l.split(' ')[0]) for l in x])
dndz = np.array([float(l.split(' ')[1]) for l in x])
dndz_interp = interp1d(z ,dndz, kind= 'linear', bounds_error=False, fill_value=0)(redshifts)
dndz_all = np.zeros((100, dndz.size))
for i in np.arange(100):
	with open('data/unWISE/blue_dNdz_err/%s.txt' % i, 'r') as FILE:
		x = FILE.readlines()
	dndz_all[i,:] = np.array([float(l.split(' ')[1]) for l in x])

dndz_all_interp = np.array([interp1d(z, dndz_all[i,:], kind='linear', bounds_error=False, fill_value=0)(redshifts) for i in np.arange(100)])

plt.figure()
plt.plot(redshifts, dndz_interp / simps(dndz_interp), lw=2, label=r'Best-fit $\frac{d\mathrm{N}}{dz}$',zorder=200,color='red')
plt.plot(redshifts, dndz_all_interp[0,:] / simps(dndz_all_interp[0,:]), lw=0.5, color='gray', alpha=0.75, label=r'Individual $\frac{d\mathrm{N}}{dz}$ realizations')
for i in np.arange(1,100):
	plt.plot(redshifts, dndz_all_interp[i,:] / simps(dndz_all_interp[i,:]), lw=0.5, color='gray', alpha=0.75)

plt.title('unWISE redshift distribution')
plt.xlabel(r'$z$')
plt.xlim([redshifts.min(), redshifts.max()])
leg = plt.legend()
for lh in leg.legendHandles[1:]:
	lh.set_alpha(1.)
	lh.set_linewidth(1.)

plt.tight_layout()
plt.savefig(outdir+'dndz')

plt.figure()
plt.semilogy(csm.Clgg[0,0,:], lw=2, color='red', zorder=200, label=r'$C_\ell^{\mathrm{gg}}$ for best-fit $\frac{d\mathrm{N}}{dz}$')
plt.semilogy(Clgg_store[0,:], lw=0.5, color='gray', alpha=0.75, label=r'$C_\ell^{\mathrm{gg}}$ for $\frac{d\mathrm{N}}{dz}$ realizations')
for i in np.arange(1,100):
	plt.semilogy(Clgg_store[i,:], lw=0.5, color='gray', alpha=0.75)

plt.title('unWISE blue modelled galaxy power spectrum')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{gg}}$')
plt.xlim([101,4000])
plt.ylim([plt.ylim()[0], csm.Clgg[0,0,101]*2])
leg = plt.legend()
for lh in leg.legendHandles[1:]:
	lh.set_alpha(1.)
	lh.set_linewidth(1.)

plt.tight_layout()
plt.savefig(outdir+'clgg')


galaxy_windows_dndz = np.zeros((100,chis.size))
for dndz_iter_ind in np.arange(100):
    galaxy_windows_dndz[dndz_iter_ind,:] = csm.get_limber_window('gerr_alex_%d.txt' % dndz_iter_ind, chis, avg=False)  # Units of 1/Mpc

window_v_me_me = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( galaxy_windows_dndz / galaxy_windows_dndz[:,zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
window_v_mm_me = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( galaxy_windows_dndz / galaxy_windows_dndz[:,zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )

window_v_fiducial = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
mm_me_bias = np.mean(window_v_mm_me,axis=0)-np.mean(window_v_me_me,axis=0)

def y_fmt(x, y):
    return '{:2.1e}'.format(x)


plt.figure()
plt.plot(redshifts, window_v_fiducial,color=linecolors[1],lw=2,label='Windowed velocity')
plt.fill_between(redshifts, window_v_fiducial-np.std(window_v_mm_me,axis=0),window_v_fiducial+np.sqrt(np.std(window_v_mm_me,axis=0)**2+mm_me_bias**2),color=linecolors[1],alpha=0.5)
plt.legend()
plt.xlabel(r'$z$')
plt.ylabel(r'$W_v\left(z\right)$')
plt.title('Signal velocity window')
plt.xlim([redshifts.min(),redshifts.max()])
plt.gca().yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
plt.tight_layout()
plt.savefig(outdir+'window_v')


plt.semilogy(ells, np.repeat(noise_recon,ells.size), c='k',label='Predicted Noise', ls='--', zorder=10, lw=2)

y1,y2=plt.ylim()
plt.semilogy(velocity_compute_ells, clv_windowed+noise_recon, color=linecolors[1],lw=2,label='Windowed velocity')
plt.fill_between(velocity_compute_ells[:20], clv_windowed[:20]-np.std(Clv_windowed_store,axis=0)[:20]+noise_recon, clv_windowed_mm_me[:20]+np.std(Clv_windowed_mm_me_store,axis=0)[:20]+noise_recon, alpha=0.5, color=linecolors[1])
plt.fill_between(ells, (np.mean(velocity_noise_Cls,axis=0)-np.std(velocity_noise_Cls,axis=0))[:ells.size], (np.mean(velocity_noise_Cls,axis=0)+np.std(velocity_noise_Cls,axis=0))[:ells.size],alpha=0.35,color=linecolors[1])
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
plt.title('Planck x unWISE Reconstruction')

plt.tight_layout()
plt.savefig(outdir+'frequency_analysis_2pt.png')

# weights_bluestar = hp.ud_grade(fits.open('data/unWISE/blue_star_weights.fits')[1].data['I'].flatten(), 2048)
# weights_w2 = hp.ud_grade(fits.open('data/unWISE/blue_w2_5sig_weights.fits')[1].data['I'].flatten(), 2048)
# noisemap_143 = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/143ghz/ffp10_noise_143_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
# noisemap_143_Cls = hp.anafast(noisemap_143)
# noise_recon_143 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=noisemap_143_Cls.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
# _, _, outmap_143_unweighted = estim.combine(noisemap_143, unWISEmap, total_mask, noisemap_143_Cls, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon_143,6144), convert_K=False)
# _, _, outmap_143_weighted = estim.combine(noisemap_143, unWISEmap*weights_bluestar*weights_w2, total_mask, noisemap_143_Cls, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon_143,6144), convert_K=False)
# recon_Cls_143_unweighted = hp.anafast(outmap_143_unweighted)
# recon_Cls_143_weighted = hp.anafast(outmap_143_weighted)

# plt.figure()
# plt.semilogy(recon_Cls_143_unweighted,label='Unweighted unWISE')
# plt.semilogy(recon_Cls_143_weighted,  label='Weighted unWISE')
# plt.legend()
# plt.title('Reconstruction on 143GHz Noise')
# plt.xlabel(r'$\ell$')
# plt.xlim([1,30])
# plt.savefig(outdir+'weighted_unweighted_noise_recon')

centres = lambda BIN : (BIN[:-1]+BIN[1:]) / 2

# lowpass_output = lowpass(outmap_SMICA)
# n, bins = np.histogram(lowpass_output[np.where(total_mask!=0)], bins=nbin_hist)
# plt.figure()
# plt.errorbar(centres(bins), n, yerr=np.std(n_dndz,axis=0))
# plt.savefig(outdir+'dndz_err_histogram')





bessel = lambda bins, Tmap, lssmap : kn(0, np.abs(centres(bins)) / (np.std(Tmap)*np.std(lssmap)))
normal_product = lambda bins, Tmap, lssmap : bessel(bins,Tmap,lssmap) / (np.pi * np.std(Tmap) * np.std(lssmap))
pixel_scaling = lambda distribution : (12*2048**2) * (distribution / simps(distribution))
pixel_scaling_masked = lambda distribution, FSKY : (12*2048**2) * FSKY * (distribution / simps(distribution))

n_out_normprod, bins_out_normprod = np.histogram(outmap_SMICA[np.where(total_mask!=0)], bins=10000)

ClTT_filter = ClTT.copy()
Clgg_filter = csm.Clgg[0,0,:].copy()
ClTT_filter[:100] = 1e15
Clgg_filter[:100] = 1e15

Tmap_filtered  = hp.alm2map(hp.almxfl(hp.map2alm(SMICAmap_real), np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), 2048)*total_mask
lssmap_filtered = hp.alm2map(hp.almxfl(hp.map2alm(unWISEmap),    np.divide(Cltaug_at_zbar.copy(),     Clgg_filter, out=np.zeros_like(Cltaug_at_zbar.copy()),     where=Clgg_filter!=0)), 2048)*total_mask

std_Tmap_filtered   = np.std(Tmap_filtered[np.where(total_mask!=0)])
std_lssmap_filtered = np.std(lssmap_filtered[np.where(total_mask!=0)])

expect_normprod = normal_product(bins_out_normprod,Tmap_filtered[np.where(total_mask!=0)]*noise_recon,lssmap_filtered[np.where(total_mask!=0)])

plt.figure()
plt.fill_between(centres(bins_out_normprod), np.zeros(n_out_normprod.size), pixel_scaling_masked(n_out_normprod,fsky)/1e5, label='Velocity reconstruction')
plt.plot(centres(bins_out_normprod), pixel_scaling_masked(expect_normprod, fsky)/1e5,color='k', ls='--', lw=2., label='Normal product distribution')
y1, y2 = plt.ylim()
plt.ylim([0, y2]) 
plt.xlim([-.3,.3])
plt.xlabel(r'$\frac{\Delta T}{T}$', fontsize=16)
plt.ylabel(r'$N_{\mathrm{pix}}\ \left[\times 10^5\right]$')
plt.title('Velocity reconstruction pixel value distribution')
plt.legend()
plt.tight_layout()
plt.savefig(outdir+'recon_1pt')


nbin_hist=np.linspace(-1500,1000,10000) / 299792.458
n_100, bins_100 = np.histogram(lowpass_output_100[np.where(total_mask!=0)], bins=nbin_hist)
n_143, bins_143 = np.histogram(lowpass_output_143[np.where(total_mask!=0)], bins=nbin_hist)
n_217, bins_217 = np.histogram(lowpass_output_217[np.where(total_mask!=0)], bins=nbin_hist)
n_100_huge, bins_100_huge = np.histogram(lowpass_output_100_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_143_huge, bins_143_huge = np.histogram(lowpass_output_143_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_217_huge, bins_217_huge = np.histogram(lowpass_output_217_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)


darken = lambda color, amount :  colorsys.hls_to_rgb(colorsys.rgb_to_hls(*mc.to_rgb(color))[0], 1 - amount * (1 - colorsys.rgb_to_hls(*mc.to_rgb(color))[1]), colorsys.rgb_to_hls(*mc.to_rgb(color))[2])

plt.figure()
l1, = plt.plot(centres(nbin_hist)*299792.458, n_100/simps(n_100), label='100 GHz')
l2, = plt.plot(centres(nbin_hist)*299792.458, n_143/simps(n_143), label='143 GHz')
l3, = plt.plot(centres(nbin_hist)*299792.458, n_217/simps(n_217), label='217 GHz (fiducial mask)')
plt.plot(centres(nbin_hist)*299792.458, n_100_huge/simps(n_100_huge), label='100 GHz (large mask)', c=darken(l1.get_c(),1.4))
plt.plot(centres(nbin_hist)*299792.458, n_143_huge/simps(n_143_huge), label='143 GHz (large mask)', c=darken(l2.get_c(),1.5))
plt.plot(centres(nbin_hist)*299792.458, n_217_huge/simps(n_217_huge), label='217 GHz (large mask)', c=darken(l3.get_c(),1.3))
plt.xlabel('km/s')
plt.ylabel(r'Normalized $\mathrm{N}_{\mathrm{pix}}$')
plt.xlim([-1500,1000])
plt.legend()
plt.savefig(outdir+'1ptdrift')






unwise_gauss = hp.synfast(hp.anafast(unWISEmap), 2048)
SMICA_gauss = hp.alm2map(hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(SMICAinp*planckmask)/(np.where(planckmask!=0)[0].size/planckmask.size), 2048)), 1/SMICAbeam), 2048)

SMICAmap_use = hp.alm2map(hp.almxfl(hp.map2alm(SMICAinp*planckmask) / (np.where(planckmask!=0)[0].size/planckmask.size), 1/SMICAbeam), 2048)
ClTT_use = hp.anafast(SMICA_gauss)
ClTT_use[ls.max()+1:] = 0.

ClTT_filter = ClTT_use.copy()
Clgg_filter = csm.Clgg[0,0,:].copy()
ClTT_filter[:100] = 1e15
Clgg_filter[:100] = 1e15

noise_recon_gaussing = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_filter.copy(), clgg_binned=Clgg_filter.copy(), cltaudg_binned=Cltaug_at_zbar.copy())

Tmap_filtered  = hp.alm2map(hp.almxfl(hp.map2alm(SMICA_gauss), np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), 2048)*total_mask
Tmap_filtered_fiducial  = hp.alm2map(hp.almxfl(hp.map2alm(SMICAmap_real), np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), 2048)*total_mask
lssmap_filtered = hp.alm2map(hp.almxfl(hp.map2alm(unwise_gauss),    np.divide(Cltaug_at_zbar.copy(),     Clgg_filter, out=np.zeros_like(Cltaug_at_zbar.copy()),     where=Clgg_filter!=0)), 2048)*total_mask
lssmap_filtered_fiducial = hp.alm2map(hp.almxfl(hp.map2alm(unWISEmap),    np.divide(Cltaug_at_zbar.copy(),     Clgg_filter, out=np.zeros_like(Cltaug_at_zbar.copy()),     where=Clgg_filter!=0)), 2048)*total_mask

std_Tmap_filtered   = np.std(Tmap_filtered[np.where(total_mask!=0)])
std_lssmap_filtered = np.std(lssmap_filtered[np.where(total_mask!=0)])

_, _, outmap_SMICA_unWISE = estim.combine(SMICAmap_use, unWISEmap, total_mask, ClTT_use, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon_gaussing,6144), convert_K=False)
_, _, outmap_SMICA_gauss = estim.combine(SMICAmap_use, unwise_gauss, total_mask, ClTT_use, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon_gaussing,6144), convert_K=False)
_, _, outmap_gauss_unWISE = estim.combine(SMICA_gauss, unWISEmap, total_mask, ClTT_use, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon_gaussing,6144), convert_K=False)
_, _, outmap_gauss_gauss = estim.combine(SMICA_gauss, unwise_gauss, total_mask, ClTT_use, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_recon_gaussing,6144), convert_K=False)

n_out_normprod_SMICA_unWISE, bins_out_normprod_SMICA_unWISE = np.histogram(outmap_SMICA_unWISE[np.where(total_mask!=0)], bins=2500000)
n_out_normprod_SMICA_gauss, _ = np.histogram(outmap_SMICA_gauss[np.where(total_mask!=0)], bins=bins_out_normprod_SMICA_unWISE)
n_out_normprod_gauss_unWISE, _ = np.histogram(outmap_gauss_unWISE[np.where(total_mask!=0)], bins=bins_out_normprod_SMICA_unWISE)
n_out_normprod_gauss_gauss, _ = np.histogram(outmap_gauss_gauss[np.where(total_mask!=0)], bins=bins_out_normprod_SMICA_unWISE)

expect_normprod_gaussing = normal_product(bins_out_normprod_SMICA_unWISE,Tmap_filtered[np.where(total_mask!=0)]*noise_recon_gaussing,lssmap_filtered[np.where(total_mask!=0)])
expect_normprod_gaussing_fiducial = normal_product(bins_out_normprod_SMICA_unWISE,Tmap_filtered_fiducial[np.where(total_mask!=0)]*noise_recon,lssmap_filtered_fiducial[np.where(total_mask!=0)])

plt.figure()
plt.plot(centres(bins_out_normprod_SMICA_unWISE)*299792.458, pixel_scaling_masked(n_out_normprod_SMICA_unWISE,fsky)  /1e3,                           label='SMICA x unWISE')
plt.plot(centres(bins_out_normprod_SMICA_unWISE)*299792.458, pixel_scaling_masked(n_out_normprod_SMICA_gauss,fsky)  /1e3,                           label='SMICA x Gauss')
plt.plot(centres(bins_out_normprod_SMICA_unWISE)*299792.458, pixel_scaling_masked(n_out_normprod_gauss_unWISE,fsky)  /1e3,                           label='Gauss x unWISE')
plt.plot(centres(bins_out_normprod_SMICA_unWISE)*299792.458, pixel_scaling_masked(n_out_normprod_gauss_gauss,fsky)  /1e3,                           label='Gauss x Gauss')
#plt.plot(centres(bins_out_normprod_SMICA_unWISE)*299792.458, pixel_scaling_masked(expect_normprod_gaussing, fsky)/1e3,color='k', ls='--', lw=2., label='NPD (Gauss)')
plt.plot(centres(bins_out_normprod_SMICA_unWISE)*299792.458, pixel_scaling_masked(expect_normprod_gaussing_fiducial, fsky)/1e3,color='gray', ls='--', lw=2., label='NPD (Fiducial)')
y1, y2 = plt.ylim()
plt.ylim([0, y2]) 
plt.xlim([-.3,.3])
plt.xlabel(r'$\frac{\mathrm{km}}{\mathrm{s}}$', fontsize=16)
plt.ylabel(r'$N_{\mathrm{pix}}\ \left[\times 10^5\right]$')
plt.title('Velocity reconstruction pixel value distribution')
plt.legend(ncol=2)
plt.xlim([-0.0002*299792.458, 0.0002*299792.458])
plt.ylim([1.350, 5.150])
#plt.ylim([2.25,3.05])
plt.tight_layout()
plt.savefig(outdir+'recon_1pt_many')


redshifts_int = csm.chi_to_z(np.linspace(0,6000,100))
ks_int = np.logspace(-4, 1, redshifts.size)
Pmm_full = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=100., zmax=redshifts.max())
Pmm_linear = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False, kmax=100., zmax=redshifts.max())

Pmm_z = Pmm_full.P(redshifts_int, ks_int, grid=True)  # Returns Pmm_z(z,k)
Pmm_z_linear = Pmm_linear.P(redshifts_int, ks_int, grid=True)  # Returns Pmm_z(z,k)

approx_ells = np.unique(np.geomspace(1,4000).astype(int))

Clmm_full = np.zeros((approx_ells.size, redshifts_int.size, redshifts_int.size))
for lid, ell in enumerate(approx_ells):
	print('Computing ell = %d' % ell)
	bessel_j = jn(ell, ks_int*redshifts_int)
	for zid, z in enumerate(redshifts_int):
		for zpid, zp in enumerate(redshifts_int):
			Clmm_full[lid, zid, zpid] = (2 / np.pi) * np.trapz(ks_int**2 * np.sqrt(Pmm_z[zid, :] * Pmm_z[zpid, :]) * bessel_j[zid] * bessel_j[zpid], x=ks_int)

plt.figure()
plt.loglog(approx_ells, Clmm_full[:,50,50])
plt.savefig(outdir+'model')


plt.figure()
plt.semilogy(redshifts_int, np.sqrt(Pmm_z/Pmm_z_linear[0,:][np.newaxis,:])[:,0])
plt.savefig(outdir+'test')

#growth = results.get_redshift_evolution(ks_int,redshifts_int,vars=['growth'])[:,:,0].transpose()  # Returns growth(k,z) transposed to (z,k)
growth = np.sqrt(Pmm_z_linear/Pmm_z_linear[0,:][np.newaxis,:])  # Indexed (z,k)

plt.figure()
plt.plot(redshifts_int, growth[:,0],label='k=%.3f'%ks_int[0])
plt.plot(redshifts_int, growth[:,20],label='k=%.3f'%ks_int[20])
plt.plot(redshifts_int, growth[:,50],label='k=%.3f'%ks_int[50])
plt.plot(redshifts_int, growth[:,80],label='k=%.3f'%ks_int[80])
plt.legend()
plt.ylabel(r'$D\left( z\right)$')
plt.xlabel(r'$z$')
plt.tight_layout()
plt.savefig(outdir+'growth')

plt.figure()
for k in np.arange(ks_int.size)[::10]:
	plt.loglog(redshifts_int, ks_int[k]**3*Pmm_z_linear[:,k],label='k=%.3f'%ks_int[k])

plt.legend()
plt.ylabel(r'$k^3\,P_{mm}\left( z,k \right)$')
plt.xlabel(r'$z$')
plt.tight_layout()
plt.savefig(outdir+'Pmm')


cmm_k_z = ks_int[np.newaxis,:]**3 * Pmm_z_linear / growth**2  # Indexed to call as (z,k)

plt.figure()
for k in np.arange(ks_int.size)[::10]:
	plt.loglog(redshifts_int, cmm_k_z[:,k],label='k=%.3f'%ks_int[k])

plt.legend()
plt.savefig(outdir+'cmm')



cmm = np.median(Pmm_z_linear*ks_int[np.newaxis,:]**3/growth**2,axis=0)  # cmm(z)
Clmm_approx = np.zeros((approx_ells.size,redshifts_int.size))
for lid, ell in enumerate(approx_ells):
	Clmm_approx[lid,:] = cmm*np.median(growth,axis=1)**2/np.pi/ell

plt.figure()
plt.loglog(approx_ells, csm.Clmm[0,0,approx_ells])
plt.loglog(approx_ells, Clmm_approx[:,zbar_index])
plt.savefig(outdir+'compare_clmm')











ells_cgg = np.unique(np.append(np.geomspace(1,6143,120).astype(int), 6143))
chis_full = np.linspace(chis.min(), chis.max(), 1000)
redshifts_full = csm.cosmology_data.redshift_at_comoving_radial_distance(chis_full)
as_full = 1/(1+redshifts_full)
s_blue = 0.455
omegas = csm.cosmology_data.get_background_densities(as_full,vars=['tot','cdm','nu'])
Om_m = omegas['cdm'] / omegas['tot']
Om_nu = omegas['nu'] / omegas['tot']

H0 = csm.cosmology_data.h_of_z(0)

with open('data/unWISE/blue.txt', 'r') as FILE:
    x = FILE.readlines()

z = np.array([float(l.split(' ')[0]) for l in x])
dndz = np.array([float(l.split(' ')[1]) for l in x])
dndz_interp_full = interp1d(z ,dndz, kind= 'linear', bounds_error=False, fill_value=0)(redshifts_full)

chi_star = csm.cosmology_data.tau0 - csm.cosmology_data.tau_maxvis
g_i_chis_full = np.array([np.trapz(np.nan_to_num(chi * (chis_full[np.where(chis_full>=chi)] - chi) / chis_full[np.where(chis_full>=chi)]) * csm.cosmology_data.h_of_z(redshifts_full[np.where(chis_full>=chi)]) * dndz_interp_full[np.where(chis_full>=chi)], chis_full[np.where(chis_full>=chi)]) for chi in chis_full])
b_sml_p = 0.8 + 1.2*redshifts_full
b_eff = np.trapz(b_sml_p * dndz_interp_full, redshifts_full)
f_z_dN_dz = b_sml_p * dndz_interp_full / b_eff
lensing_window = (5*s_blue - 2) * (3/2) * (Om_m+Om_nu) * H0**2 * (1+redshifts_full) * g_i_chis_full
galaxy_window = csm.get_limber_window('g', chis_full, avg=False)
Cgg_bin = np.zeros(ells_cgg.size)
for l, ell in enumerate(ells_cgg):
    Pmm_full_chi = np.diagonal(np.flip(Pmm_full.P(csm.chi_to_z(chis_full), (ell+0.5)/chis_full[::-1], grid=True), axis=1))
    # for taubin in np.arange(self.nbin):            
    #     chis = np.linspace(self.chi_bin_boundaries[taubin], self.chi_bin_boundaries[taubin+1], 300)
    #     galaxy_window_binned = self.get_limber_window(gtag, chis, avg=False, gwindow_zdep=gwindow_zdep)
    #     taud1_window  = self.get_limber_window('taud', chis, avg=False, gwindow_zdep=gwindow_zdep)        
    #     Pmm_bin1_chi = np.diagonal(np.flip(Pmm_full.P(self.chi_to_z(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))
    #     for taubin2 in np.arange(self.nbin):
    #         chis_binned2 = np.linspace(self.chi_bin_boundaries[taubin2], self.chi_bin_boundaries[taubin2+1], 300)
    #         taud2_window   = self.get_limber_window('taud', chis_binned2, avg=False, gwindow_zdep=gwindow_zdep)
    #         Pmm_bin2_chi = np.diagonal(np.flip(Pmm_full.P(self.chi_to_z(chis_binned2), (ell+0.5)/chis_binned2[::-1], grid=True), axis=1))
    #         if use_m_to_e:
    #             #m_to_e = np.diagonal(np.flip(self.bias_e2(self.chi_to_z(chis), (ell+0.5)/chis[::-1])**2 / self.fe(self.chi_to_z(chis))**.5, axis=1))
    #             m_to_e = np.diagonal(np.flip(self.bias_e2(self.chi_to_z(chis), (ell+0.5)/chis[::-1]), axis=1))
    #             m_to_e2 = np.diagonal(np.flip(self.bias_e2(self.chi_to_z(chis_binned2), (ell+0.5)/chis_binned2[::-1]), axis=1))
    #         else:
    #             m_to_e = m_to_e2 = np.ones(Pmm_bin1_chi.shape)
    #         Pee_binned_chi = np.sqrt(Pmm_bin1_chi*Pmm_bin2_chi) * m_to_e * m_to_e2
    #         Ctt_bin[taubin, taubin2, l] = simps(np.nan_to_num(Pee_binned_chi * taud1_window * taud2_window / (chis*chis_binned2),posinf=0.), np.sqrt(chis*chis_binned2))
    #     Pem_bin1_chi = Pmm_bin1_chi * m_to_e
    #     Ctg_bin[taubin, l] = simps(np.nan_to_num(Pem_bin1_chi *              galaxy_window_binned * taud1_window   / chis**2,posinf=0.), chis)
    #Cgg_bin[l] = simps(np.nan_to_num(Pmm_full_chi * galaxy_window**2 / chis_full**2,posinf=0.), chis_full)
    Cgg_bin[l] = b_eff**2 * simps(np.nan_to_num(1/chis_full**2) * csm.cosmology_data.h_of_z(redshifts_full)**2 * f_z_dN_dz**2 * Pmm_full_chi, chis_full) \
    		   + b_eff    * simps(np.nan_to_num(lensing_window/chis_full**2) * csm.cosmology_data.h_of_z(redshifts_full) * f_z_dN_dz * Pmm_full_chi, chis_full) \
    		   +            simps(np.nan_to_num(lensing_window**2/chis_full**2) * Pmm_full_chi)

Clgg_manual = (interp1d(ells_cgg, Cgg_bin, bounds_error=False, fill_value='extrapolate')(np.arange(6144)) + 9.2e-8) * ngbar**2

# for b1 in np.arange(self.nbin):
#     self.Cltaudg[b1,0,:] = (interp1d(ells_cgg, Ctg_bin[b1,:], bounds_error=False, fill_value='extrapolate')(np.arange(6144))) * ngbar
#     for b2 in np.arange(self.nbin):
#         self.Cltaudtaud[b1,b2,:] = interp1d(ells_cgg, Ctt_bin[b1,b2,:], bounds_error=False, fill_value='extrapolate')(np.arange(6144))    



plt.figure()
plt.plot(csm.Clgg[0,0,:] * 1e5/ ngbar**2, label='Eq. (3.2)')
plt.plot(Clgg_manual * 1e5 / ngbar**2, label='Eq. (6.5)')
plt.ylim([0,0.15])
plt.ylabel(r'$C_\ell^{\mathrm{gg}}\ \left[\times 10^5\right]$')
plt.xlabel(r'$\ell$')
plt.title('Galaxy-galaxy spectrum check against Alex fig.9')
plt.legend()
plt.xlim([100,1000])
plt.savefig(outdir+'clgg_lensing_test')













# Linear Pmm 
print('\n\nCompleted successfully!\n\n')






















T353inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_353-psb_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725
T545inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_545_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725
T857inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_857_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725

Tmap_noCMB_353 = T353inp - SMICAmap_real
Tmap_noCMB_545 = T545inp - SMICAmap_real
Tmap_noCMB_857 = T857inp - SMICAmap_real

T353beam = hp.gauss_beam(fwhm=np.radians(4.94/60), lmax=6143)
T545beam = hp.gauss_beam(fwhm=np.radians(4.83/60), lmax=6143)
T857beam = hp.gauss_beam(fwhm=np.radians(4.64/60), lmax=6143)

###
print('Masking/debeaming maps...')
T353_alms_masked_debeamed = hp.almxfl(hp.map2alm(T353inp*total_mask), 1/T353beam)
T545_alms_masked_debeamed = hp.almxfl(hp.map2alm(T545inp*total_mask), 1/T545beam)
T857_alms_masked_debeamed = hp.almxfl(hp.map2alm(T857inp*total_mask), 1/T857beam)
T353_alms_masked_debeamed_hugemask = hp.almxfl(hp.map2alm(T353inp*hugemask_unwise), 1/T353beam)
T545_alms_masked_debeamed_hugemask = hp.almxfl(hp.map2alm(T545inp*hugemask_unwise), 1/T545beam)
T857_alms_masked_debeamed_hugemask = hp.almxfl(hp.map2alm(T857inp*hugemask_unwise), 1/T857beam)
Tmap_noCMB_353_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_353*total_mask), 1/T353beam)
Tmap_noCMB_545_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_545*total_mask), 1/T545beam)
Tmap_noCMB_857_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_857*total_mask), 1/T857beam)
Tmap_noCMB_353_masked_debeamed_alms_hugemask = hp.almxfl(hp.map2alm(Tmap_noCMB_353*hugemask_unwise), 1/T353beam)
Tmap_noCMB_545_masked_debeamed_alms_hugemask = hp.almxfl(hp.map2alm(Tmap_noCMB_545*hugemask_unwise), 1/T545beam)
Tmap_noCMB_857_masked_debeamed_alms_hugemask = hp.almxfl(hp.map2alm(Tmap_noCMB_857*hugemask_unwise), 1/T857beam)

T353map_real = hp.alm2map(T353_alms_masked_debeamed, nside=2048)
T545map_real = hp.alm2map(T545_alms_masked_debeamed, nside=2048)
T857map_real = hp.alm2map(T857_alms_masked_debeamed, nside=2048)
T353map_real_hugemask = hp.alm2map(T353_alms_masked_debeamed_hugemask, nside=2048)
T545map_real_hugemask = hp.alm2map(T545_alms_masked_debeamed_hugemask, nside=2048)
T857map_real_hugemask = hp.alm2map(T857_alms_masked_debeamed_hugemask, nside=2048)
Tmap_noCMB_353_masked_debeamed = hp.alm2map(Tmap_noCMB_353_masked_debeamed_alms, 2048)
Tmap_noCMB_545_masked_debeamed = hp.alm2map(Tmap_noCMB_545_masked_debeamed_alms, 2048)
Tmap_noCMB_857_masked_debeamed = hp.alm2map(Tmap_noCMB_857_masked_debeamed_alms, 2048)
Tmap_noCMB_353_masked_debeamed_hugemask = hp.alm2map(Tmap_noCMB_353_masked_debeamed_alms_hugemask, 2048)
Tmap_noCMB_545_masked_debeamed_hugemask = hp.alm2map(Tmap_noCMB_545_masked_debeamed_alms_hugemask, 2048)
Tmap_noCMB_857_masked_debeamed_hugemask = hp.alm2map(Tmap_noCMB_857_masked_debeamed_alms_hugemask, 2048)

# Power spectra
ClTT_353 = hp.alm2cl(T353_alms_masked_debeamed) / fsky
ClTT_545 = hp.alm2cl(T545_alms_masked_debeamed) / fsky
ClTT_857 = hp.alm2cl(T857_alms_masked_debeamed) / fsky
ClTT_353_huge = hp.alm2cl(T353_alms_masked_debeamed_hugemask) / fsky_huge
ClTT_545_huge = hp.alm2cl(T545_alms_masked_debeamed_hugemask) / fsky_huge
ClTT_857_huge = hp.alm2cl(T857_alms_masked_debeamed_hugemask) / fsky_huge
ClTT_353[ls.max()+1:] = 0.
ClTT_545[ls.max()+1:] = 0.
ClTT_857[ls.max()+1:] = 0.
ClTT_353_huge[ls.max()+1:] = 0.
ClTT_545_huge[ls.max()+1:] = 0.
ClTT_857_huge[ls.max()+1:] = 0.





print('    Secondary reconstruction: 353GHz x unWISE')
noise_353 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_353.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_353_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_353_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_353 = estim.combine(T353map_real, unWISEmap, total_mask, ClTT_353, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_353,6144), convert_K=False)
_, _, outmap_353foregrounds = estim.combine(Tmap_noCMB_353_masked_debeamed, unWISEmap, total_mask, ClTT_353, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_353,6144), convert_K=False)
_, _, outmap_353_huge = estim.combine(T353map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_353_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_353_huge,6144), convert_K=False)
_, _, outmap_353foregrounds_huge = estim.combine(Tmap_noCMB_353_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_353_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_353_huge,6144), convert_K=False)
recon_Cls_353 = hp.anafast(outmap_353)
recon_Cls_353_foregrounds = hp.anafast(outmap_353foregrounds)
recon_Cls_353_huge = hp.anafast(outmap_353_huge)
recon_Cls_353_foregrounds_huge = hp.anafast(outmap_353foregrounds_huge)
lowpass_output_353 = lowpass(outmap_353)
lowpass_output_353foregrounds = lowpass(outmap_353foregrounds)
lowpass_output_353_huge = lowpass(outmap_353_huge)
lowpass_output_353foregrounds_huge = lowpass(outmap_353foregrounds_huge)
n_353, bins_353 = np.histogram(lowpass_output_353[np.where(total_mask!=0)], bins=nbin_hist)
n_353_foregrounds, bins_353_foregrounds = np.histogram(lowpass_output_353foregrounds[np.where(total_mask!=0)], bins=nbin_hist)
n_353_huge, bins_353_huge = np.histogram(lowpass_output_353_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_353_foregrounds_huge, bins_353_foregrounds_huge = np.histogram(lowpass_output_353foregrounds_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)



print('    Secondary reconstruction: 545GHz x unWISE')
noise_545 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_545.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_545_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_545_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_545 = estim.combine(T545map_real, unWISEmap, total_mask, ClTT_545, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_545,6144), convert_K=False)
_, _, outmap_545foregrounds = estim.combine(Tmap_noCMB_545_masked_debeamed, unWISEmap, total_mask, ClTT_545, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_545,6144), convert_K=False)
_, _, outmap_545_huge = estim.combine(T545map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_545_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_545_huge,6144), convert_K=False)
_, _, outmap_545foregrounds_huge = estim.combine(Tmap_noCMB_545_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_545_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_545_huge,6144), convert_K=False)
recon_Cls_545 = hp.anafast(outmap_545)
recon_Cls_545_foregrounds = hp.anafast(outmap_545foregrounds)
recon_Cls_545_huge = hp.anafast(outmap_545_huge)
recon_Cls_545_foregrounds_huge = hp.anafast(outmap_545foregrounds_huge)
lowpass_output_545 = lowpass(outmap_545)
lowpass_output_545foregrounds = lowpass(outmap_545foregrounds)
lowpass_output_545_huge = lowpass(outmap_545_huge)
lowpass_output_545foregrounds_huge = lowpass(outmap_545foregrounds_huge)
n_545, bins_545 = np.histogram(lowpass_output_545[np.where(total_mask!=0)], bins=nbin_hist)
n_545_foregrounds, bins_545_foregrounds = np.histogram(lowpass_output_545foregrounds[np.where(total_mask!=0)], bins=nbin_hist)
n_545_huge, bins_545_huge = np.histogram(lowpass_output_545_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_545_foregrounds_huge, bins_545_foregrounds_huge = np.histogram(lowpass_output_545foregrounds_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)




print('    Secondary reconstruction: 857GHz x unWISE')
noise_857 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_857.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_857_huge = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_857_huge.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
_, _, outmap_857 = estim.combine(T857map_real, unWISEmap, total_mask, ClTT_857, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_857,6144), convert_K=False)
_, _, outmap_857foregrounds = estim.combine(Tmap_noCMB_857_masked_debeamed, unWISEmap, total_mask, ClTT_857, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_857,6144), convert_K=False)
_, _, outmap_857_huge = estim.combine(T857map_real_hugemask, unWISEmap, hugemask_unwise, ClTT_857_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_857_huge,6144), convert_K=False)
_, _, outmap_857foregrounds_huge = estim.combine(Tmap_noCMB_857_masked_debeamed_hugemask, unWISEmap, hugemask_unwise, ClTT_857_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_857_huge,6144), convert_K=False)
recon_Cls_857 = hp.anafast(outmap_857)
recon_Cls_857_foregrounds = hp.anafast(outmap_857foregrounds)
recon_Cls_857_huge = hp.anafast(outmap_857_huge)
recon_Cls_857_foregrounds_huge = hp.anafast(outmap_857foregrounds_huge)
lowpass_output_857 = lowpass(outmap_857)
lowpass_output_857foregrounds = lowpass(outmap_857foregrounds)
lowpass_output_857_huge = lowpass(outmap_857_huge)
lowpass_output_857foregrounds_huge = lowpass(outmap_857foregrounds_huge)
n_857, bins_857 = np.histogram(lowpass_output_857[np.where(total_mask!=0)], bins=nbin_hist)
n_857_foregrounds, bins_857_foregrounds = np.histogram(lowpass_output_857foregrounds[np.where(total_mask!=0)], bins=nbin_hist)
n_857_huge, bins_857_huge = np.histogram(lowpass_output_857_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_857_foregrounds_huge, bins_857_foregrounds_huge = np.histogram(lowpass_output_857foregrounds_huge[np.where(hugemask_unwise!=0)], bins=nbin_hist)




from astropy import units as u
CIB_353 = fits.open('data/planck_data_testing/CIB/cib_fullmission_353.hpx.fits')[1].data['CIB'] * (u.MJy / u.sr).to(1. * u.K, equivalencies=u.thermodynamic_temperature(353 * u.GHz, T_cmb=csm.cambpars.TCMB*u.K)) / 2.725
CIB_545 = fits.open('data/planck_data_testing/CIB/cib_fullmission_545.hpx.fits')[1].data['CIB'] * (u.MJy / u.sr).to(1. * u.K, equivalencies=u.thermodynamic_temperature(545 * u.GHz, T_cmb=csm.cambpars.TCMB*u.K)) / 2.725
CIB_857 = fits.open('data/planck_data_testing/CIB/cib_fullmission_857.hpx.fits')[1].data['CIB'] * (u.MJy / u.sr).to(1. * u.K, equivalencies=u.thermodynamic_temperature(857 * u.GHz, T_cmb=csm.cambpars.TCMB*u.K)) / 2.725

# 5.0 arcmin fwhm for CIB maps, happens to be equal to SMICAbeam so I'm borrowing the variable. Also CIB maps are masked at least to fsky=0.68 so no fiducial mask comparisons.
Tmap_CIB_353_hugemask = hp.alm2map(hp.almxfl(hp.map2alm(np.nan_to_num(CIB_353*hugemask_unwise)), 1/SMICAbeam), 2048)
Tmap_CIB_545_hugemask = hp.alm2map(hp.almxfl(hp.map2alm(np.nan_to_num(CIB_545*hugemask_unwise)), 1/SMICAbeam), 2048)
Tmap_CIB_857_hugemask = hp.alm2map(hp.almxfl(hp.map2alm(np.nan_to_num(CIB_857*hugemask_unwise)), 1/SMICAbeam), 2048)

_, _, outmap_CIB_353 = estim.combine(Tmap_CIB_353_hugemask, unWISEmap, hugemask_unwise, ClTT_353_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_353_huge,6144), convert_K=False)
_, _, outmap_CIB_545 = estim.combine(Tmap_CIB_545_hugemask, unWISEmap, hugemask_unwise, ClTT_545_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_545_huge,6144), convert_K=False)
_, _, outmap_CIB_857 = estim.combine(Tmap_CIB_857_hugemask, unWISEmap, hugemask_unwise, ClTT_857_huge, csm.Clgg[0,0,:].copy(), Cltaug_at_zbar.copy(), np.repeat(noise_857_huge,6144), convert_K=False)
recon_Cls_353_CIB = hp.anafast(outmap_CIB_353)
recon_Cls_545_CIB = hp.anafast(outmap_CIB_545)
recon_Cls_857_CIB = hp.anafast(outmap_CIB_857)
lowpass_output_CIB_353 = lowpass(outmap_CIB_353)
lowpass_output_CIB_545 = lowpass(outmap_CIB_545)
lowpass_output_CIB_857 = lowpass(outmap_CIB_857)
n_353_CIB, bins_353_CIB = np.histogram(lowpass_output_CIB_353[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_545_CIB, bins_545_CIB = np.histogram(lowpass_output_CIB_545[np.where(hugemask_unwise!=0)], bins=nbin_hist)
n_857_CIB, bins_857_CIB = np.histogram(lowpass_output_CIB_857[np.where(hugemask_unwise!=0)], bins=nbin_hist)



fig, ((ax353, ax545), (ax857, ax857huge)) = plt.subplots(2,2,figsize=(12,12))

ax353.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_353_huge[:ells_plot.size] / fsky), label='353 GHz x unWISE', ls='None', marker='x', zorder=353,color=linecolors[2])
ax353.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_353_foregrounds_huge[:ells_plot.size] / fsky), label='(353 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=353,color=linecolors[3])
ax353.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_353_CIB[:ells_plot.size] / fsky_huge), label='CIB x unWISE', ls='None', marker='x', zorder=353,color=linecolors[1])
ax545.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_545_huge[:ells_plot.size] / fsky), label='545 GHz x unWISE', ls='None', marker='x', zorder=353,color=linecolors[2])
ax545.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_545_foregrounds_huge[:ells_plot.size] / fsky), label='(545 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=545,color=linecolors[3])
ax545.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_545_CIB[:ells_plot.size] / fsky_huge), label='CIB x unWISE', ls='None', marker='x', zorder=545,color=linecolors[1])
ax857.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_857_huge[:ells_plot.size] / fsky), label='857 GHz x unWISE', ls='None', marker='x', zorder=353,color=linecolors[2])
ax857.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_857_foregrounds_huge[:ells_plot.size] / fsky),label='(857 GHz - SMICA) x unWISE', ls='None', marker='x', zorder=857,color=linecolors[3])
ax857.plot(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_857_CIB[:ells_plot.size] / fsky_huge), label='CIB x unWISE', ls='None', marker='x', zorder=857,color=linecolors[1])

for ax in [ax353, ax545, ax857]:
	ax.set_xlim([0, 25])
	#ax.set_ylim([1e-10,3e-6])
	ax.set_yscale('log')
	#ax.errorbar(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls[:ells_plot.size] / fsky), yerr=recon_errs,label='Fiducial reconstruction', ls='None', marker='x', zorder=353,color=linecolors[0], capsize=3,alpha=0.75)
	#ax.errorbar(velocity_compute_ells, clv_windowed+noise_recon, color=linecolors[1],lw=2,label='Windowed velocity',alpha=1*0.8)
	#ax.fill_between(velocity_compute_ells[:20], clv_windowed[:20]-np.std(Clv_windowed_store,axis=0)[:20]+noise_recon, clv_windowed_mm_me[:20]+np.std(Clv_windowed_mm_me_store,axis=0)[:20]+noise_recon, alpha=0.5*0.8, color=linecolors[1])
	#ax.fill_between(ells, (np.mean(velocity_noise_Cls,axis=0)-np.std(velocity_noise_Cls,axis=0))[:ells.size], (np.mean(velocity_noise_Cls,axis=0)+np.std(velocity_noise_Cls,axis=0))[:ells.size],alpha=0.35*0.8,color=linecolors[1])
	handles, labels = ax.get_legend_handles_labels()
	order = [0,1,2]
	ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
	ax.legend()
	ax.set_xlabel(r'$\ell$')
	ax.set_ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
	ax.set_title('Reconstructions at %d GHz, fsky = %.2f' % {ax353 : (353,fsky_huge), ax545 : (545,fsky_huge), ax857 : (857,fsky_huge), ax857huge : (857,fsky_huge)}[ax])

plt.tight_layout()
plt.savefig(outdir+'Foreground_Contributions_high')
