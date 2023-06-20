#### Get frequency maps to match theory noise which we take as correct
#### Scale window velocity, your cltaug is just at zbar so it's times that equation thing and W(v) ^2. W(v) is not normalized to 1, normalization is fixed to choice of zbar

import os
import matplotlib
from matplotlib import pyplot as plt
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
SMICAinp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725  # Remove K_CMB units
unWISEmap = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()

unwise_mask = np.load('data/mask_unWISE_thres_v10.npy')
huge_mask = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL020'],n2r=True)
cltt_measure_mask = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL070'],n2r=True)
hugemask_unwise = huge_mask.astype(np.float32) * unwise_mask

T100inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
T143inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725
T217inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.725

gauss_unwisemask = np.load('data/gauss_reals/sims_mask_unWISE_reconstructions.npz')
gauss_planckmask = np.load('data/gauss_reals/fsky20_sims_mask_unWISE_reconstructions.npz')

Tmap_noCMB_100 = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten()
Tmap_noCMB_143 = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.725
Tmap_noCMB_217 = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.725

synchrotron_100 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_synchrotron-ffp10-skyinbands-100_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()
thermaldust_100 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-100_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()
spinningdust_100 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_spindust-ffp10-skyinbands-100_2048_R3.00_full.fits')[1].data['UNKNOWN1'].flatten()
freefree_100 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_freefree-ffp10-skyinbands-100_2048_R3.00_full.fits')[1].data['UNKNOWN1'].flatten()

synchrotron_143 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_synchrotron-ffp10-skyinbands-143_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.725
thermaldust_143 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-143_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.725
spinningdust_143 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_spindust-ffp10-skyinbands-143_2048_R3.00_full.fits')[1].data['UNKNOWN1'].flatten()  / 2.725
freefree_143 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_freefree-ffp10-skyinbands-143_2048_R3.00_full.fits')[1].data['UNKNOWN1'].flatten()  / 2.725

synchrotron_217 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.725
thermaldust_217 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-217_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.725
spinningdust_217 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_spindust-ffp10-skyinbands-217_2048_R3.00_full.fits')[1].data['UNKNOWN1'].flatten()  / 2.725
freefree_217 = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits')[1].data['UNKNOWN1'].flatten()  / 2.725

####
# Map-based values
fsky = np.where(unwise_mask!=0)[0].size / unwise_mask.size
fsky_cltt = np.where(cltt_measure_mask!=0)[0].size / cltt_measure_mask.size
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
SMICAmap_real = hp.alm2map(hp.almxfl(hp.map2alm(SMICAinp), 1/SMICAbeam), 2048)  # De-beamed unitless SMICA map
SMICAmap_premask = hp.alm2map(hp.almxfl(hp.map2alm(SMICAinp*unwise_mask), 1/SMICAbeam), 2048)  # De-beamed unitless SMICA map
SMICAmap_gauss = hp.alm2map(hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(SMICAinp*cltt_measure_mask)/fsky_cltt, 2048)), 1/SMICAbeam), 2048)

T100_alms_masked_debeamed = hp.almxfl(hp.map2alm(T100inp*unwise_mask), 1/T100beam)
T143_alms_masked_debeamed = hp.almxfl(hp.map2alm(T143inp*unwise_mask), 1/T143beam)
T217_alms_masked_debeamed = hp.almxfl(hp.map2alm(T217inp*unwise_mask), 1/T217beam)
T100map_real = hp.alm2map(T100_alms_masked_debeamed, nside=2048)
T143map_real = hp.alm2map(T143_alms_masked_debeamed, nside=2048)
T217map_real = hp.alm2map(T217_alms_masked_debeamed, nside=2048)

Tmap_noCMB_100_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_100*unwise_mask), 1/T100beam)
Tmap_noCMB_143_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_143*unwise_mask), 1/T143beam)
Tmap_noCMB_217_masked_debeamed_alms = hp.almxfl(hp.map2alm(Tmap_noCMB_217*unwise_mask), 1/T217beam)
Tmap_noCMB_100_masked_debeamed = hp.alm2map(Tmap_noCMB_100_masked_debeamed_alms, 2048)
Tmap_noCMB_143_masked_debeamed = hp.alm2map(Tmap_noCMB_143_masked_debeamed_alms, 2048)
Tmap_noCMB_217_masked_debeamed = hp.alm2map(Tmap_noCMB_217_masked_debeamed_alms, 2048)

Tmap_synchrotron_100 = hp.alm2map(hp.almxfl(hp.map2alm(synchrotron_100*unwise_mask), 1/T100beam), 2048)
Tmap_thermaldust_100 = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_100*unwise_mask), 1/T100beam), 2048)
Tmap_spinningdust_100 = hp.alm2map(hp.almxfl(hp.map2alm(spinningdust_100*unwise_mask), 1/T100beam), 2048)
Tmap_freefree_100 = hp.alm2map(hp.almxfl(hp.map2alm(freefree_100*unwise_mask), 1/T100beam), 2048)

Tmap_synchrotron_143 = hp.alm2map(hp.almxfl(hp.map2alm(synchrotron_143*unwise_mask), 1/T143beam), 2048)
Tmap_thermaldust_143 = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_143*unwise_mask), 1/T143beam), 2048)
Tmap_spinningdust_143 = hp.alm2map(hp.almxfl(hp.map2alm(spinningdust_143*unwise_mask), 1/T143beam), 2048)
Tmap_freefree_143 = hp.alm2map(hp.almxfl(hp.map2alm(freefree_143*unwise_mask), 1/T143beam), 2048)

Tmap_synchrotron_217 = hp.alm2map(hp.almxfl(hp.map2alm(synchrotron_217*unwise_mask), 1/T217beam), 2048)
Tmap_thermaldust_217 = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_217*unwise_mask), 1/T217beam), 2048)
Tmap_spinningdust_217 = hp.alm2map(hp.almxfl(hp.map2alm(spinningdust_217*unwise_mask), 1/T217beam), 2048)
Tmap_freefree_217 = hp.alm2map(hp.almxfl(hp.map2alm(freefree_217*unwise_mask), 1/T217beam), 2048)

Tmap_100_hugemasked_debeamed = hp.alm2map(hp.almxfl(hp.map2alm(T100inp*hugemask_unwise), 1/T100beam), 2048)
Tmap_noCMB_100_hugemasked_debeamed = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_noCMB_100*hugemask_unwise), 1/T100beam), 2048)
Tmap_synchrotron_100hugemask = hp.alm2map(hp.almxfl(hp.map2alm(synchrotron_100*hugemask_unwise), 1/T100beam), 2048)
Tmap_thermaldust_100hugemask = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_100*hugemask_unwise), 1/T100beam), 2048)
Tmap_spinningdust_100hugemask = hp.alm2map(hp.almxfl(hp.map2alm(spinningdust_100*hugemask_unwise), 1/T100beam), 2048)
Tmap_freefree_100hugemask = hp.alm2map(hp.almxfl(hp.map2alm(freefree_100*hugemask_unwise), 1/T100beam), 2048)

Tmap_143_hugemasked_debeamed = hp.alm2map(hp.almxfl(hp.map2alm(T143inp*hugemask_unwise), 1/T143beam), 2048)
Tmap_noCMB_143_hugemasked_debeamed = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_noCMB_143*hugemask_unwise), 1/T143beam), 2048)
Tmap_synchrotron_143hugemask = hp.alm2map(hp.almxfl(hp.map2alm(synchrotron_143*hugemask_unwise), 1/T143beam), 2048)
Tmap_thermaldust_143hugemask = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_143*hugemask_unwise), 1/T143beam), 2048)
Tmap_spinningdust_143hugemask = hp.alm2map(hp.almxfl(hp.map2alm(spinningdust_143*hugemask_unwise), 1/T143beam), 2048)
Tmap_freefree_143hugemask = hp.alm2map(hp.almxfl(hp.map2alm(freefree_143*hugemask_unwise), 1/T143beam), 2048)

Tmap_217_hugemasked_debeamed = hp.alm2map(hp.almxfl(hp.map2alm(T217inp*hugemask_unwise), 1/T217beam), 2048)
Tmap_noCMB_217_hugemasked_debeamed = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_noCMB_217*hugemask_unwise), 1/T217beam), 2048)
Tmap_synchrotron_217hugemask = hp.alm2map(hp.almxfl(hp.map2alm(synchrotron_217*hugemask_unwise), 1/T217beam), 2048)
Tmap_thermaldust_217hugemask = hp.alm2map(hp.almxfl(hp.map2alm(thermaldust_217*hugemask_unwise), 1/T217beam), 2048)
Tmap_spinningdust_217hugemask = hp.alm2map(hp.almxfl(hp.map2alm(spinningdust_217*hugemask_unwise), 1/T217beam), 2048)
Tmap_freefree_217hugemask = hp.alm2map(hp.almxfl(hp.map2alm(freefree_217*hugemask_unwise), 1/T217beam), 2048)

# Power spectra
ClTT = hp.anafast(SMICAmap_real)
ClTT_premask = hp.anafast(SMICAmap_premask) / fsky
ClTT_100 = hp.alm2cl(T100_alms_masked_debeamed) / fsky
ClTT_143 = hp.alm2cl(T143_alms_masked_debeamed) / fsky
ClTT_217 = hp.alm2cl(T217_alms_masked_debeamed) / fsky
ClTT_100foregrounds = hp.alm2cl(Tmap_noCMB_100_masked_debeamed_alms) / fsky
ClTT_143foregrounds = hp.alm2cl(Tmap_noCMB_143_masked_debeamed_alms) / fsky
ClTT_217foregrounds = hp.alm2cl(Tmap_noCMB_217_masked_debeamed_alms) / fsky
ClTT_100hugemask = hp.alm2cl(hp.almxfl(hp.map2alm(T100inp*hugemask_unwise), 1/T100beam)) / fsky_huge
ClTT_143hugemask = hp.alm2cl(hp.almxfl(hp.map2alm(T143inp*hugemask_unwise), 1/T143beam)) / fsky_huge
ClTT_217hugemask = hp.alm2cl(hp.almxfl(hp.map2alm(T217inp*hugemask_unwise), 1/T217beam)) / fsky_huge
# Zero Cls above our lmax
ClTT[ls.max()+1:] = 0.  
ClTT_premask[ls.max()+1:] = 0.
ClTT_100[ls.max()+1:] = 0.
ClTT_143[ls.max()+1:] = 0.
ClTT_217[ls.max()+1:] = 0.
ClTT_100foregrounds[ls.max()+1:] = 0.
ClTT_143foregrounds[ls.max()+1:] = 0.
ClTT_217foregrounds[ls.max()+1:] = 0.
ClTT_100hugemask[ls.max()+1:] = 0.
ClTT_143hugemask[ls.max()+1:] = 0.
ClTT_217hugemask[ls.max()+1:] = 0.

### Cosmology
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
csm.compute_Cls(ngbar=ngbar)  # These are the integrated Cls over the entire bin

fullspectrum_ls = np.unique(np.append(np.geomspace(1,6144-1,200).astype(int), 6144-1))
# Now we compute the same Cls we just did, but at zbar instead of over the entire bin
Pmms = np.zeros((chis.size,fullspectrum_ls.size))
Pmm_full = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=100., zmax=redshifts.max())
for l, ell in enumerate(fullspectrum_ls):  # Do limber approximation: P(z,k) -> P(z, (ell+0.5)/chi )
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(csm.cosmology_data.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))

Pem_bin1_chi = np.zeros((fullspectrum_ls.size))
for l, ell in enumerate(fullspectrum_ls):
	Pem_bin1_chi[l] = Pmms[zbar_index,l] * np.diagonal(np.flip(csm.bias_e2(csm.chi_to_z(chis), (ell+0.5)/chis[::-1]), axis=1))[zbar_index]  # Convert Pmm to Pem

galaxy_window_binned = csm.get_limber_window('g', chis, avg=False)[zbar_index]  # Units of 1/Mpc
taud1_window  = csm.get_limber_window('taud', chis, avg=False)[zbar_index]  # Units of 1/Mpc
chibar = csm.z_to_chi(redshifts[zbar_index])

# Manual Cls at zbar. Same form as in csm.compute_Cls but instead of integrating over chi we multiply the integrand by deltachi, where the integrand is evaluated at zbar
Cltaug_at_zbar = interp1d(fullspectrum_ls, (Pem_bin1_chi * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

######## VELOCITY
PKv = camb.get_matter_power_interpolator(pars,hubble_units=False, k_hunit=False, var1='v_newtonian_cdm',var2='v_newtonian_cdm')

h = results.h_of_z(redshifts)

chibar = csm.z_to_chi(redshifts[zbar_index])

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
velocity_compute_ells = np.append(np.unique(np.geomspace(1,30,10).astype(int)),100)
clv = np.zeros((velocity_compute_ells.shape[0],redshifts.shape[0],redshifts.shape[0]))
for l in range(velocity_compute_ells.shape[0]):
	print('l = %d' % velocity_compute_ells[l])
	for z1 in range(redshifts.shape[0]):
		for z2 in range(redshifts.shape[0]):
			integrand_k = scipy.special.spherical_jn(velocity_compute_ells[l],ks*chis[z1])*scipy.special.spherical_jn(velocity_compute_ells[l],ks*chis[z2]) * (h[z1]/(1+redshifts[z1]))*(h[z2]/(1+redshifts[z2])) * np.sqrt(PKv.P(redshifts[z1],ks)*PKv.P(redshifts[z2],ks))
			clv[l,z1,z2] = (2./np.pi)*np.trapz(integrand_k,ks)


window_v_chi1 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )
window_v_chi2 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_me_me  )

window_v_mm_me_chi1 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )
window_v_mm_me_chi2 = np.nan_to_num(  ( 1/csm.bin_width ) * ( chibar**2 / chis**2 ) * ( window_g / window_g[zbar_index] ) * ( (1+csm.chi_to_z(chis))**2 / (1+csm.chi_to_z(chibar))**2 ) * ratio_mm_me  )


clv_windowed = np.zeros(velocity_compute_ells.size)
clv_windowed_mm_me = np.zeros(velocity_compute_ells.size)
for i in np.arange(velocity_compute_ells.size):
	clv_windowed[i] = np.trapz(window_v_chi1*np.trapz(window_v_chi2*clv[i,:,:], chis,axis=1), chis)
	clv_windowed_mm_me[i] = np.trapz(window_v_mm_me_chi1*np.trapz(window_v_mm_me_chi2*clv[i,:,:], chis,axis=1), chis)

clv_windowed_interp = interp1d(velocity_compute_ells,clv_windowed,fill_value=0.,bounds_error=False)(np.arange(6144))

#########################
#### Noise
noise_SMICA = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_SMICApremask = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_premask.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())

noise_100 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_100.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_143 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_143.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_217 = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())

noise_100hugemask = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_100hugemask.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_143hugemask = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_143hugemask.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())
noise_217hugemask = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT_217hugemask.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())

#### Reconstructions
lowpass = lambda MAP : hp.alm2map(hp.almxfl(hp.map2alm(MAP), [0 if l > 50 else 1 for l in np.arange(6144)]), 2048)
centres = lambda BIN : (BIN[:-1]+BIN[1:]) / 2

_, _, outmap_gaussreals = estim.combine(SMICAmap_gauss, unWISEmap, unwise_mask,  ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA,6144), convert_K=False)
_, _, outmap_SMICA = estim.combine(SMICAmap_real, unWISEmap, unwise_mask, ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA,6144), convert_K=False)
_, _, outmap_SMICApremask = estim.combine(SMICAmap_premask, unWISEmap, unwise_mask, ClTT_premask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICApremask,6144), convert_K=False)
_, _, outmap_100 = estim.combine(T100map_real, unWISEmap, unwise_mask, ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_143 = estim.combine(T143map_real, unWISEmap, unwise_mask, ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_217 = estim.combine(T217map_real, unWISEmap, unwise_mask, ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_100foregrounds = estim.combine(Tmap_noCMB_100_masked_debeamed, unWISEmap, unwise_mask, ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_143foregrounds = estim.combine(Tmap_noCMB_143_masked_debeamed, unWISEmap, unwise_mask, ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_217foregrounds = estim.combine(Tmap_noCMB_217_masked_debeamed, unWISEmap, unwise_mask, ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_synchrotron_100 = estim.combine(Tmap_synchrotron_100, unWISEmap, unwise_mask, ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_thermaldust_100 = estim.combine(Tmap_thermaldust_100, unWISEmap, unwise_mask, ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_spinningdust_100 = estim.combine(Tmap_spinningdust_100, unWISEmap, unwise_mask, ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_freefree_100 = estim.combine(Tmap_freefree_100, unWISEmap, unwise_mask, ClTT_100, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100,6144), convert_K=True)
_, _, outmap_synchrotron_143 = estim.combine(Tmap_synchrotron_143, unWISEmap, unwise_mask, ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_thermaldust_143 = estim.combine(Tmap_thermaldust_143, unWISEmap, unwise_mask, ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_spinningdust_143 = estim.combine(Tmap_spinningdust_143, unWISEmap, unwise_mask, ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_freefree_143 = estim.combine(Tmap_freefree_143, unWISEmap, unwise_mask, ClTT_143, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143,6144), convert_K=False)
_, _, outmap_synchrotron_217 = estim.combine(Tmap_synchrotron_217, unWISEmap, unwise_mask, ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_thermaldust_217 = estim.combine(Tmap_thermaldust_217, unWISEmap, unwise_mask, ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_spinningdust_217 = estim.combine(Tmap_spinningdust_217, unWISEmap, unwise_mask, ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_freefree_217 = estim.combine(Tmap_freefree_217, unWISEmap, unwise_mask, ClTT_217, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217,6144), convert_K=False)
_, _, outmap_100_hugemask = estim.combine(Tmap_100_hugemasked_debeamed, unWISEmap, hugemask_unwise, ClTT_100hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100hugemask,6144), convert_K=True)
_, _, outmap_100foregrounds_hugemask = estim.combine(Tmap_noCMB_100_hugemasked_debeamed, unWISEmap, hugemask_unwise, ClTT_100hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100hugemask,6144), convert_K=True)
_, _, outmap_synchrotron_100hugemask = estim.combine(Tmap_synchrotron_100hugemask, unWISEmap, hugemask_unwise, ClTT_100hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100hugemask,6144), convert_K=True)
_, _, outmap_thermaldust_100hugemask = estim.combine(Tmap_thermaldust_100hugemask, unWISEmap, hugemask_unwise, ClTT_100hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100hugemask,6144), convert_K=True)
_, _, outmap_spinningdust_100hugemask = estim.combine(Tmap_spinningdust_100hugemask, unWISEmap, hugemask_unwise, ClTT_100hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100hugemask,6144), convert_K=True)
_, _, outmap_freefree_100hugemask = estim.combine(Tmap_freefree_100hugemask, unWISEmap, hugemask_unwise, ClTT_100hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_100hugemask,6144), convert_K=True)
_, _, outmap_143_hugemask = estim.combine(Tmap_143_hugemasked_debeamed, unWISEmap, hugemask_unwise, ClTT_143hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143hugemask,6144))
_, _, outmap_143foregrounds_hugemask = estim.combine(Tmap_noCMB_143_hugemasked_debeamed, unWISEmap, hugemask_unwise, ClTT_143hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143hugemask,6144))
_, _, outmap_synchrotron_143hugemask = estim.combine(Tmap_synchrotron_143hugemask, unWISEmap, hugemask_unwise, ClTT_143hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143hugemask,6144))
_, _, outmap_thermaldust_143hugemask = estim.combine(Tmap_thermaldust_143hugemask, unWISEmap, hugemask_unwise, ClTT_143hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143hugemask,6144))
_, _, outmap_spinningdust_143hugemask = estim.combine(Tmap_spinningdust_143hugemask, unWISEmap, hugemask_unwise, ClTT_143hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143hugemask,6144))
_, _, outmap_freefree_143hugemask = estim.combine(Tmap_freefree_143hugemask, unWISEmap, hugemask_unwise, ClTT_143hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_143hugemask,6144))
_, _, outmap_217_hugemask = estim.combine(Tmap_217_hugemasked_debeamed, unWISEmap, hugemask_unwise, ClTT_217hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217hugemask,6144))
_, _, outmap_217foregrounds_hugemask = estim.combine(Tmap_noCMB_217_hugemasked_debeamed, unWISEmap, hugemask_unwise, ClTT_217hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217hugemask,6144))
_, _, outmap_synchrotron_217hugemask = estim.combine(Tmap_synchrotron_217hugemask, unWISEmap, hugemask_unwise, ClTT_217hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217hugemask,6144))
_, _, outmap_thermaldust_217hugemask = estim.combine(Tmap_thermaldust_217hugemask, unWISEmap, hugemask_unwise, ClTT_217hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217hugemask,6144))
_, _, outmap_spinningdust_217hugemask = estim.combine(Tmap_spinningdust_217hugemask, unWISEmap, hugemask_unwise, ClTT_217hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217hugemask,6144))
_, _, outmap_freefree_217hugemask = estim.combine(Tmap_freefree_217hugemask, unWISEmap, hugemask_unwise, ClTT_217hugemask, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_217hugemask,6144))
_, _, outmap_hugemask = estim.combine(SMICAmap_real, unWISEmap, hugemask_unwise,  ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA,6144), convert_K=False)

# Power spectra of reconstructions
recon_Cls_gaussreals = hp.anafast(outmap_gaussreals)
recon_Cls_SMICA = hp.anafast(outmap_SMICA)
recon_Cls_SMICApremask = hp.anafast(outmap_SMICApremask)
recon_Cls_100 = hp.anafast(outmap_100)
recon_Cls_143 = hp.anafast(outmap_143)
recon_Cls_217 = hp.anafast(outmap_217)
recon_Cls_100foregrounds = hp.anafast(outmap_100foregrounds)
recon_Cls_143foregrounds = hp.anafast(outmap_143foregrounds)
recon_Cls_217foregrounds = hp.anafast(outmap_217foregrounds)
recon_Cls_synchrotron_100 = hp.anafast(outmap_synchrotron_100)
recon_Cls_thermaldust_100 = hp.anafast(outmap_thermaldust_100)
recon_Cls_spinningdust_100 = hp.anafast(outmap_spinningdust_100)
recon_Cls_freefree_100 = hp.anafast(outmap_freefree_100)
recon_Cls_synchrotron_143 = hp.anafast(outmap_synchrotron_143)
recon_Cls_thermaldust_143 = hp.anafast(outmap_thermaldust_143)
recon_Cls_spinningdust_143 = hp.anafast(outmap_spinningdust_143)
recon_Cls_freefree_143 = hp.anafast(outmap_freefree_143)
recon_Cls_synchrotron_217 = hp.anafast(outmap_synchrotron_217)
recon_Cls_thermaldust_217 = hp.anafast(outmap_thermaldust_217)
recon_Cls_spinningdust_217 = hp.anafast(outmap_spinningdust_217)
recon_Cls_freefree_217 = hp.anafast(outmap_freefree_217)
recon_Cls_100hugemask = hp.anafast(outmap_100_hugemask)
recon_Cls_100hugemaskforegrounds = hp.anafast(outmap_100foregrounds_hugemask)
recon_Cls_synchrotron_100hugemask = hp.anafast(outmap_synchrotron_100hugemask)
recon_Cls_thermaldust_100hugemask = hp.anafast(outmap_thermaldust_100hugemask)
recon_Cls_spinningdust_100hugemask = hp.anafast(outmap_spinningdust_100hugemask)
recon_Cls_freefree_100hugemask = hp.anafast(outmap_freefree_100hugemask)
recon_Cls_143hugemask = hp.anafast(outmap_143_hugemask)
recon_Cls_143hugemaskforegrounds = hp.anafast(outmap_143foregrounds_hugemask)
recon_Cls_synchrotron_143hugemask = hp.anafast(outmap_synchrotron_143hugemask)
recon_Cls_thermaldust_143hugemask = hp.anafast(outmap_thermaldust_143hugemask)
recon_Cls_spinningdust_143hugemask = hp.anafast(outmap_spinningdust_143hugemask)
recon_Cls_freefree_143hugemask = hp.anafast(outmap_freefree_143hugemask)
recon_Cls_217hugemask = hp.anafast(outmap_217_hugemask)
recon_Cls_217hugemaskforegrounds = hp.anafast(outmap_217foregrounds_hugemask)
recon_Cls_synchrotron_217hugemask = hp.anafast(outmap_synchrotron_217hugemask)
recon_Cls_thermaldust_217hugemask = hp.anafast(outmap_thermaldust_217hugemask)
recon_Cls_spinningdust_217hugemask = hp.anafast(outmap_spinningdust_217hugemask)
recon_Cls_freefree_217hugemask = hp.anafast(outmap_freefree_217hugemask)

# Lowpass filter reconstructions for histograms
lowpass_output = lowpass(outmap_SMICA)
lowpass_gauss  = lowpass(outmap_gaussreals)
lowpass_100 = lowpass(outmap_100foregrounds)
lowpass_143 = lowpass(outmap_143foregrounds)
lowpass_217 = lowpass(outmap_217foregrounds)
lowpass_output_hugemask = lowpass(outmap_hugemask)
lowpass_100hugemask = lowpass(outmap_100foregrounds_hugemask)
lowpass_143hugemask = lowpass(outmap_143foregrounds_hugemask)
lowpass_217hugemask = lowpass(outmap_217foregrounds_hugemask)
# Compute 1-pt statistics
n, bins   = np.histogram(lowpass_output[np.where(unwise_mask!=0)], bins=50)
ngauss, _ = np.histogram(lowpass_gauss[np.where(unwise_mask!=0)],  bins=bins)
n100, _ = np.histogram(lowpass_100[np.where(unwise_mask!=0)], bins=bins)
n143, _ = np.histogram(lowpass_143[np.where(unwise_mask!=0)], bins=bins)
n217, _ = np.histogram(lowpass_217[np.where(unwise_mask!=0)], bins=bins)
n_hugemask, bins_hugemask   = np.histogram(lowpass_output_hugemask[np.where(hugemask_unwise!=0)], bins=np.linspace(-1000,1000,250)/299792.458)
n100huge, _ = np.histogram(lowpass_100hugemask[np.where(hugemask_unwise!=0)], bins=bins_hugemask)
n143huge, _ = np.histogram(lowpass_143hugemask[np.where(hugemask_unwise!=0)], bins=bins_hugemask)
n217huge, _ = np.histogram(lowpass_217hugemask[np.where(hugemask_unwise!=0)], bins=bins_hugemask)

### Fitting
def gaussian(x, a, b, c):
	return a*np.exp(-(x-b)**2/(2*c**2))

popt100, _ = curve_fit(gaussian, centres(bins_hugemask), n100huge, p0=[n100huge.max(),0.00,1e-7**.5])
popt143, _ = curve_fit(gaussian, centres(bins_hugemask), n143huge, p0=[n143huge.max(),0.00,1e-7**.5])
popt217, _ = curve_fit(gaussian, centres(bins_hugemask), n217huge, p0=[n217huge.max(),0.00,1e-7**.5])

test100 = hp.synfast([noise_100hugemask/2.725**2 for i in np.arange(6144)], 2048)
test143 = hp.synfast([noise_143hugemask for i in np.arange(6144)], 2048)
test217 = hp.synfast([noise_217hugemask for i in np.arange(6144)], 2048)

testing100 = lowpass(test100)
testing143 = lowpass(test143)
testing217 = lowpass(test217)

pixvar100 = np.var(testing100[np.where(hugemask_unwise!=0)]) * fsky_huge
pixvar143 = np.var(testing143[np.where(hugemask_unwise!=0)]) * fsky_huge
pixvar217 = np.var(testing217[np.where(hugemask_unwise!=0)]) * fsky_huge

#### Plots
premask_fudge = np.mean(recon_Cls_SMICApremask[20:100])/fsky/noise_SMICApremask  # Noise floor increase due to premask, set by SMICA where we can postmask and find the minimum factor. Maps with foregrounds will be worse but this is the best we can justify waving away
nells_bands = 5  # One that gives us an integer ell
bandpowers_shape = (ls.max() // nells_bands, nells_bands)
bandpowers = lambda spectrum : np.reshape(spectrum[1:ls.max()-3], bandpowers_shape).mean(axis=1)

ells = np.arange(ls.max()+1)

plt.figure()
plt.semilogy(velocity_compute_ells[:20],clv[:20,zbar_index,zbar_index], label='True velocity')
plt.semilogy(velocity_compute_ells[:20],clv_windowed[:20], label='Windowed velocity')
plt.semilogy(velocity_compute_ells[:20],clv_windowed_mm_me[:20], label=r'Windowed velocity if $P_{\mathrm{em}}=P_{\mathrm{mm}}$')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$', rotation=0.)
plt.legend()
plt.title('Velocity')
plt.savefig(outdir+'velocities2')



plt.figure()

l1, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_SMICA[:ells.size] / fsky), label='Planck x unWISE Reconstruction', ls='None', marker='^', zorder=100)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_SMICA[:ells.size] / fsky)[1:ls.max()], c=l1.get_c(), ls='--', alpha=0.5)
plt.semilogy(ells[1:], np.repeat(noise_SMICA,ells.size-1), c='k',label='Predicted Noise', ls='--', zorder=10, lw=2)
plt.xlim([0, 40])


mean_unwisemask = np.mean(gauss_unwisemask['Cl_Tgauss_greal'],axis=0) / fsky
std_unwisemask  = np.std( gauss_unwisemask['Cl_Tgauss_greal'],axis=0) / fsky
err_min_unwisemask = mean_unwisemask - std_unwisemask
err_max_unwisemask = mean_unwisemask + std_unwisemask
maxerr_min_unwisemask = mean_unwisemask - 5*std_unwisemask
maxerr_max_unwisemask = mean_unwisemask + 5*std_unwisemask

plt.fill_between(ells[1:], err_min_unwisemask[1:ells.size], err_max_unwisemask[1:ells.size],label='100x Gaussian Reconstructions',alpha=0.35)
plt.fill_between(ells[1:], err_min_unwisemask[1:ells.size], maxerr_max_unwisemask[1:ells.size],alpha=0.15,color='#1f77b4')

mean_planckmask = np.mean(gauss_planckmask['Cl_Tgauss_greal'],axis=0) / fsky_huge
std_planckmask  = np.std( gauss_planckmask['Cl_Tgauss_greal'],axis=0) / fsky_huge
err_min_planckmask = mean_planckmask - std_planckmask
err_max_planckmask = mean_planckmask + std_planckmask
y1,y2=plt.ylim()

plt.fill_between(ells, interp1d(velocity_compute_ells, clv_windowed,bounds_error=False,fill_value=0.)(np.arange(ells.size)), interp1d(velocity_compute_ells, clv_windowed_mm_me,bounds_error=False,fill_value=0.)(np.arange(ells.size)), label='Velocity Signal')

plt.ylim([y1,y2])
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
plt.title('Planck x unWISE Reconstruction')

plt.tight_layout()
plt.savefig(outdir+'signal_noise_gauss.png')



plt.figure()
l1, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_SMICApremask[:ells.size] / fsky), label='Premask Reconstruction / fsky', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_SMICApremask[:ells.size] / fsky)[1:ls.max()], c=l1.get_c(), ls='--', alpha=0.5)
l5, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_SMICA[:ells.size] / fsky), label='Postmask Reconstruction / fsky', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_SMICA[:ells.size] / fsky)[1:ls.max()], c=l5.get_c(), ls='--', alpha=0.5)
#l7, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_SMICApostmask_20[:ells.size] / fsky20), label='Postmask fsky=0.2', ls='None', marker='^', zorder=11)
#plt.semilogy(np.arange(1,ls.max()), (recon_Cls_SMICApostmask_20[:ells.size] / fsky20)[1:ls.max()], c=l7.get_c(), ls='--', alpha=0.5)
#l6, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_SMICApostmask_20union[:ells.size] / fskyunion), label='Postmask fsky=0.18', ls='None', marker='^', zorder=11)
#plt.semilogy(np.arange(1,ls.max()), (recon_Cls_SMICApostmask_20union[:ells.size] / fskyunion)[1:ls.max()], c=l6.get_c(), ls='--', alpha=0.5)
#l2, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
#plt.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
#plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_SMICApremask,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
plt.semilogy(np.arange(ls.max()+1), np.repeat(noise_SMICApremask,ells.size), c=l1.get_c(),label='Premask Theory Noise', ls='--', zorder=0, lw=2)
plt.semilogy(np.arange(ls.max()+1), np.repeat(noise_SMICA,ells.size), c=l5.get_c(),label='Postmask Theory Noise', ls='--', zorder=0, lw=2)
_ = plt.xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
plt.xlim([0, 40])
#plt.ylim([1e-12, 1e-5])
plt.ylim([3e-10, 3e-7])
plt.title('Reconstruction for pre-masked SMICA input')
plt.ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
plt.xlabel(r'$\ell$',fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig(outdir+'signal_noise_reconstruction_premask-SMICA_bandpowers.png')


plt.figure()
l1, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217foregrounds[:ells.size] / fsky) / premask_fudge, label='Reconstruction / fsky', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_217foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l1.get_c(), ls='--', alpha=0.5)
l2, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
plt.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_217,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
_ = plt.xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
plt.xlim([0, 100])
plt.ylim([1e-12, 1e-5])
plt.title('Reconstruction and Signal for 217GHz SMICA-subtracted T input')
plt.ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
plt.xlabel(r'$\ell$',fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig(outdir+'T217_noCMB_bandpowers.png')




plt.figure()
l0, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217[:ells.size] / fsky) / premask_fudge, label='217 GHz Map', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_217[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l0.get_c(), ls='--', alpha=0.5)
l1, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217foregrounds[:ells.size] / fsky) / premask_fudge, label='217 GHz no CMB', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_217foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l1.get_c(), ls='--', alpha=0.5)
l2, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
plt.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_217,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
_ = plt.xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
plt.xlim([0, 100])
plt.ylim([1e-12, 1e-5])
plt.title('Reconstruction and Signal for 217GHz')
plt.ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
plt.xlabel(r'$\ell$',fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig(outdir+'T217_all_bandpowers.png')



plt.figure()
l0, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143[:ells.size] / fsky) / premask_fudge, label='143 GHz Map', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_143[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l0.get_c(), ls='--', alpha=0.5)
l1, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143foregrounds[:ells.size] / fsky) / premask_fudge, label='143 GHz no CMB', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_143foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l1.get_c(), ls='--', alpha=0.5)
l2, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
plt.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_143,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
_ = plt.xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
plt.xlim([0, 100])
plt.ylim([1e-12, 1e-5])
plt.title('Reconstruction and Signal for 143GHz')
plt.ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
plt.xlabel(r'$\ell$',fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig(outdir+'T143_all_bandpowers.png')


plt.figure()
l0, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100[:ells.size] / fsky) / premask_fudge, label='100 GHz Map', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_100[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l0.get_c(), ls='--', alpha=0.5)
l1, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100foregrounds[:ells.size] / fsky) / premask_fudge, label='100 GHz no CMB', ls='None', marker='^', zorder=10)
plt.semilogy(np.arange(1,ls.max()), (recon_Cls_100foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l1.get_c(), ls='--', alpha=0.5)
l2, = plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
plt.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
plt.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_100/2.725**2,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
_ = plt.xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
plt.xlim([0, 100])
plt.ylim([1e-12, 1e-5])
plt.title('Reconstruction and Signal for 100GHz')
plt.ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
plt.xlabel(r'$\ell$',fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig(outdir+'T100_all_bandpowers.png')




fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,6))
# 100 GHz
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_100[:ells.size]              / fsky)[1:ls.max()] / premask_fudge, label='Full Sky')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_100foregrounds[:ells.size]   / fsky)[1:ls.max()] / premask_fudge, label='SMICA Subtracted Sky')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_thermaldust_100[:ells.size]  / fsky)[1:ls.max()] / premask_fudge, label='Thermal Dust Sim')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_freefree_100[:ells.size]     / fsky)[1:ls.max()] / premask_fudge, label='Free-Free Emission Sim')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_synchrotron_100[:ells.size]  / fsky)[1:ls.max()] / premask_fudge, label='Synchrotron Sim')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_spinningdust_100[:ells.size] / fsky)[1:ls.max()] / premask_fudge, label='Spinning Dust Sim')
# 143 GHz
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_143[:ells.size]              / fsky)[1:ls.max()] / premask_fudge, label='Full Sky')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_143foregrounds[:ells.size]   / fsky)[1:ls.max()] / premask_fudge, label='SMICA Subtracted Sky')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_thermaldust_143[:ells.size]  / fsky)[1:ls.max()] / premask_fudge, label='Thermal Dust Sim')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_freefree_143[:ells.size]     / fsky)[1:ls.max()] / premask_fudge, label='Free-Free Emission Sim')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_synchrotron_143[:ells.size]  / fsky)[1:ls.max()] / premask_fudge, label='Synchrotron Sim')
#ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_spinningdust_143[:ells.size] / fsky)[1:ls.max()] / premask_fudge, label='Spinning Dust Sim')
# 217 GHz
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_217[:ells.size]              / fsky)[1:ls.max()] / premask_fudge, label='Full Sky')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_217foregrounds[:ells.size]   / fsky)[1:ls.max()] / premask_fudge, label='SMICA Subtracted Sky')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_thermaldust_217[:ells.size]  / fsky)[1:ls.max()] / premask_fudge, label='Thermal Dust Sim')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_freefree_217[:ells.size]     / fsky)[1:ls.max()] / premask_fudge, label='Free-Free Emission Sim')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_synchrotron_217[:ells.size]  / fsky)[1:ls.max()] / premask_fudge, label='Synchrotron Sim')
#ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_spinningdust_217[:ells.size] / fsky)[1:ls.max()] / premask_fudge, label='Spinning Dust Sim')


for freq, ax in zip((100,143,217),(ax1, ax2,ax3)):
	_ = ax.set_xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
	ax.set_xlim([0, 50])
	ax.set_ylim([5e-14, 5e-7])
	ax.set_title('%dGHz'%freq)
	ax.set_ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
	ax.set_xlabel(r'$\ell$',fontsize=16)
	ax.legend()

plt.savefig(outdir+'Foreground_Contributions.png')


plt.figure()
plt.plot(centres(bins)*299792.458, n100, label='100 GHz')
plt.plot(centres(bins)*299792.458, n143, label='143 GHz')
plt.plot(centres(bins)*299792.458, n217, label='217 GHz')
plt.legend()
plt.xlabel('km / s')
plt.ylabel('Npix')
plt.title('Reconstruction of frequency maps x unWISE')
plt.savefig(outdir+'hist_freqmaps')






fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,6))


l0, = ax1.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100[:ells.size] / fsky) / premask_fudge, label='143 GHz Map', ls='None', marker='^', zorder=10)
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_100[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l0.get_c(), ls='--', alpha=0.5)
l1, = ax1.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_100foregrounds[:ells.size] / fsky) / premask_fudge, label='143 GHz no CMB', ls='None', marker='^', zorder=10)
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_100foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l1.get_c(), ls='--', alpha=0.5)
#l2, = ax1.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
#ax1.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
ax1.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_100/2.725**2,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
l0, = ax2.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143[:ells.size] / fsky) / premask_fudge, label='143 GHz Map', ls='None', marker='^', zorder=10)
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_143[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l0.get_c(), ls='--', alpha=0.5)
l1, = ax2.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_143foregrounds[:ells.size] / fsky) / premask_fudge, label='143 GHz no CMB', ls='None', marker='^', zorder=10)
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_143foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l1.get_c(), ls='--', alpha=0.5)
#l2, = ax2.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
#ax2.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l2.get_c(), ls='--', alpha=0.5)
ax2.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_143,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)
l3, = ax3.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217[:ells.size] / fsky) / premask_fudge, label='217 GHz Map', ls='None', marker='^', zorder=10)
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_217[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l3.get_c(), ls='--', alpha=0.5)
l4, = ax3.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(recon_Cls_217foregrounds[:ells.size] / fsky) / premask_fudge, label='217 GHz no CMB', ls='None', marker='^', zorder=10)
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_217foregrounds[:ells.size] / fsky)[1:ls.max()] / premask_fudge, c=l4.get_c(), ls='--', alpha=0.5)
#l5, = ax3.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(clv_windowed_interp), label='Theory Signal', ls='None', marker='^', zorder=9)
#ax3.semilogy(np.arange(1,ls.max()), clv_windowed_interp[1:ls.max()], c=l5.get_c(), ls='--', alpha=0.5)
ax3.semilogy(bandpowers(np.arange(ls.max()+1)), bandpowers(np.repeat(noise_217,ells.size)), c='k',label='Theory Noise', ls='--', zorder=0, lw=2)


for freq, ax in zip((100,143,217),(ax1, ax2,ax3)):
	_ = ax.set_xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
	ax.set_xlim([0, 50])
	ax.set_ylim([5e-10, 5e-7])
	ax.set_title('%dGHz'%freq)
	ax.set_ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
	ax.set_xlabel(r'$\ell$',fontsize=16)
	ax.legend()

plt.savefig(outdir+'Freq_maps.png')



plt.figure()
hp.mollview(SMICAinp,title='',cbar=False,min=np.mean(SMICAinp)-3*np.std(SMICAinp),max=np.mean(SMICAinp)+3*np.std(SMICAinp),bgcolor='#262628')
plt.savefig(outdir+'SMICA input')


plt.figure()
hp.mollview(unWISEmap,title='',cbar=False,max=15,bgcolor='#262628')
plt.savefig(outdir+'unWISE input')

plt.figure()
hp.mollview(unwise_mask,title='',cbar=False,cmap='autumn',bgcolor='#262628')
plt.savefig(outdir+'Mask input')




plt.figure()
hp.mollview(hugemask_unwise)
plt.savefig(outdir+'hugemask_unwise')




fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,6))
# 100 GHz
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_100hugemask[:ells.size]              / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Full Sky')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_100hugemaskforegrounds[:ells.size]   / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='SMICA Subtracted Sky')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_thermaldust_100hugemask[:ells.size]  / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Thermal Dust Sim')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_freefree_100hugemask[:ells.size]     / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Free-Free Emission Sim')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_synchrotron_100hugemask[:ells.size]  / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Synchrotron Sim')
ax1.semilogy(np.arange(1,ls.max()), (recon_Cls_spinningdust_100hugemask[:ells.size] / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Spinning Dust Sim')
# 143 GHz
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_143hugemask[:ells.size]              / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Full Sky')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_143hugemaskforegrounds[:ells.size]   / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='SMICA Subtracted Sky')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_thermaldust_143hugemask[:ells.size]  / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Thermal Dust Sim')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_freefree_143hugemask[:ells.size]     / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Free-Free Emission Sim')
ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_synchrotron_143hugemask[:ells.size]  / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Synchrotron Sim')
#ax2.semilogy(np.arange(1,ls.max()), (recon_Cls_spinningdust_143hugemask[:ells.size] / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Spinning Dust Sim')
# 217 GHz
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_217hugemask[:ells.size]              / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Full Sky')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_217hugemaskforegrounds[:ells.size]   / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='SMICA Subtracted Sky')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_thermaldust_217hugemask[:ells.size]  / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Thermal Dust Sim')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_freefree_217hugemask[:ells.size]     / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Free-Free Emission Sim')
ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_synchrotron_217hugemask[:ells.size]  / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Synchrotron Sim')
#ax3.semilogy(np.arange(1,ls.max()), (recon_Cls_spinningdust_217hugemask[:ells.size] / fsky_huge)[1:ls.max()] / (premask_fudge * fsky / fsky_huge), label='Spinning Dust Sim')


for freq, ax in zip((100,143,217),(ax1, ax2,ax3)):
	_ = ax.set_xticks(bandpowers(np.arange(1,ls.max()+1)), ['%d' % ell for ell in bandpowers(np.arange(1,ls.max()+1))])
	ax.set_xlim([0, 50])
	#ax.set_ylim([5e-14, 5e-7])
	ax.set_title('%dGHz'%freq)
	ax.set_ylabel(r'$\frac{v^2}{c^2}$  ', rotation=0.,fontsize=16)
	ax.set_xlabel(r'$\ell$',fontsize=16)
	ax.legend()

plt.savefig(outdir+'Foreground_Contributions_Hugemask.png')


plt.figure()
l1, = plt.plot(centres(bins_hugemask)*299792.458, n100huge, label='100 GHz')
plt.plot(centres(bins_hugemask)*299792.458, gaussian(centres(bins_hugemask),*popt100),c=l1.get_c(),ls='--')
l2, = plt.plot(centres(bins_hugemask)*299792.458, n143huge, label='143 GHz')
plt.plot(centres(bins_hugemask)*299792.458, gaussian(centres(bins_hugemask),*popt143),c=l2.get_c(),ls='--')
l3, = plt.plot(centres(bins_hugemask)*299792.458, n217huge, label='217 GHz')
plt.plot(centres(bins_hugemask)*299792.458, gaussian(centres(bins_hugemask),*popt217),c=l3.get_c(),ls='--')
plt.legend()
plt.xlabel('km / s')
plt.ylabel('Npix')
plt.title('Reconstruction of frequency maps x unWISE\nwith mask = planck fsky 20% x unWISE')
plt.savefig(outdir+'hist_freqmaps_hugemask')




# Questions:
# Gaussian for theory noise on histogram: are we tightening because of less low ell crap or because it's a part of how the noise behaves?
# Can we learn some physics from our result? Is there an optical depth map from the planck sky model we can use?

# Tasks:
# Start writing: data sets, pipeline, stuff that's fixed. Put plots in the right sections and type up around them.
# Map-level: reconstruction on full sky minus reconstruction on thermal dust - how close to CMB reconstruction?   -  not close at all
plt.figure()
hp.mollview(lowpass_output*unwise_mask,title='SMICA x unWISE')
plt.savefig(outdir+'SMICA_out')

plt.figure()
hp.mollview(lowpass(outmap_100 - outmap_thermaldust_100)*unwise_mask,title=r'$v_{\mathrm{100 GHz}}-v_{\mathrm{dust}}$')
plt.savefig(outdir+'subtraction_100GHz')
plt.figure()
hp.mollview(lowpass(outmap_143 - outmap_thermaldust_143)*unwise_mask,title=r'$v_{\mathrm{143 GHz}}-v_{\mathrm{dust}}$')
plt.savefig(outdir+'subtraction_143GHz')
plt.figure()
hp.mollview(lowpass(outmap_217 - outmap_thermaldust_217)*unwise_mask,title=r'$v_{\mathrm{217 GHz}}-v_{\mathrm{dust}}$')
plt.savefig(outdir+'subtraction_217GHz')


# Add normal product distro to 1-pt plot, specifically the SMICA x unWISE reconstruction
from scipy.integrate import simps
from scipy.special import kn
bessel = lambda bins, Tmap, lssmap : kn(0, np.abs(centres(bins)) / (np.std(Tmap)*np.std(lssmap)))
normal_product = lambda bins, Tmap, lssmap : bessel(bins,Tmap,lssmap) / (np.pi * np.std(Tmap) * np.std(lssmap))
pixel_scaling = lambda distribution : (12*2048**2) * (distribution / simps(distribution))
pixel_scaling_masked = lambda distribution, FSKY : (12*2048**2) * FSKY * (distribution / simps(distribution))

ClTT_filter = ClTT.copy()
ClTT_filter[:100] = 0.
Clgg_filter = csm.Clgg[0,0,:].copy()
Clgg_filter[:100] = 0.
Clgg_filter[spectra_lmax:] = 0.
Tlms = hp.almxfl(hp.map2alm(SMICAmap_real), np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
lsslms = hp.almxfl(hp.map2alm(unWISEmap), np.divide(Cltaug_at_zbar.copy(), Clgg_filter, out=np.zeros_like(Cltaug_at_zbar), where=Clgg_filter!=0))

Tmap = hp.alm2map(Tlms, 2048)
lssmap = hp.alm2map(lsslms,2048)
normprod_SMICA_unwise = bessel(bins,Tmap,lssmap*noise_SMICA) / (np.pi * np.std(Tmap) * np.std(lssmap*noise_SMICA))

narf,barf,_ = plt.hist(outmap_SMICA[np.where(unwise_mask!=0)],bins=10000)
klkfsf = bessel(barf,Tmap,lssmap* noise_SMICA) / (np.pi * np.std(Tmap) * np.std(lssmap* noise_SMICA))
plt.figure()
plt.plot(centres(barf)*299792.458, narf, label='Reconstruction')
plt.plot(centres(barf)*299792.458, fsky**.5 * 12*2048**2* klkfsf / simps(klkfsf), label='Normal Product Distribution',ls='--')
plt.legend()
plt.xlabel('km / s')
plt.ylabel('Npix')
plt.title('Reconstruction of unWISE x SMICA')
plt.xlim([-3e4,3e4])
plt.savefig(outdir+'1pt')

n_Tmap, bins_Tmap, _ = plt.hist(Tmap[np.where(unwise_mask!=0)],bins=np.linspace(-3*np.std(Tmap),3*np.std(Tmap),1000))
n_lssmap, bins_lssmap, _ = plt.hist(lssmap[np.where(unwise_mask!=0)],bins=np.linspace(-3*np.std(lssmap),3*np.std(lssmap),1000))

popt_SMICA, pcov_SMICA = curve_fit(gaussian, centres(bins_Tmap), n_Tmap, p0=[np.max(n_Tmap), 0., np.std(Tmap[np.where(unwise_mask!=0)])])
popt_unWISE, pcov_unWISE = curve_fit(gaussian, centres(bins_lssmap), n_lssmap, p0=[np.max(n_lssmap), 0., np.std(lssmap[np.where(unwise_mask!=0)])])

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
ax1.plot(centres(bins_Tmap),n_Tmap, label='Filtered SMICA')
ax1.plot(centres(bins_Tmap), gaussian(centres(bins_Tmap), *popt_SMICA), c='k', ls='--', label='Gaussian Fit')
ax1.set_title('Temperature Map')
ax1.legend()
ax2.plot(centres(bins_lssmap), n_lssmap, label='Filtered unWISE')
ax2.plot(centres(bins_lssmap), gaussian(centres(bins_lssmap), *popt_unWISE), c='k', ls='--', label='Gaussian Fit')
ax2.set_title('LSS Map')
ax2.legend()
plt.savefig(outdir+'input_histograms')




test1=np.random.normal(size=12*2048**2)
test2=np.random.normal(size=12*2048**2)
testprod = test1*test2

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,4))
_,_,_ = ax1.hist(test1,bins=300)
_,_,_ = ax2.hist(test2,bins=300)
testprod_n,testprod_bins,_ = ax3.hist(testprod,bins=300)
ax3.plot(centres(testprod_bins), 12*2048**2 * bessel(testprod_bins,test1,test2) / (np.pi * np.std(test1) * np.std(test2)) / simps(bessel(testprod_bins,test1,test2) / (np.pi * np.std(test1) * np.std(test2))))
plt.savefig(outdir+'test_normprod')

# do statistical errors for velocity: 1000 gauss of windowed V on the sky -> mask them -> find mean and variance of masked realizations



# Effect of photo z errors
csm_new = Cosmology(nbin=1,zmin=redshifts.min(), zmax=redshifts.max(), redshifts=redshifts, ks=ks, zerrs=False)  # Set up cosmology
csm_new.cambpars = pars
csm_new.cosmology_data = camb.get_background(pars)
csm_new.bin_width = chis[-1] - chis[0]
csm_new.compute_Cls(ngbar=ngbar)  # These are the integrated Cls over the entire bin

plt.figure()
plt.loglog(csm_new.Clgg[0,0,:], label='No error')
plt.loglog(csm.Clgg[0,0,:], label='With photo-z + catastrophic error')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{gg}}$')
plt.legend()
plt.savefig(outdir+'clgg')


# Delta z of around 0.1, with a 1% chance of being around 1
with open('data/unWISE/blue.txt', 'r') as FILE:
    x = FILE.readlines()

z = np.array([float(l.split(' ')[0]) for l in x])
dndz = np.array([float(l.split(' ')[1]) for l in x])

counts_unwise_mask = 81808220
counts_per_zbin = (dndz*counts_unwise_mask*0.01/simps(dndz,z)).astype(int)
catastrophic_counts = (counts_per_zbin * 0.01).astype(int)

redshifts_of_counts = np.zeros(counts_per_zbin.sum())
cursor = 0
for i, z_val in enumerate(z):
    zbin_counts_distribution = np.random.normal(loc=z_val, scale=0.05, size=counts_per_zbin[i])
    photoz_catastrophic = np.random.choice([0,1],size=catastrophic_counts[i])*-2+1  # \pm 1 delta z
    zbin_counts_distribution[:photoz_catastrophic.size] += photoz_catastrophic
    redshifts_of_counts[cursor:cursor+zbin_counts_distribution.size] = zbin_counts_distribution
    cursor += zbin_counts_distribution.size

n_counts, _ = np.histogram(redshifts_of_counts, bins=z)
dndz_err = n_counts * simps(dndz,z) / simps(n_counts, 0.5*(z[1:]+z[:-1]))

redshifts_of_counts = np.zeros(counts_per_zbin.sum())
cursor = 0
for i, z_val in enumerate(z):
    zbin_counts_distribution = np.random.normal(loc=z_val, scale=0.05, size=counts_per_zbin[i])
    redshifts_of_counts[cursor:cursor+zbin_counts_distribution.size] = zbin_counts_distribution
    cursor += zbin_counts_distribution.size

n_counts, _ = np.histogram(redshifts_of_counts, bins=z)
dndz_err_nocatastrophic = n_counts * simps(dndz,z) / simps(n_counts, 0.5*(z[1:]+z[:-1]))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.plot(z,dndz, label='no errors')
ax1.plot(centres(z),dndz_err_nocatastrophic,label='with photo-z error')
ax1.plot(centres(z),dndz_err,label='with photo-z + catastrophic error')
ax1.legend()
ax2.semilogy(z,dndz,label='no errors')
ax2.semilogy(centres(z),dndz_err_nocatastrophic,label='with photo-z error')
ax2.semilogy(centres(z),dndz_err,label='with photo-z + catastrophic error')
ax2.legend()
for ax in (ax1, ax2):
	ax.set_xlabel('z')
	ax.set_ylabel('dN / dz')

plt.savefig(outdir+'dndz_errs')

