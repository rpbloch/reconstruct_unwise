# hp.remove_dipole does something for reconstructions using hp.ma
# but hp.remove_dipole gives same result whether or not its used at all
# when taking power spectrum of things masked by multiplying by integer maps
# Above is sussy? Doesn't seem to apply anymore. NVM. Big difference for SMICA, not so much for COMMANDER
# Also the COMMANDER results appear to be the same as Matt?

# mask (original way) then debeam = 10% higher theory noise than reverse order in new way
# mask (new way) then debeam = exact result as above for 10% higher theory noise

# if using ClTT = hp.anafast(map) (no planck masking) for SMICA, we don't get the 10% loss.
# but ClTT unmasked map for COMMANDER is much more different than theory ClTT for COMMANDER
# when planck masking, SMICA is almost the exact same with or without the planck mask
# so why is SMICA so adversely affected by our choice of ClTT?

# Matt and I get the same COMMANDER reconstruction, but different y-axis by a bit.
# Choice of Cltaug probably. However his theory noise is a bit higher than mine
# relative to the reconstruction, so while we have the same shape (and same bandpowered shape)
# his noise line goes right through the middle so mine looks more like excess signal when it's
# just low noise.

# Now why do I get low noise?...His theory spectra are taken directly from maps

# SMICA fix by matching theory noise TT and map input TT, letting filter TT remain planckmasked.
# however due to crap in COMMANDER map this ruins the COMMANDER recon? =(

# Why do we get different shapes for our SMICA reconstructions but not for our COMMANDER ones?


# 1) get SMICA shape figured out why it's different. Maybe noise? 10% might be permanent.
# 2) Fix your cosmology to Matt's
# 3) Recompute window velocity via Matt's new paper method
# 4) Check the \ell to \ellbar approximation

# Meet Monday

import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as tick
from matplotlib import colors as mc
import colorsys
from math import lgamma
plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})
import numpy as np
import camb
from camb import model
import scipy
from paper_analysis import Estimator, Cosmology
from astropy.io import fits
from astropy import units as u
import healpy as hp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.special import kn
from scipy.special import spherical_jn as jn


class maps(object):
	def __init__(self):
		self.nside = 2048
		self.lmax = 4000
		self.Cls = {}
		self.processed_alms = {}
		self.stored_maps = {}
		print('Loading maps...')
		# K_CMB units are removed from all temperature maps except frequency maps for 100 GHz.
		# Arithmetic operations on that map or any copies thereof generate numerical garbage.
		# Instead we carry units of K_CMB through and remove them from the reconstruction itself.
		self.input_SMICA = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES'], n2r=True) / 2.7255
		self.input_COMMANDER = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-commander_2048_R3.00_full.fits')[1].data['I_STOKES'], n2r=True) / 2.7255
		self.input_unWISE = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
		self.input_T100 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
		self.input_T143 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.7255
		self.input_T217 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.7255
		self.input_T353 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_353-psb_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) / 2.7255
		self.input_T100_noCMB_SMICA = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten()
		self.input_T143_noCMB_SMICA = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
		self.input_T217_noCMB_SMICA = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
		self.input_T353_noCMB_SMICA = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-353_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
		self.input_T100_noCMB_COMMANDER = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-commander-100_R3.00.fits')[1].data['INTENSITY'].flatten()
		self.input_T143_noCMB_COMMANDER = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-commander-143_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
		self.input_T217_noCMB_COMMANDER = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-commander-217_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
		self.input_T353_noCMB_COMMANDER = fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-commander-353_R3.00.fits')[1].data['INTENSITY'].flatten() / 2.7255
		self.input_T100_thermaldust = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-100_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()
		self.input_T143_thermaldust = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-143_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.7255
		self.input_T217_thermaldust = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-217_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.7255
		self.input_T353_thermaldust = fits.open('data/planck_data_testing/foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-353_2048_R3.00_full.fits')[1].data['TEMPERATURE'].flatten()  / 2.7255
		self.input_T353_CIB = np.nan_to_num(fits.open('data/planck_data_testing/CIB/cib_fullmission_353.hpx.fits')[1].data['CIB']) * (u.MJy / u.sr).to(1. * u.K, equivalencies=u.thermodynamic_temperature(353 * u.GHz, T_cmb=2.7255*u.K)) / 2.7255
		self.mask_planck = hp.reorder(fits.open('data/planck_data_testing/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')[1].data['TMASK'],n2r=True)
		self.mask_unwise = np.load('data/mask_unWISE_thres_v10.npy')
		self.mask_GAL020 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL020'],n2r=True)
		mask_CIB = np.ones(self.input_T353_CIB.size)
		mask_CIB[np.where(np.isnan(fits.open('data/planck_data_testing/CIB/cib_fullmission_353.hpx.fits')[1].data['CIB']))] = 0
		self.mask = self.mask_unwise * self.mask_planck
		self.mask_huge = self.mask_GAL020.astype(np.float32) * self.mask_unwise * self.mask_planck * mask_CIB
		####
		# Map-based values
		self.fsky = np.where(self.mask!=0)[0].size / self.mask.size
		self.fsky_planck = np.where(self.mask_planck!=0)[0].size / self.mask_planck.size
		self.fsky_huge = np.where(self.mask_huge!=0)[0].size / self.mask_huge.size
		self.ngbar = self.input_unWISE.sum() / self.input_unWISE.size
		####
		# Beams
		self.SMICAbeam = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143)
		self.T100beam = hp.gauss_beam(fwhm=np.radians(9.68/60), lmax=6143)
		self.T143beam = hp.gauss_beam(fwhm=np.radians(7.30/60), lmax=6143)
		self.T217beam = hp.gauss_beam(fwhm=np.radians(5.02/60), lmax=6143)
		self.T353beam = hp.gauss_beam(fwhm=np.radians(4.94/60), lmax=6143)
		self.T100beam[self.lmax:] = self.T100beam[self.lmax]  # Extremely high beam for 100GHz at high ell ruins the map
		###
		print('Masking/debeaming maps...')
	def mask_and_debeam(self, input_map, mask_map, beam):
		return hp.almxfl(hp.map2alm(input_map * mask_map, lmax=self.lmax), 1/beam)
	def alm2cl(self, alms, fsky):
		return hp.alm2cl(alms) / fsky
	def lowpass_filter(self, input_map, lmax=25):
		ell_filter = [0 if l > lmax else 1 for l in np.arange(self.lmax)]
		map_alms = hp.map2alm(input_map, lmax=self.lmax)
		return hp.alm2map(hp.almxfl(map_alms, ell_filter), lmax=self.lmax, nside=self.nside)


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

def combine_alm(dTlm, dlm, mask, ClTT, Clgg, Cltaudg, Noise, lmax, nside_out, convert_K=False):
    ClTT_filter = ClTT.copy()[:lmax+1]
    Clgg_filter = Clgg.copy()[:lmax+1]
    Cltaudg = Cltaudg.copy()[:lmax+1]
    ClTT_filter[:100] = 1e15
    Clgg_filter[:100] = 1e15
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
    dlm_zeta = hp.almxfl(dlm, np.divide(Cltaudg, Clgg_filter, out=np.zeros_like(Cltaudg), where=Clgg_filter!=0))
    Tmap_filtered = hp.alm2map(dTlm_xi, lmax=lmax, nside=nside_out) * mask
    lssmap_filtered = hp.alm2map(dlm_zeta, lmax=lmax, nside=nside_out) * mask
    outmap_filtered = Tmap_filtered*lssmap_filtered
    outmap = -outmap_filtered * Noise * mask
    if convert_K:  # output map has units of K
        outmap /= 2.725
    return outmap

def bias_e2(z, k):
    bstar2 = lambda z : 0.971 - 0.013*z
    gamma = lambda z : 1.91 - 0.59*z + 0.10*z**2
    kstar = lambda z : 4.36 - 3.24*z + 3.10*z**2 - 0.42*z**3
    bias_squared = np.zeros((z.size, k.size))
    for zid, redshift in enumerate(z):
        bias_squared[zid, :] = bstar2(redshift) / ( 1 + (k/kstar(redshift))**gamma(redshift) )
    return np.sqrt(bias_squared)

def compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt, clgg, cltaug, sample_ells, chis, chibar, zbar_index, bin_width):
    terms = 0
    terms_with_me_entry = np.zeros(chis.size)
    terms_with_mm_me_entry = np.zeros(chis.size)
    ell_const = 2
    for l2 in np.arange(spectra_lmax):
    	Pmm_at_ellprime = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (l2+0.5)/chis[::-1], grid=True), axis=1))
    	Pem_at_ellprime = Pmm_at_ellprime * np.diagonal(np.flip(bias_e2(results.redshift_at_comoving_radial_distance(chis), (l2+0.5)/chis[::-1]), axis=1))  # Convert Pmm to Pem
    	Pem_at_ellprime_at_zbar = Pem_at_ellprime[zbar_index]
    	for l1 in np.arange(np.abs(l2-ell_const),l2+ell_const+1):
    		if l1 > spectra_lmax-1 or l1 <2:   #triangle rule
    			continue
    		gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell_const+1)/(4*np.pi))*wigner_symbol(ell_const, l1, l2)*cltaug[l2]
    		term_with_me_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg[l2])) * (Pem_at_ellprime/Pem_at_ellprime_at_zbar)
    		term_with_mm_me_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg[l2])) * (Pmm_at_ellprime/Pem_at_ellprime_at_zbar)
    		term_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg[l2]))
    		if np.isfinite(term_entry):
    			terms += term_entry
    			terms_with_me_entry += term_with_me_entry
    			terms_with_mm_me_entry += term_with_mm_me_entry
    ratio_me_me = terms_with_me_entry / terms
    ratio_mm_me = terms_with_mm_me_entry / terms
    window_v = np.nan_to_num(  ( 1/bin_width ) * ( chibar**2 / chis**2 ) * ( galaxy_window / galaxy_window[zbar_index] ) * ( (1+results.redshift_at_comoving_radial_distance(chis))**2 / (1+results.redshift_at_comoving_radial_distance(chibar))**2 ) * ratio_me_me  )
    window_v_mm = np.nan_to_num(  ( 1/bin_width ) * ( chibar**2 / chis**2 ) * ( galaxy_window / galaxy_window[zbar_index] ) * ( (1+results.redshift_at_comoving_radial_distance(chis))**2 / (1+results.redshift_at_comoving_radial_distance(chibar))**2 ) * ratio_mm_me  )
    clv_windowed = np.zeros(sample_ells.size)
    clv_windowed_mm = np.zeros(sample_ells.size)
    for i in np.arange(sample_ells.size):
    	clv_windowed[i] = np.trapz(window_v*np.trapz(window_v*clv[i,:,:], chis,axis=1), chis)
    	clv_windowed_mm[i] = np.trapz(window_v_mm*np.trapz(window_v_mm*clv[i,:,:], chis,axis=1), chis)
    return clv_windowed, clv_windowed_mm

def compute_common(dTlm, ClTT, Clgg, lmax, nside_out):
	ClTT_filter = ClTT.copy()[:lmax+1]
	Clgg_filter = Clgg.copy()[:lmax+1]
	ClTT_filter[:100] = 1e15
	Clgg_filter[:100] = 1e15
	dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
	Tmap_filtered = hp.alm2map(dTlm_xi, lmax=lmax, nside=nside_out)
	return ClTT_filter, Clgg_filter, Tmap_filtered
    
### File Handling
outdir = 'plots/paper/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

### Setup
redshifts = np.linspace(0.0,2.5,100)
zbar_index = 30
spectra_lmax = 4000
recon_lmax = 100
ls = np.unique(np.append(np.geomspace(1,spectra_lmax,200).astype(int), spectra_lmax))
maplist = maps()

### Cosmology
print('Setting up cosmology')
nks = 2000
ks = np.logspace(-4,1,nks)
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(redshifts=redshifts, kmax=2.0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
cosmo_data = camb.get_background(pars)

chis = results.comoving_radial_distance(redshifts)
bin_width = chis[-1] - chis[0]
fullspectrum_ls = np.unique(np.append(np.geomspace(1,spectra_lmax-1,200).astype(int), spectra_lmax-1))

Pmms = np.zeros((chis.size,fullspectrum_ls.size))
Pmm_full = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=100., zmax=redshifts.max())
for l, ell in enumerate(fullspectrum_ls):  # Do limber approximation: P(z,k) -> P(z, (ell+0.5)/chi )
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))

thomson_SI = 6.6524e-29
m_per_Mpc = 3.086e22
ne_0 = 0.16773206895639853  # Average electron density today in 1/m^3

taud_window  = (thomson_SI * ne_0 * (1+results.redshift_at_comoving_radial_distance(chis))**2 * m_per_Mpc)[zbar_index]  # Units of 1/Mpc
h = results.h_of_z(redshifts)
chibar = results.comoving_radial_distance(redshifts[zbar_index])

### Windows and spectra
print('Computing galaxy windows and tau-g cross-power')
galaxy_bias = (0.8+1.2*results.redshift_at_comoving_radial_distance(chis))

with open('data/unWISE/blue.txt', 'r') as FILE:
    dndz_data = FILE.readlines()

z_alex = np.array([float(l.split(' ')[0]) for l in dndz_data])
dN_alex = np.array([float(l.split(' ')[1]) for l in dndz_data])
dNdz_fiducial = interp1d(z_alex, dN_alex, kind= 'linear', bounds_error=False, fill_value=0)(results.redshift_at_comoving_radial_distance(chis))

dNdz_realization = np.zeros((100, chis.size))
for i in np.arange(100):
	with open('data/unWISE/blue_dNdz_err/%s.txt' % i, 'r') as FILE:
	    dndz_data = FILE.readlines()
	z_alex_err = np.array([float(l.split(' ')[0]) for l in dndz_data])
	dN_alex_err = np.array([float(l.split(' ')[1]) for l in dndz_data])
	dNdz_err = interp1d(z_alex_err, dN_alex_err, kind= 'linear', bounds_error=False, fill_value=0)(results.redshift_at_comoving_radial_distance(chis))
	dNdz_realization[i,:] = dNdz_err.copy()

galaxy_window = galaxy_bias * dNdz_fiducial * cosmo_data.h_of_z(results.redshift_at_comoving_radial_distance(chis))
galaxy_window_dndz = galaxy_bias[np.newaxis,:] * dNdz_realization * cosmo_data.h_of_z(results.redshift_at_comoving_radial_distance(chis))

cltaug_fiducial_coarse = np.zeros((chis.size, ls.size))
cltaug_dndz_coarse = np.zeros((100, chis.size, ls.size))
cltaug_mm_fiducial_coarse = np.zeros((chis.size, ls.size))
cltaug_mm_dndz_coarse = np.zeros((100, chis.size, ls.size))
for l, ell in enumerate(ls):
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))
	m_to_e = np.diagonal(np.flip(bias_e2(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1]), axis=1))
	Pme_at_ell = Pmms[:,l] * m_to_e
	cltaug_fiducial_coarse[:, l] = np.nan_to_num(Pme_at_ell * galaxy_window * taud_window / chis**2, posinf=0.) * bin_width * maplist.ngbar
	cltaug_mm_fiducial_coarse[:, l] = np.nan_to_num(Pmms[:,l]  * galaxy_window * taud_window / chis**2, posinf=0.) * bin_width * maplist.ngbar
	cltaug_dndz_coarse[:,:,l] = np.nan_to_num(Pme_at_ell[np.newaxis,:] * galaxy_window_dndz * taud_window / chis[np.newaxis,:]**2, posinf=0.) * bin_width * maplist.ngbar
	cltaug_mm_dndz_coarse[:,:,l] = np.nan_to_num(Pmms[np.newaxis,:,l] * galaxy_window_dndz * taud_window / chis[np.newaxis,:]**2, posinf=0.) * bin_width * maplist.ngbar

cltaug_fiducial = interp1d(ls, cltaug_fiducial_coarse[zbar_index,:], bounds_error=False, fill_value='extrapolate')(np.arange(maplist.lmax+1))
cltaug_dndz = np.array([interp1d(ls, cltaug_dndz_coarse[i,zbar_index,:], bounds_error=False, fill_value='extrapolate')(np.arange(maplist.lmax+1)) for i in np.arange(100)])
cltaug_fiducial_mm = interp1d(ls, cltaug_mm_fiducial_coarse[zbar_index,:], bounds_error=False, fill_value='extrapolate')(np.arange(maplist.lmax+1))
cltaug_dndz_mm = np.array([interp1d(ls, cltaug_mm_dndz_coarse[i,zbar_index,:], bounds_error=False, fill_value='extrapolate')(np.arange(maplist.lmax+1)) for i in np.arange(100)])

### Velocity
print('Computing true velocity...')
velocity_compute_ells = np.unique(np.concatenate([np.arange(1,16),np.geomspace(16,35,5)]).astype(int))
clv = np.zeros((velocity_compute_ells.shape[0],redshifts.shape[0],redshifts.shape[0]))
PKv = camb.get_matter_power_interpolator(pars,hubble_units=False, k_hunit=False, var1='v_newtonian_cdm',var2='v_newtonian_cdm')
for l in range(velocity_compute_ells.shape[0]):
	print('    @ l = %d' % velocity_compute_ells[l])
	for z1 in range(redshifts.shape[0]):
		for z2 in range(redshifts.shape[0]):
			integrand_k = scipy.special.spherical_jn(velocity_compute_ells[l],ks*chis[z1])*scipy.special.spherical_jn(velocity_compute_ells[l],ks*chis[z2]) * (h[z1]/(1+redshifts[z1]))*(h[z2]/(1+redshifts[z2])) * np.sqrt(PKv.P(redshifts[z1],ks)*PKv.P(redshifts[z2],ks))
			clv[l,z1,z2] = (2./np.pi)*np.trapz(integrand_k,ks)

### Reconstructions
print('Processing CMB reconstructions...')
map_container = {'100GHz' : maplist.input_T100,
				 '143GHz' : maplist.input_T143,
				 '217GHz' : maplist.input_T217,
				 '353GHz' : maplist.input_T353,
				 '100GHz_noSMICA' : maplist.input_T100_noCMB_SMICA,
				 '143GHz_noSMICA' : maplist.input_T143_noCMB_SMICA,
				 '217GHz_noSMICA' : maplist.input_T217_noCMB_SMICA,
				 '353GHz_noSMICA' : maplist.input_T353_noCMB_SMICA,
				 '100GHz_noCOMMANDER' : maplist.input_T100_noCMB_COMMANDER,
				 '143GHz_noCOMMANDER' : maplist.input_T143_noCMB_COMMANDER,
				 '217GHz_noCOMMANDER' : maplist.input_T217_noCMB_COMMANDER,
				 '353GHz_noCOMMANDER' : maplist.input_T353_noCMB_COMMANDER,
				 '100GHz_thermaldust' : maplist.input_T100_thermaldust,
				 '143GHz_thermaldust' : maplist.input_T143_thermaldust,
				 '217GHz_thermaldust' : maplist.input_T217_thermaldust,
				 '353GHz_thermaldust' : maplist.input_T353_thermaldust,
				 '353GHz_CIB' : maplist.input_T353_CIB}

beam_container = {'100GHz' : maplist.T100beam,
				  '143GHz' : maplist.T143beam,
				  '217GHz' : maplist.T217beam,
				  '353GHz' : maplist.T353beam,
				  '100GHz_noSMICA' : maplist.T100beam,
				  '143GHz_noSMICA' : maplist.T143beam,
				  '217GHz_noSMICA' : maplist.T217beam,
				  '353GHz_noSMICA' : maplist.T353beam,
				  '100GHz_noCOMMANDER' : maplist.T100beam,
				  '143GHz_noCOMMANDER' : maplist.T143beam,
				  '217GHz_noCOMMANDER' : maplist.T217beam,
				  '353GHz_noCOMMANDER' : maplist.T353beam,
				  '100GHz_thermaldust' : maplist.T100beam,
				  '143GHz_thermaldust' : maplist.T143beam,
				  '217GHz_thermaldust' : maplist.T217beam,
				  '353GHz_thermaldust' : maplist.T353beam,
				  '353GHz_CIB' : maplist.SMICAbeam}

reconstructions = {}
noises = {}
recon_Cls = {}

print('    Preprocessing SMICA, COMMANDER, and unWISE maps')
maplist.processed_alms['SMICA'] = maplist.mask_and_debeam(maplist.input_SMICA, np.ones(maplist.mask.size), maplist.SMICAbeam)
maplist.processed_alms['COMMANDER'] = maplist.mask_and_debeam(maplist.input_COMMANDER, np.ones(maplist.mask.size), maplist.SMICAbeam)
maplist.processed_alms['unWISE'] = hp.map2alm(maplist.input_unWISE, lmax=4000)

maplist.Cls['SMICA'] = maplist.alm2cl(maplist.mask_and_debeam(maplist.input_SMICA, maplist.mask_planck, maplist.SMICAbeam), maplist.fsky_planck)
maplist.Cls['COMMANDER'] = maplist.alm2cl(maplist.mask_and_debeam(maplist.input_COMMANDER, maplist.mask_planck, maplist.SMICAbeam), maplist.fsky_planck)
maplist.Cls['unWISE'] = hp.alm2cl(maplist.processed_alms['unWISE'])

print('    Computing fiducial reconstructions')
noises['SMICA'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['SMICA'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
noises['COMMANDER'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['COMMANDER'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
reconstructions['SMICA'] = combine_alm(maplist.processed_alms['SMICA'], maplist.processed_alms['unWISE'], maplist.mask, maplist.Cls['SMICA'], maplist.Cls['unWISE'], cltaug_fiducial, noises['SMICA'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
reconstructions['COMMANDER'] = combine_alm(maplist.processed_alms['COMMANDER'], maplist.processed_alms['unWISE'], maplist.mask, maplist.Cls['COMMANDER'], maplist.Cls['unWISE'], cltaug_fiducial, noises['COMMANDER'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
recon_Cls['SMICA'] = maplist.alm2cl(hp.map2alm(reconstructions['SMICA'], lmax=recon_lmax), maplist.fsky)
recon_Cls['COMMANDER'] = maplist.alm2cl(hp.map2alm(reconstructions['COMMANDER'], lmax=recon_lmax), maplist.fsky)
recon_Cls['SMICAxCOMMANDER'] = hp.alm2cl(hp.map2alm(reconstructions['SMICA'], lmax=recon_lmax), hp.map2alm(reconstructions['COMMANDER'], lmax=recon_lmax)) / maplist.fsky
noises['SMICA_mm'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['SMICA'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial_mm)
noises['COMMANDER_mm'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['COMMANDER'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial_mm)
reconstructions['SMICA_mm'] = combine_alm(maplist.processed_alms['SMICA'], maplist.processed_alms['unWISE'], maplist.mask, maplist.Cls['SMICA'], maplist.Cls['unWISE'], cltaug_fiducial_mm, noises['SMICA_mm'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
reconstructions['COMMANDER_mm'] = combine_alm(maplist.processed_alms['COMMANDER'], maplist.processed_alms['unWISE'], maplist.mask, maplist.Cls['COMMANDER'], maplist.Cls['unWISE'], cltaug_fiducial_mm, noises['COMMANDER_mm'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
recon_Cls['SMICAxCOMMANDER_mm'] = hp.alm2cl(hp.map2alm(reconstructions['SMICA_mm'], lmax=recon_lmax), hp.map2alm(reconstructions['COMMANDER_mm'], lmax=recon_lmax)) / maplist.fsky
recon_Cls['SMICA_mm'] = maplist.alm2cl(hp.map2alm(reconstructions['SMICA_mm'], lmax=recon_lmax), maplist.fsky)
recon_Cls['COMMANDER_mm'] = maplist.alm2cl(hp.map2alm(reconstructions['COMMANDER_mm'], lmax=recon_lmax), maplist.fsky)

for key in map_container:
	print('    Preprocessing %s' % key)
	maplist.processed_alms[key] = maplist.mask_and_debeam(map_container[key], maplist.mask, beam_container[key])
	maplist.processed_alms[key+'_CIBmask'] = maplist.mask_and_debeam(map_container[key], maplist.mask_huge, beam_container[key])
	maplist.Cls[key] = maplist.alm2cl(maplist.processed_alms[key], maplist.fsky)
	maplist.Cls[key+'_CIBmask'] = maplist.alm2cl(maplist.processed_alms[key+'_CIBmask'], maplist.fsky_huge)

for key in map_container:
	print('    Reconstructing %s' % key)
	master_cltt = maplist.Cls[key.split('_')[0]]  # Theory ClTT should be on the full frequency sky
	noises[key] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=master_cltt, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
	noises[key+'_mm'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=master_cltt, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial_mm)
	if '353GHz' not in key:
		convert_K_flag = True if '100GHz' in key else False
		reconstructions[key] = combine_alm(maplist.processed_alms[key], maplist.processed_alms['unWISE'], maplist.mask, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial, noises[key], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=convert_K_flag)
		reconstructions[key+'_mm'] = combine_alm(maplist.processed_alms[key], maplist.processed_alms['unWISE'], maplist.mask, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial_mm, noises[key+'_mm'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=convert_K_flag)
		recon_Cls[key] = maplist.alm2cl(hp.map2alm(reconstructions[key], lmax=recon_lmax), maplist.fsky)
		recon_Cls[key+'_mm'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_mm'], lmax=recon_lmax), maplist.fsky)
	if ('217GHz' in key) or ('353GHz' in key):
		reconstructions[key+'_CIBmask'] = combine_alm(maplist.processed_alms[key+'_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial, noises[key], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
		reconstructions[key+'_mm_CIBmask'] = combine_alm(maplist.processed_alms[key+'_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial_mm, noises[key+'_mm'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
		recon_Cls[key+'_CIBmask'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_CIBmask'], lmax=recon_lmax), maplist.fsky_huge)
		recon_Cls[key+'_mm_CIBmask'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_mm_CIBmask'], lmax=recon_lmax), maplist.fsky_huge)


maplist.stored_maps['recon_SMICA'] = maplist.lowpass_filter(reconstructions['SMICA'], lmax=25)
maplist.stored_maps['recon_COMMANDER'] = maplist.lowpass_filter(reconstructions['COMMANDER'], lmax=25)

print('    Computing windowed velocity')
clv_windowed = {key : np.zeros(velocity_compute_ells.size) for key in ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz')}
clv_windowed_mm = {key : np.zeros(velocity_compute_ells.size) for key in ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz')}
clv_windowed['COMMANDER'], clv_windowed_mm['COMMANDER'] = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['COMMANDER'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed['100GHz'], clv_windowed_mm['100GHz'] = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['100GHz'] / pars.TCMB**2, clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed['143GHz'], clv_windowed_mm['143GHz'] = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['143GHz'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed['217GHz'], clv_windowed_mm['217GHz'] = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['217GHz'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed['353GHz'], clv_windowed_mm['353GHz'] = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['353GHz'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)

print('Post-processing: computing statistics and lowpass-filtered statistics')
# Common functions
centres = lambda BIN : (BIN[:-1]+BIN[1:]) / 2
bessel = lambda bins, Tmap, lssmap : kn(0, np.abs(centres(bins)) / (np.std(Tmap)*np.std(lssmap)))
normal_product = lambda bins, Tmap, lssmap : bessel(bins,Tmap,lssmap) / (np.pi * np.std(Tmap) * np.std(lssmap))
nbin_hist = 1000000
bins_freqhist = np.linspace(-3000,1000,np.min([nbin_hist, 5000])) / 299792.458
bins_353hist = np.linspace(-6100,500,np.min([nbin_hist, 5000])) / 299792.458
# Recon 1pt plot: reconstruction vs normal product distribution
n_out_normprod_COMMANDER, bins_out_normprod_COMMANDER = np.histogram(reconstructions['COMMANDER'][np.where(maplist.mask!=0)], bins=nbin_hist)
normprod_COMMANDERmap_filtered = hp.alm2map(hp.almxfl(maplist.processed_alms['COMMANDER'], np.divide(np.ones_like(maplist.Cls['COMMANDER']), maplist.Cls['COMMANDER'], out=np.zeros_like(maplist.Cls['COMMANDER']), where=np.arange(cltaug_fiducial.size)>=100)), lmax=maplist.lmax, nside=maplist.nside)
normprod_lssmap_filtered = hp.alm2map(hp.almxfl(maplist.processed_alms['unWISE'], np.divide(cltaug_fiducial, maplist.Cls['unWISE'], out=np.zeros_like(cltaug_fiducial), where=np.arange(cltaug_fiducial.size)>=100)), lmax=maplist.lmax, nside=maplist.nside)
normprod_std_COMMANDERmap_filtered = np.std(normprod_COMMANDERmap_filtered[np.where(maplist.mask!=0)])
normprod_std_lssmap_filtered = np.std(normprod_lssmap_filtered[np.where(maplist.mask!=0)])
expect_normprod_COMMANDER = normal_product(bins_out_normprod_COMMANDER,normprod_COMMANDERmap_filtered[np.where(maplist.mask!=0)]*noises['COMMANDER'],normprod_lssmap_filtered[np.where(maplist.mask!=0)])

# 1pt frequency map plots: lowpass filters and histograms
maplist.stored_maps['recon_100GHz'] = maplist.lowpass_filter(reconstructions['100GHz'], lmax=25)
maplist.stored_maps['recon_143GHz'] = maplist.lowpass_filter(reconstructions['143GHz'], lmax=25)
maplist.stored_maps['recon_217GHz'] = maplist.lowpass_filter(reconstructions['217GHz'], lmax=25)
maplist.stored_maps['recon_217GHz_CIBmask'] = maplist.lowpass_filter(reconstructions['217GHz_CIBmask'], lmax=25)
maplist.stored_maps['recon_353GHz_CIBmask'] = maplist.lowpass_filter(reconstructions['353GHz_CIBmask'], lmax=25)
maplist.stored_maps['recon_353GHz_thermaldust_CIBmask'] = maplist.lowpass_filter(reconstructions['353GHz_thermaldust_CIBmask'], lmax=25)
maplist.stored_maps['recon_353GHz_CIB_CIBmask'] = maplist.lowpass_filter(reconstructions['353GHz_CIB_CIBmask'], lmax=25)
maplist.stored_maps['recon_353GHz_noSMICA_CIBmask'] = maplist.lowpass_filter(reconstructions['353GHz_noSMICA_CIBmask'], lmax=25)
maplist.stored_maps['recon_353GHz_noCOMMANDER_CIBmask'] = maplist.lowpass_filter(reconstructions['353GHz_noCOMMANDER_CIBmask'], lmax=25)

# With binning consistent with the multi-frequency 1-pt plot
n_out_100, _ = np.histogram(maplist.stored_maps['recon_100GHz'][np.where(maplist.mask!=0)], bins=bins_freqhist)
n_out_143, _ = np.histogram(maplist.stored_maps['recon_143GHz'][np.where(maplist.mask!=0)], bins=bins_freqhist)
n_out_217, _ = np.histogram(maplist.stored_maps['recon_217GHz'][np.where(maplist.mask!=0)], bins=bins_freqhist)
n_out_217_huge, _ = np.histogram(maplist.stored_maps['recon_217GHz_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_freqhist)
n_out_353_huge, _ = np.histogram(maplist.stored_maps['recon_353GHz_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_freqhist)
# With binning for the 353 GHz 1-pt plot
n_out_353_353hist, _ = np.histogram(maplist.stored_maps['recon_353GHz_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_353hist)
n_out_353_thermaldust, _ = np.histogram(maplist.stored_maps['recon_353GHz_thermaldust_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_353hist)
n_out_353_CIB, _ = np.histogram(maplist.stored_maps['recon_353GHz_CIB_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_353hist)
n_out_353_noSMICA, _ = np.histogram(maplist.stored_maps['recon_353GHz_noSMICA_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_353hist)
n_out_353_noCOMMANDER, _ = np.histogram(maplist.stored_maps['recon_353GHz_noCOMMANDER_CIBmask'][np.where(maplist.mask_huge!=0)], bins=bins_353hist)

# P_me approximation plot
Pme_at_ells = np.zeros((chis.size, ls.size))
slice_indices = np.arange(chis.size)[10:][::20]
for l, ell in enumerate(ls):
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))
	m_to_e = np.diagonal(np.flip(bias_e2(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1]), axis=1))
	Pme_at_ells[:,l] = Pmms[:,l] * m_to_e

# Compute dN/dz iterations for P_mm and P_me cases
print('Errors pre-processing: computing common T fields')
ClTT_filter_SMICA, Clgg_filter, Tmap_filtered_SMICA = compute_common(maplist.processed_alms['SMICA'], maplist.Cls['SMICA'], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)
ClTT_filter_COMMANDER, _, Tmap_filtered_COMMANDER = compute_common(maplist.processed_alms['COMMANDER'], maplist.Cls['COMMANDER'], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)
ClTT_filter_freq = {}
Tmap_filtered_freq = {}
for key in ('100GHz', '143GHz', '217GHz', '353GHz'):
	ClTT_filter_freq[key], _, Tmap_filtered_freq[key] = compute_common(maplist.processed_alms[key], maplist.Cls[key], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)
	_, _, Tmap_filtered_freq[key+'_noSMICA'] = compute_common(maplist.processed_alms[key + '_noSMICA'], maplist.Cls[key], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)
	_, _, Tmap_filtered_freq[key+'_noCOMMANDER'] = compute_common(maplist.processed_alms[key + '_noCOMMANDER'], maplist.Cls[key], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)
	_, _, Tmap_filtered_freq[key+'_thermaldust'] = compute_common(maplist.processed_alms[key + '_thermaldust'], maplist.Cls[key], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)
	if key == '353GHz':
		_, _, Tmap_filtered_freq[key+'_CIB'] = compute_common(maplist.processed_alms[key + '_CIB'], maplist.Cls[key], maplist.Cls['unWISE'], lmax=maplist.lmax, nside_out=maplist.nside)

print('Computing dN/dz errors')
histkeys = ('100GHz', '143GHz', '217GHz', '217GHz_CIBmask', '353GHz_CIBmask')
histkeys_353 = ('353GHz_CIBmask', '353GHz_noSMICA_CIBmask', '353GHz_noCOMMANDER_CIBmask', '353GHz_thermaldust_CIBmask', '353GHz_CIB_CIBmask')

n_out_COMMANDER_dndz = np.zeros((100, bins_out_normprod_COMMANDER.size-1))
n_out_COMMANDER_dndz_mm = np.zeros((100, bins_out_normprod_COMMANDER.size-1))
n_out_dndz = {key : np.zeros((100, bins_freqhist.size-1)) for key in histkeys}
n_out_dndz_mm = {key : np.zeros((100, bins_freqhist.size-1)) for key in histkeys}
n_out_353hist_dndz = {key : np.zeros((100, bins_353hist.size-1)) for key in histkeys_353}
n_out_353hist_dndz_mm = {key : np.zeros((100, bins_353hist.size-1)) for key in histkeys_353}

for i in np.arange(100):
	print('    Computing reconstructions using dN/dz realization %d of 100' % (i+1))
	if not os.path.exists('data/cache/lssmap_filtered_dndz-%02d.npy' % i):
		dlm_zeta = hp.almxfl(maplist.processed_alms['unWISE'], np.divide(cltaug_dndz[i,:], Clgg_filter, out=np.zeros_like(Clgg_filter), where=Clgg_filter!=0))
		lssmap_filtered = hp.alm2map(dlm_zeta, lmax=maplist.lmax, nside=maplist.nside)
		np.save('data/cache/lssmap_filtered_dndz-%02d' % i, lssmap_filtered)
	else:
		lssmap_filtered = np.load('data/cache/lssmap_filtered_dndz-%02d.npy' % i)
	if not os.path.exists('data/cache/lssmap_filtered_dndz-%02d_mm.npy' % i):
		dlm_zeta = hp.almxfl(maplist.processed_alms['unWISE'], np.divide(cltaug_dndz_mm[i,:], Clgg_filter, out=np.zeros_like(Clgg_filter), where=Clgg_filter!=0))
		lssmap_filtered_mm = hp.alm2map(dlm_zeta, lmax=maplist.lmax, nside=maplist.nside)
		np.save('data/cache/lssmap_filtered_dndz-%02d_mm' % i, lssmap_filtered_mm)
	else:
		lssmap_filtered_mm = np.load('data/cache/lssmap_filtered_dndz-%02d_mm.npy' % i)
	noises['SMICA_dndz-%02d' % i] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_filter_SMICA, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_dndz[i,:])
	noises['COMMANDER_dndz-%02d' % i] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_filter_COMMANDER, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_dndz[i,:])
	noises['SMICA_dndz-%02d_mm' % i] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_filter_SMICA, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_dndz_mm[i,:])
	noises['COMMANDER_dndz-%02d_mm' % i] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_filter_COMMANDER, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_dndz_mm[i,:])
	if not os.path.exists('data/cache/Cls/' + 'SMICA_dndz-%02d.npy'% i):
		outmap_SMICA = -Tmap_filtered_SMICA * lssmap_filtered * maplist.mask * noises['SMICA_dndz-%02d' % i]
		recon_Cls['SMICA_dndz-%02d' % i] = maplist.alm2cl(hp.map2alm(outmap_SMICA, lmax=recon_lmax), maplist.fsky)
	else:
		recon_Cls['SMICA_dndz-%02d' % i] = np.load('data/cache/Cls/SMICA_dndz-%02d.npy'%i)
		if 'outmap_SMICA' in globals():
			del outmap_SMICA
	if not os.path.exists('data/cache/Cls/' + 'COMMANDER_dndz-%02d.npy' % i):
		outmap_COMMANDER = -Tmap_filtered_COMMANDER * lssmap_filtered * maplist.mask * noises['COMMANDER_dndz-%02d' % i]
		recon_Cls['COMMANDER_dndz-%02d' % i] = maplist.alm2cl(hp.map2alm(outmap_COMMANDER, lmax=recon_lmax), maplist.fsky)
	else:
		recon_Cls['COMMANDER_dndz-%02d' % i] = np.load('data/cache/Cls/COMMANDER_dndz-%02d.npy'%i)
		if 'outmap_COMMANDER' in globals():
			del outmap_COMMANDER
	if not os.path.exists('data/cache/Cls/' + 'SMICAxCOMMANDER_dndz-%02d.npy' % i):
		if 'outmap_SMICA' not in globals():
			outmap_SMICA = -Tmap_filtered_SMICA * lssmap_filtered * maplist.mask * noises['SMICA_dndz-%02d' % i]
		if 'outmap_COMMANDER' not in globals():
			outmap_COMMANDER = -Tmap_filtered_COMMANDER * lssmap_filtered * maplist.mask * noises['COMMANDER_dndz-%02d' % i]
		recon_Cls['SMICAxCOMMANDER_dndz-%02d' % i] = hp.alm2cl(hp.map2alm(outmap_SMICA, lmax=recon_lmax), hp.map2alm(outmap_COMMANDER, lmax=recon_lmax)) / maplist.fsky
	else:
		recon_Cls['SMICAxCOMMANDER_dndz-%02d' % i] = np.load('data/cache/Cls/SMICAxCOMMANDER_dndz-%02d.npy' % i)
	if not os.path.exists('data/cache/Cls/' + 'SMICA_dndz-%02d_mm.npy'% i):
		outmap_SMICA_mm = -Tmap_filtered_SMICA * lssmap_filtered_mm * maplist.mask * noises['SMICA_dndz-%02d_mm' % i]
		recon_Cls['SMICA_dndz-%02d_mm' % i] = maplist.alm2cl(hp.map2alm(outmap_SMICA_mm, lmax=recon_lmax), maplist.fsky)
	else:
		recon_Cls['SMICA_dndz-%02d_mm' % i] = np.load('data/cache/Cls/SMICA_dndz-%02d_mm.npy'%i)
		if 'outmap_SMICA_mm' in globals():
			del outmap_SMICA_mm
	if not os.path.exists('data/cache/Cls/' + 'COMMANDER_dndz-%02d_mm.npy' % i):
		outmap_COMMANDER_mm = -Tmap_filtered_COMMANDER * lssmap_filtered_mm * maplist.mask * noises['COMMANDER_dndz-%02d_mm' % i]
		recon_Cls['COMMANDER_dndz-%02d_mm' % i] = maplist.alm2cl(hp.map2alm(outmap_COMMANDER_mm, lmax=recon_lmax), maplist.fsky)
	else:
		recon_Cls['COMMANDER_dndz-%02d_mm' % i] = np.load('data/cache/Cls/COMMANDER_dndz-%02d_mm.npy'%i)
		if 'outmap_COMMANDER_mm' in globals():
			del outmap_COMMANDER_mm
	if not os.path.exists('data/cache/Cls/' + 'SMICAxCOMMANDER_dndz-%02d_mm.npy' % i):
		if 'outmap_SMICA_mm' not in globals():
			outmap_SMICA_mm = -Tmap_filtered_SMICA * lssmap_filtered_mm * maplist.mask * noises['SMICA_dndz-%02d_mm' % i]
		if 'outmap_COMMANDER_mm' not in globals():
			outmap_COMMANDER_mm = -Tmap_filtered_COMMANDER * lssmap_filtered_mm * maplist.mask * noises['COMMANDER_dndz-%02d_mm' % i]
		recon_Cls['SMICAxCOMMANDER_dndz-%02d_mm' % i] = hp.alm2cl(hp.map2alm(outmap_SMICA_mm, lmax=recon_lmax), hp.map2alm(outmap_COMMANDER_mm, lmax=recon_lmax)) / maplist.fsky
	else:
		recon_Cls['SMICAxCOMMANDER_dndz-%02d_mm' % i] = np.load('data/cache/Cls/SMICAxCOMMANDER_dndz-%02d_mm.npy' % i)
	if not os.path.exists('data/cache/histdata_multispec_COMMANDER.npy'):
		if 'outmap_COMMANDER' not in globals():
			outmap_COMMANDER = -Tmap_filtered_COMMANDER * lssmap_filtered * maplist.mask * noises['COMMANDER_dndz-%02d' % i]
		n_out_COMMANDER_dndz[i,:], _ = np.histogram(outmap_COMMANDER[np.where(maplist.mask!=0)], bins=bins_out_normprod_COMMANDER)
	else:
		n_out_COMMANDER_dndz[i,:] = np.load('data/cache/histdata_multispec_COMMANDER.npy')[i,:]
	if not os.path.exists('data/cache/histdata_multispec_COMMANDER_mm.npy'):
		if 'outmap_COMMANDER_mm' not in globals():
			outmap_COMMANDER_mm = -Tmap_filtered_COMMANDER * lssmap_filtered_mm * maplist.mask * noises['COMMANDER_dndz-%02d_mm' % i]
		n_out_COMMANDER_dndz_mm[i,:], _ = np.histogram(outmap_COMMANDER_mm[np.where(maplist.mask!=0)], bins=bins_out_normprod_COMMANDER)
	else:
		n_out_COMMANDER_dndz_mm[i,:] = np.load('data/cache/histdata_multispec_COMMANDER_mm.npy')[i,:]
	for key in ('100GHz', '143GHz', '217GHz', '353GHz'):
		Tcorr = pars.TCMB if key == '100GHz' else 1.
		noises[key+'_dndz-%02d' % i] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_filter_freq[key], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_dndz[i,:])
		noises[key+'_dndz-%02d_mm' % i] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_filter_freq[key], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_dndz_mm[i,:])
		for case in ('', '_noSMICA', '_noCOMMANDER', '_thermaldust', '_CIB'):
			if key == '353GHz':
				if not os.path.exists('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d.npy'%i):
					outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask_huge * noises['353GHz_dndz-%02d' % i]
					recon_Cls[key+case+'_CIBmask_dndz-%02d' % i] = maplist.alm2cl(hp.map2alm(outmap_freq, lmax=recon_lmax), maplist.fsky_huge)
				else:
					recon_Cls[key+case+'_CIBmask_dndz-%02d' % i] = np.load('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d.npy'%i)
					if 'outmap_freq' in globals():
						del outmap_freq
				if not os.path.exists('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d_mm.npy'%  i):
					outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask_huge * noises['353GHz_dndz-%02d_mm' % i]
					recon_Cls[key+case+'_CIBmask_dndz-%02d_mm' % i] = maplist.alm2cl(hp.map2alm(outmap_freq_mm, lmax=recon_lmax), maplist.fsky_huge)
				else:
					recon_Cls[key+case+'_CIBmask_dndz-%02d_mm' % i] = np.load('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d_mm.npy'%  i)
					if 'outmap_freq_mm' in globals():
						del outmap_freq_mm
				if key+case+'_CIBmask' in histkeys_353:
					if not os.path.exists('data/cache/histdata_353hist_' + key + '_CIBmask.npy'):
						if 'outmap_freq' not in globals():
							outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask_huge * noises['353GHz_dndz-%02d' % i]
						n_out_353hist_dndz[key+case+'_CIBmask'][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq, lmax=25)[np.where(maplist.mask_huge!=0)], bins=bins_353hist)
					else:
						n_out_353hist_dndz[key+case+'_CIBmask'][i,:] = np.load('data/cache/histdata_353hist_' + key + '_CIBmask.npy')[i,:]
					if not os.path.exists('data/cache/histdata_353hist_' + key + '_CIBmask_mm.npy'):
						if 'outmap_freq_mm' not in globals():
							outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask_huge * noises['353GHz_dndz-%02d_mm' % i]
						n_out_353hist_dndz_mm[key+case+'_CIBmask'][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq_mm, lmax=25)[np.where(maplist.mask_huge!=0)], bins=bins_353hist)
					else:
						n_out_353hist_dndz_mm[key+case+'_CIBmask'][i,:] = np.load('data/cache/histdata_353hist_' + key + '_CIBmask_mm.npy')[i,:]
				if key+case+'_CIBmask' in histkeys:
					if not os.path.exists('data/cache/histdata_multispec_' + key + '_CIBmask.npy'):
						if 'outmap_freq' not in globals():
							outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask_huge * noises['353GHz_dndz-%02d' % i]
						n_out_dndz[key+case+'_CIBmask'][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq, lmax=25)[np.where(maplist.mask_huge!=0)], bins=bins_freqhist)
					else:
						n_out_dndz[key+case+'_CIBmask'][i,:] = np.load('data/cache/histdata_multispec_' + key + '_CIBmask.npy')[i,:]
					if not os.path.exists('data/cache/histdata_multispec_' + key + '_CIBmask_mm.npy'):
						if 'outmap_freq_mm' not in globals():
							outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask_huge * noises['353GHz_dndz-%02d_mm' % i]
						n_out_dndz_mm[key+case+'_CIBmask'][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq_mm, lmax=25)[np.where(maplist.mask_huge!=0)], bins=bins_freqhist)
					else:
						n_out_dndz_mm[key+case+'_CIBmask'][i,:] = np.load('data/cache/histdata_multispec_' + key + '_CIBmask_mm.npy')[i,:]
			else:
				if case == '_CIB':
					continue
				if not os.path.exists('data/cache/Cls/' + key + case + '_dndz-%02d.npy' % i):
					outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask * noises[key+'_dndz-%02d' % i] / Tcorr
					recon_Cls[key+case+'_dndz-%02d' % i] = maplist.alm2cl(hp.map2alm(outmap_freq, lmax=recon_lmax), maplist.fsky)
				else:
					recon_Cls[key+case+'_dndz-%02d' % i] = np.load('data/cache/Cls/' + key + case + '_dndz-%02d.npy' % i)
					if 'outmap_freq' in globals():
						del outmap_freq
				if not os.path.exists('data/cache/Cls/' + key + case + '_dndz-%02d_mm.npy' % i):
					outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask * noises[key+'_dndz-%02d_mm' % i] / Tcorr
					recon_Cls[key+case+'_dndz-%02d_mm' % i] = maplist.alm2cl(hp.map2alm(outmap_freq_mm, lmax=recon_lmax), maplist.fsky)
				else:
					recon_Cls[key+case+'_dndz-%02d_mm' % i] = np.load('data/cache/Cls/' + key + case + '_dndz-%02d_mm.npy' % i)
					if 'outmap_freq_mm' in globals():
						del outmap_freq_mm
				if key+case in histkeys:
					if not os.path.exists('data/cache/histdata_multispec_' + key + '.npy'):
						if 'outmap_freq' not in globals():
							outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask * noises[key+'_dndz-%02d' % i] / Tcorr
						n_out_dndz[key+case][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq, lmax=25)[np.where(maplist.mask!=0)], bins=bins_freqhist)
					else:
						n_out_dndz[key+case][i,:] = np.load('data/cache/histdata_multispec_' + key + '.npy')[i,:]
					if not os.path.exists('data/cache/histdata_multispec_' + key + '_mm.npy'):
						if 'outmap_freq_mm' not in globals():
							outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask * noises[key+'_dndz-%02d_mm' % i] / Tcorr
						n_out_dndz_mm[key+case][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq_mm, lmax=25)[np.where(maplist.mask!=0)], bins=bins_freqhist)
					else:
						n_out_dndz_mm[key+case][i,:] = np.load('data/cache/histdata_multispec_' + key + '_mm.npy')[i,:]
				if key+case == '217GHz':
					if not os.path.exists('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d.npy'% i):
						outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask_huge * noises['217GHz_dndz-%02d' % i]
						recon_Cls[key+case+'_CIBmask_dndz-%02d' % i] = maplist.alm2cl(hp.map2alm(outmap_freq, lmax=recon_lmax), maplist.fsky_huge)
					else:
						recon_Cls[key+case+'_CIBmask_dndz-%02d' % i] = np.load('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d.npy'% i)
						if 'outmap_freq' in globals():
							del outmap_freq
					if not os.path.exists('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d_mm.npy'% i):
						outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask_huge * noises['217GHz_dndz-%02d_mm' % i]
						recon_Cls[key+case+'_CIBmask_dndz-%02d_mm' % i] = maplist.alm2cl(hp.map2alm(outmap_freq_mm, lmax=recon_lmax), maplist.fsky_huge)
					else:
						recon_Cls[key+case+'_CIBmask_dndz-%02d_mm' % i] = np.load('data/cache/Cls/' + key + case + '_CIBmask_dndz-%02d_mm.npy'% i)
						if 'outmap_freq_mm' in globals():
							del outmap_freq_mm
					if not os.path.exists('data/cache/histdata_multispec_' + key + '_CIBmask.npy'):
						if 'outmap_freq' not in globals():
							outmap_freq = -Tmap_filtered_freq[key+case] * lssmap_filtered * maplist.mask_huge * noises['217GHz_dndz-%02d' % i]
						n_out_dndz[key+case+'_CIBmask'][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq, lmax=25)[np.where(maplist.mask_huge!=0)], bins=bins_freqhist)					
					else:
						n_out_dndz[key+case+'_CIBmask'][i,:] = np.load('data/cache/histdata_multispec_' + key + '_CIBmask.npy')[i,:]
					if not os.path.exists('data/cache/histdata_multispec_' + key + '_CIBmask_mm.npy'):
						if 'outmap_freq_mm' not in globals():
							outmap_freq_mm = -Tmap_filtered_freq[key+case] * lssmap_filtered_mm * maplist.mask_huge * noises['217GHz_dndz-%02d_mm' % i]
						n_out_dndz_mm[key+case+'_CIBmask'][i,:], _ = np.histogram(maplist.lowpass_filter(outmap_freq_mm, lmax=25)[np.where(maplist.mask_huge!=0)], bins=bins_freqhist)					
					else:
						n_out_dndz_mm[key+case+'_CIBmask'][i,:] = np.load('data/cache/histdata_multispec_' + key + '_CIBmask_mm.npy')[i,:]


# np.save('data/cache/histdata_multispec_COMMANDER', n_out_COMMANDER_dndz)
# np.save('data/cache/histdata_multispec_COMMANDER_mm', n_out_COMMANDER_dndz_mm)

# for key in [k for k in recon_Cls if '_dndz-' in k]:
# 	np.save('data/cache/Cls/' + key, recon_Cls[key])

# for key in histkeys:
# 	np.save('data/cache/histdata_multispec_'+key, n_out_dndz[key])
# 	np.save('data/cache/histdata_multispec_'+key+'_mm', n_out_dndz_mm[key])

# for key in histkeys_353:
# 	np.save('data/cache/histdata_353hist_'+key, n_out_353hist_dndz[key])
# 	np.save('data/cache/histdata_353hist_'+key+'_mm', n_out_353hist_dndz_mm[key])

print('    Computing windowed velocity for dN/dz realizations')
clgg = maplist.Cls['unWISE'].copy()
keys = ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz')
cltt = {key : maplist.Cls[key] for key in keys}
terms = {key : np.zeros((100, 1)) for key in keys}
terms_with_me_entry = {key : np.zeros((100, chis.size)) for key in keys}
terms_with_mm_me_entry = {key : np.zeros((100, chis.size)) for key in keys}
ell_const = 2
for l2 in np.arange(spectra_lmax):
	Pmm_at_ellprime = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (l2+0.5)/chis[::-1], grid=True), axis=1))
	Pem_at_ellprime = Pmm_at_ellprime * np.diagonal(np.flip(bias_e2(results.redshift_at_comoving_radial_distance(chis), (l2+0.5)/chis[::-1]), axis=1))  # Convert Pmm to Pem
	Pem_at_ellprime_at_zbar = Pem_at_ellprime[zbar_index]
	for l1 in np.arange(np.abs(l2-ell_const),l2+ell_const+1):
		if l1 > spectra_lmax-1 or l1 <2:   #triangle rule
			continue
		gamma_ksz_common = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell_const+1)/(4*np.pi))*wigner_symbol(ell_const, l1, l2)
		gamma_ksz = gamma_ksz_common*cltaug_dndz[:,l2]
		for key in keys:
			term_with_me_entry = (gamma_ksz*gamma_ksz/(cltt[key][l1]*clgg[l2]))[:,np.newaxis] * (Pem_at_ellprime/Pem_at_ellprime_at_zbar)[np.newaxis,:]  # #dndz by chi array, rows = dndz realization
			term_with_mm_me_entry = (gamma_ksz*gamma_ksz/(cltt[key][l1]*clgg[l2]))[:,np.newaxis] * (Pmm_at_ellprime/Pem_at_ellprime_at_zbar)[np.newaxis,:]
			term_entry = (gamma_ksz*gamma_ksz/(cltt[key][l1]*clgg[l2]))[:,np.newaxis]
			if False not in np.isfinite(term_entry):
				terms[key] += term_entry
				terms_with_me_entry[key] += term_with_me_entry
				terms_with_mm_me_entry[key] += term_with_mm_me_entry

ratio_me_me = {key : np.zeros((100, chis.size)) for key in keys}
ratio_mm_me = {key : np.zeros((100, chis.size)) for key in keys}
window_v = {key : np.zeros((100, chis.size)) for key in keys}
window_v_mm = {key : np.zeros((100, chis.size)) for key in keys}
clv_windowed_dndz = {key : np.zeros((100, velocity_compute_ells.size)) for key in keys}
clv_windowed_mm_dndz = {key : np.zeros((100, velocity_compute_ells.size)) for key in keys}

for key in keys:
	ratio_me_me[key] = terms_with_me_entry[key] / terms[key]
	ratio_mm_me[key] = terms_with_mm_me_entry[key] / terms[key]
	window_v[key] = np.nan_to_num(  ( 1/bin_width ) * ( chibar**2 / chis**2 )[np.newaxis,:] * ( galaxy_window_dndz / galaxy_window_dndz[:,zbar_index,np.newaxis] ) * ( (1+results.redshift_at_comoving_radial_distance(chis))**2 / (1+results.redshift_at_comoving_radial_distance(chibar))**2 )[np.newaxis,:] * ratio_me_me[key]  )
	window_v_mm[key] = np.nan_to_num(  ( 1/bin_width ) * ( chibar**2 / chis**2 )[np.newaxis,:] * ( galaxy_window_dndz / galaxy_window_dndz[:,zbar_index,np.newaxis] ) * ( (1+results.redshift_at_comoving_radial_distance(chis))**2 / (1+results.redshift_at_comoving_radial_distance(chibar))**2 )[np.newaxis,:] * ratio_mm_me[key]  )
	for i in np.arange(velocity_compute_ells.size):
		clv_windowed_dndz[key][:,i] = np.trapz(window_v[key]*np.trapz(window_v[key][:,np.newaxis,:]*clv[i,np.newaxis,:,:], chis,axis=2), chis, axis=1)
		clv_windowed_mm_dndz[key][:,i] = np.trapz(window_v_mm[key]*np.trapz(window_v_mm[key][:,np.newaxis,:]*clv[i,np.newaxis,:,:], chis,axis=2), chis, axis=1)

print('Computing statistical errors for P_me case')
dlm_zeta = hp.almxfl(maplist.processed_alms['unWISE'], np.divide(cltaug_fiducial, Clgg_filter, out=np.zeros_like(Clgg_filter), where=Clgg_filter!=0))
lssmap_filtered = hp.alm2map(dlm_zeta, lmax=maplist.lmax, nside=maplist.nside)

n_stat_iter = 1000
statistical_Cls = {key : np.zeros((n_stat_iter, velocity_compute_ells.max()+1)) for key in ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz')}
for key in ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz'):
	print('    Computing realizations for ' + key)
	vspec = interp1d(velocity_compute_ells, clv_windowed[key], bounds_error=False, fill_value=0.)(np.arange(velocity_compute_ells.max()+1))
	for i in np.arange(n_stat_iter):
		if os.path.exists('data/cache/Cls/stat/stat_v_plus_n_'+key+'_%03d.npy'%i):
			statistical_Cls[key][i,:] = np.load('data/cache/Cls/stat/stat_v_plus_n_'+key+'_%03d.npy'%i)
			continue
		Tcorr = pars.TCMB if key == '100GHz' else 1.
		vmap = hp.synfast(vspec, lmax=velocity_compute_ells.max(), nside=maplist.nside)
		Tlms = hp.synalm(maplist.Cls[key], lmax=maplist.lmax)
		ClTT_filter = maplist.Cls[key].copy()[:maplist.lmax+1]
		ClTT_filter[:100] = 1e15
		Tmap_filtered = hp.alm2map(hp.almxfl(Tlms, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), lmax=maplist.lmax, nside=maplist.nside)
		outmap_noiserecon = -Tmap_filtered * lssmap_filtered * maplist.mask * noises[key] / Tcorr
		statistical_Cls[key][i,:] = hp.anafast((vmap + outmap_noiserecon) * maplist.mask, lmax=velocity_compute_ells.max()) / maplist.fsky
		np.save('data/cache/Cls/stat/stat_v_plus_n_'+key+'_%03d'%i, statistical_Cls[key][i,:])
		if (i % 100 == 0) and (i > 0):
			print('        Completed %d of %d' % (i, n_stat_iter))
	print('        Completed %d of %d' % (n_stat_iter, n_stat_iter))

print('Computing statistical errors for P_mm case')
dlm_zeta = hp.almxfl(maplist.processed_alms['unWISE'], np.divide(cltaug_fiducial_mm, Clgg_filter, out=np.zeros_like(Clgg_filter), where=Clgg_filter!=0))
lssmap_filtered = hp.alm2map(dlm_zeta, lmax=maplist.lmax, nside=maplist.nside)

statistical_Cls_mm = {key : np.zeros((n_stat_iter, velocity_compute_ells.max()+1)) for key in ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz')}
for key in ('COMMANDER', '100GHz', '143GHz', '217GHz', '353GHz'):
	print('    Computing realizations for ' + key)
	vspec = interp1d(velocity_compute_ells, clv_windowed_mm[key], bounds_error=False, fill_value=0.)(np.arange(velocity_compute_ells.max()+1))
	for i in np.arange(n_stat_iter):
		if os.path.exists('data/cache/Cls/stat/stat_v_plus_n_mm_'+key+'_%03d.npy'%i):
			statistical_Cls_mm[key][i,:] = np.load('data/cache/Cls/stat/stat_v_plus_n_mm_'+key+'_%03d.npy'%i)
			continue
		Tcorr = pars.TCMB if key == '100GHz' else 1.
		vmap = hp.synfast(vspec, lmax=velocity_compute_ells.max(), nside=maplist.nside)
		Tlms = hp.synalm(maplist.Cls[key], lmax=maplist.lmax)
		ClTT_filter = maplist.Cls[key].copy()[:maplist.lmax+1]
		ClTT_filter[:100] = 1e15
		Tmap_filtered = hp.alm2map(hp.almxfl(Tlms, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), lmax=maplist.lmax, nside=maplist.nside)
		outmap_noiserecon = -Tmap_filtered * lssmap_filtered * maplist.mask * noises[key+'_mm'] / Tcorr
		statistical_Cls_mm[key][i,:] = hp.anafast((vmap + outmap_noiserecon) * maplist.mask, lmax=velocity_compute_ells.max()) / maplist.fsky
		np.save('data/cache/Cls/stat/stat_v_plus_n_mm_'+key+'_%03d'%i, statistical_Cls_mm[key][i,:])
		if (i % 100 == 0) and (i > 0):
			print('        Completed %d of %d' % (i, n_stat_iter))
	print('        Completed %d of %d' % (n_stat_iter, n_stat_iter))

print('Errors post-processing: computing plot errors')
# 1) Get the v_fiducial and the v_fiducial_mm lines and put in their respective errors properly propagated
# 2) At each ell choose the min/max range values using both sets. For interest check if it's one for bottom one for top.
make_arr_dndz = lambda arr, key, mmtag : np.array([arr[key+'_dndz-%02d%s'%(i, mmtag)] for i in np.arange(100)])
errors = {}
for key in ('SMICA', 'COMMANDER', 'SMICAxCOMMANDER', '100GHz', '143GHz', '217GHz', '353GHz'):
	Tcorr = pars.TCMB**2 if key == '100GHz' else 1.
	for case in ('', '_noSMICA', '_noCOMMANDER', '_thermaldust', '_CIB'):
		if (key in ('SMICA', 'COMMANDER', 'SMICAxCOMMANDER')) and (case != ''):
			continue
		if key == '353GHz':
			rfid    = recon_Cls[key+case+'_CIBmask']
			rfid_mm = recon_Cls[key+case+'_mm_CIBmask']
			rdndz_err    = np.std(make_arr_dndz(recon_Cls, key+case+'_CIBmask', ''), axis=0)
			rdndz_err_mm = np.std(make_arr_dndz(recon_Cls, key+case+'_CIBmask', '_mm'), axis=0)
			negbias_err = np.where((rfid - rfid_mm) > 0, rfid - rfid_mm, 0.)  # Incoporate offset from fiducial to fiducial mm in correct direction for error: to positive side if mm reconstruction is higher, to negative if lower
			posbias_err = np.where((rfid - rfid_mm) < 0, rfid_mm - rfid, 0.)
			errors[key+case+'_CIBmask'] = ( np.min([rdndz_err_mm, rdndz_err], axis=0) + negbias_err, np.max([rdndz_err_mm, rdndz_err], axis=0) + posbias_err )  # Tuple for errorbars
		else:
			if case == '_CIB':
				continue
			rfid    = recon_Cls[key+case]
			rfid_mm = recon_Cls[key+case+'_mm']
			rdndz_err    = np.std(make_arr_dndz(recon_Cls, key+case, ''), axis=0)
			rdndz_err_mm = np.std(make_arr_dndz(recon_Cls, key+case, '_mm'), axis=0)
			negbias_err = np.where((rfid - rfid_mm) > 0, rfid - rfid_mm, 0.)  # Incoporate offset from fiducial to fiducial mm in correct direction for error: to positive side if mm reconstruction is higher, to negative if lower
			posbias_err = np.where((rfid - rfid_mm) < 0, rfid_mm - rfid, 0.)
			errors[key+case] = ( np.min([rdndz_err_mm, rdndz_err], axis=0) + negbias_err, np.max([rdndz_err_mm, rdndz_err], axis=0) + posbias_err )  # Tuple for errorbars
			if key+case == '217GHz':
				rfid    = recon_Cls[key+case+'_CIBmask']
				rfid_mm = recon_Cls[key+case+'_mm_CIBmask']
				rdndz_err    = np.std(make_arr_dndz(recon_Cls, key+case+'_CIBmask', ''), axis=0)
				rdndz_err_mm = np.std(make_arr_dndz(recon_Cls, key+case+'_CIBmask', '_mm'), axis=0)
				negbias_err = np.where((rfid - rfid_mm) > 0, rfid - rfid_mm, 0.)  # Incoporate offset from fiducial to fiducial mm in correct direction for error: to positive side if mm reconstruction is higher, to negative if lower
				posbias_err = np.where((rfid - rfid_mm) < 0, rfid_mm - rfid, 0.)
				errors[key+case+'_CIBmask'] = ( np.min([rdndz_err_mm, rdndz_err], axis=0) + negbias_err, np.max([rdndz_err_mm, rdndz_err], axis=0) + posbias_err )  # Tuple for errorbars
	# Velocity errors
	if key in ('SMICA', 'SMICAxCOMMANDER'):
		continue
	vfid         = clv_windowed[key]    + (noises[key]       / Tcorr)
	vfid_mm      = clv_windowed_mm[key] + (noises[key+'_mm'] / Tcorr)
	vdndz_err    = np.std(clv_windowed_dndz[key]    + np.array([noises[key+'_dndz-%02d'%i]    / Tcorr for i in np.arange(100)])[:,np.newaxis], axis=0)
	vdndz_err_mm = np.std(clv_windowed_mm_dndz[key] + np.array([noises[key+'_dndz-%02d_mm'%i] / Tcorr for i in np.arange(100)])[:,np.newaxis], axis=0)
	vstat_err    = np.std(statistical_Cls[key][:,velocity_compute_ells], axis=0)
	vstat_err_mm = np.std(statistical_Cls_mm[key][:,velocity_compute_ells], axis=0)
	verr    = np.sqrt(vdndz_err**2 + vstat_err**2)
	verr_mm = np.sqrt(vdndz_err_mm**2 + vstat_err_mm**2)
	errors['v_dndz_min'+key] = np.min([vfid_mm - vdndz_err_mm, vfid - vdndz_err], axis=0)
	errors['v_dndz_max'+key] = np.max([vfid_mm + vdndz_err_mm, vfid + vdndz_err], axis=0)
	errors['v_total_min'+key] = np.min([vfid_mm - verr_mm, vfid - verr], axis=0)
	errors['v_total_max'+key] = np.max([vfid_mm + verr_mm, vfid + verr], axis=0)
	for vcase in ('v_dndz_max', 'v_dndz_min', 'v_total_max', 'v_total_min'):
		errors[vcase+key] = interp1d(velocity_compute_ells, errors[vcase+key], bounds_error=False, fill_value=0.)(np.arange(velocity_compute_ells.max()+1))

# Plot-related variables: v_interp for shorthand, and plot/mask value assignments for recon_out plot
v_COMMANDER_interp = interp1d(velocity_compute_ells,clv_windowed['COMMANDER']+noises['COMMANDER'],bounds_error=False,fill_value=0.)(np.arange(velocity_compute_ells.max()+1))
v_100GHz_interp = interp1d(velocity_compute_ells,clv_windowed['100GHz']+(noises['100GHz']/pars.TCMB**2),bounds_error=False,fill_value=0.)(np.arange(velocity_compute_ells.max()+1))
v_143GHz_interp = interp1d(velocity_compute_ells,clv_windowed['143GHz']+noises['143GHz'],bounds_error=False,fill_value=0.)(np.arange(velocity_compute_ells.max()+1))
v_217GHz_interp = interp1d(velocity_compute_ells,clv_windowed['217GHz']+noises['217GHz'],bounds_error=False,fill_value=0.)(np.arange(velocity_compute_ells.max()+1))
v_353GHz_interp = interp1d(velocity_compute_ells,clv_windowed['353GHz']+noises['353GHz'],bounds_error=False,fill_value=0.)(np.arange(velocity_compute_ells.max()+1))

reconplot_minval = np.min([(maplist.stored_maps['recon_SMICA']*maplist.mask).min(),(maplist.stored_maps['recon_COMMANDER']*maplist.mask).min()])
reconplot_maxval = np.max([(maplist.stored_maps['recon_SMICA']*maplist.mask).max(),(maplist.stored_maps['recon_COMMANDER']*maplist.mask).max()])
plotmask = maplist.mask.copy()
plotmask[np.where(maplist.mask==0)] = np.nan

print('Computing dipoles')
n_dipole_iter = 1000
Clgg_filter = maplist.Cls['unWISE'].copy()[:maplist.lmax+1]
Clgg_filter[:100] = 1e15
inverse_fiducial_mask = -(maplist.mask-1)
for case in ('COMMANDER','SMICA','100GHz', '143GHz', '217GHz'):
	Tcorr = pars.TCMB if case == '100GHz' else 1
	ClTT_filter = maplist.Cls[case].copy()[:maplist.lmax+1]
	ClTT_filter[:100] = 1e15
	print('    case: ' + case)
	for i in np.arange(n_dipole_iter):
		if (i % 100 == 0) and (i > 0):
			print('        Completed %d of %d' % (i, n_dipole_iter))
		if os.path.exists('data/cache/stat/dipole/dipole_fiducial_%s_%03d.npy'%(case,i)):
			continue
		Tlms = hp.synalm(maplist.Cls[case], lmax=maplist.lmax)
		lsslms = hp.synalm(maplist.Cls['unWISE'], lmax=maplist.lmax)
		Tmap_filtered = hp.alm2map(hp.almxfl(Tlms, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), lmax=maplist.lmax, nside=maplist.nside)
		lssmap_filtered = hp.alm2map(hp.almxfl(lsslms, np.divide(cltaug_fiducial, Clgg_filter, out=np.zeros_like(Clgg_filter), where=Clgg_filter!=0)), lmax=maplist.lmax, nside=maplist.nside)
		outmap_noiserecon = -Tmap_filtered * lssmap_filtered * inverse_fiducial_mask * noises[case]
		recon_inpainted = reconstructions[case] + outmap_noiserecon
		dipole_alms = hp.almxfl(hp.map2alm(recon_inpainted,lmax=1),[0,1])
		np.save('data/cache/stat/dipole/dipole_fiducial_%s_%03d.npy'%(case,i), dipole_alms)

dipole_alms = {key : np.zeros((n_dipole_iter, 3), dtype=np.complex128) for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
for case in ('COMMANDER','SMICA','100GHz', '143GHz', '217GHz'):
	for i in np.arange(n_dipole_iter):
		dipole_alms[case][i,:] = np.load('data/cache/stat/dipole/dipole_fiducial_%s_%03d.npy'%(case,i))

measured_dipoles = {key : np.mean(dipole_alms[key], axis=0) for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
mean_dipole = {key : hp.fit_dipole(hp.alm2map(measured_dipoles[key],nside=2,lmax=1))[1] for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
dipoles = {key : np.zeros((n_dipole_iter, 3)) for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
for case in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']:
	for i in np.arange(n_dipole_iter):
		dipoles[case][i,:] = hp.fit_dipole(hp.alm2map(dipole_alms[case][i,:],nside=2,lmax=1))[1]

monopole_maps = {key : hp.fit_dipole((np.repeat(np.mean(reconstructions[key][np.where(maplist.mask!=0)]), hp.nside2npix(maplist.nside))) * maplist.mask)[1] for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}

lat_mean = {key : hp.vec2ang(mean_dipole[key])[0][0] for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
lon_mean = {key : hp.vec2ang(mean_dipole[key])[1][0] for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
lat_check = {key : hp.vec2ang(dipoles[key])[0] for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
lon_check = {key : hp.vec2ang(dipoles[key])[1] for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}

dipole_angseps = {key : np.arccos(np.sin(lat_mean[key])*np.sin(lat_check[key]) + np.cos(lat_mean[key])*np.cos(lat_check[key])*np.cos(lon_mean[key]-lon_check[key])) for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}
# Plot a circle at z=0. Then use rot keyword to rotate.
thetas   = {key : np.degrees(np.std(dipole_angseps[key]))*np.sin(np.linspace(0,2*np.pi,100)) for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}  # lat
phis = {key : np.degrees(np.std(dipole_angseps[key]))*np.cos(np.linspace(0,2*np.pi,100)) for key in ['COMMANDER','SMICA','100GHz', '143GHz', '217GHz']}  # lon


### Plotting
# Common functions
bandpowers = lambda spectrum : np.array([spectrum[2:][1+(5*i):1+(5*(i+1))].mean() for i in np.arange(spectrum.size//5)])
bandpowers_errbar = lambda spectrum : np.array([np.array([spectrum[j][2:][1+(5*i):1+(5*(i+1))].mean() for i in np.arange(spectrum[j].size//5)]) for j in np.arange(len(spectrum))])
pixel_scaling_masked = lambda distribution, FSKY : (12*2048**2) * FSKY * (distribution / simps(distribution))
c = mpl.colors.ListedColormap(['darkred', 'gold'])
cmap_dipole = mpl.colors.ListedColormap(['#AFAFAF', '#EEEEEE'])
x_ells = bandpowers(np.arange(recon_lmax+3))
linecolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
darken = lambda color, amount :  colorsys.hls_to_rgb(colorsys.rgb_to_hls(*mc.to_rgb(color))[0], 1 - amount * (1 - colorsys.rgb_to_hls(*mc.to_rgb(color))[1]), colorsys.rgb_to_hls(*mc.to_rgb(color))[2])

## Fiducial reconstruction maps: needs to be cropped by 3rd party program, hp.mollview is uncooperative with matplotlib formatting for whitespace.
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,15))
plt.axes(ax1)
hp.mollview(maplist.stored_maps['recon_SMICA']*plotmask, title=r'SMICA x unWISE Reconstruction $\left(\ell_{\mathrm{max}}\leq 25\right)$', unit=r'$\frac{v}{c}$', min=reconplot_minval, max=reconplot_maxval, hold=True)
plt.axes(ax2)
hp.mollview(maplist.stored_maps['recon_COMMANDER']*plotmask, title=r'COMMANDER x unWISE Reconstruction $\left(\ell_{\mathrm{max}}\leq 25\right)$',unit=r'$\frac{v}{c}$', min=reconplot_minval, max=reconplot_maxval, hold=True)
plt.tight_layout()
plt.savefig(outdir+'recon_outputs')

# SNR plot, main result for fiducial reconstructions
plt.figure()
plt.errorbar(x_ells, bandpowers(recon_Cls['SMICA']), yerr=bandpowers_errbar(errors['SMICA']), capsize=3, label='SMICA x unWISE Reconstruction', ls='None', marker='x', zorder=100, color=linecolors[0])
plt.errorbar(x_ells, bandpowers(recon_Cls['COMMANDER']), yerr=bandpowers_errbar(errors['COMMANDER']), capsize=3, label='COMMANDER x unWISE Reconstruction', ls='None', marker='x', zorder=100, color=linecolors[3])
plt.errorbar(x_ells, bandpowers(recon_Cls['SMICAxCOMMANDER']), yerr=bandpowers_errbar(errors['SMICAxCOMMANDER']), capsize=3, label='SMICA x COMMANDER Reconstructions', ls='None', marker='x', zorder=100, color=linecolors[4])
plt.semilogy(np.arange(100), np.repeat(noises['COMMANDER'],100), c='k',label='Theory Noise', ls='--', zorder=10, lw=2)
plt.xlim([2, 27])
plt.semilogy(np.arange(velocity_compute_ells.max()+1), v_COMMANDER_interp, color=darken(linecolors[1],1.2),lw=2,label='Windowed velocity + noise')
plt.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_dndz_minCOMMANDER'], errors['v_dndz_maxCOMMANDER'], color=linecolors[1], alpha=0.75)
plt.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_total_minCOMMANDER'], errors['v_total_maxCOMMANDER'], color=linecolors[1], alpha=0.35)
order = [2,3,4,1,0]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
plt.title('Planck x unWISE Reconstruction')
plt.tight_layout()
plt.savefig(outdir+'signal_noise_gauss.png')

# 2pt frequency statistics for 100, 143, 217 GHz with fiducial mask
fig, (ax100, ax143, ax217) = plt.subplots(1,3,figsize=(18,6))
for freq, ax in zip([100, 143, 217], [ax100, ax143, ax217]):
	Tcorr = pars.TCMB**2 if freq == 100 else 1.
	v_freq = {100 : v_100GHz_interp, 143 : v_143GHz_interp, 217 : v_217GHz_interp}[freq]
	ax.errorbar(x_ells, bandpowers(recon_Cls['%dGHz' % freq]), label='%d GHz x unWISE' % freq, yerr=bandpowers_errbar(errors['%dGHz' % freq]), capsize=3, ls='None', marker='x', zorder=100, color=linecolors[0])
	ax.errorbar(x_ells, bandpowers(recon_Cls['%dGHz_noSMICA' % freq]), label='%d GHz - SMICA x unWISE' % freq, yerr=bandpowers_errbar(errors['%dGHz_noSMICA' % freq]), capsize=3, ls='None', marker='x', zorder=100, color=linecolors[2])
	ax.errorbar(x_ells, bandpowers(recon_Cls['%dGHz_noCOMMANDER' % freq]), label='%d GHz - COMMANDER x unWISE' % freq, yerr=bandpowers_errbar(errors['%dGHz_noCOMMANDER' % freq]), capsize=3, ls='None', marker='x', zorder=100, color=linecolors[3])
	ax.errorbar(x_ells, bandpowers(recon_Cls['%dGHz_thermaldust' % freq]), label='Thermal dust x unWISE', yerr=bandpowers_errbar(errors['%dGHz_thermaldust' % freq]), capsize=3, ls='None', marker='x', zorder=100, color=linecolors[4])
	ax.semilogy(np.arange(100), np.repeat(noises['%dGHz'%freq],100)/Tcorr, c='k', label='Theory Noise', ls='--', zorder=10, lw=2)
	ax.semilogy(np.arange(velocity_compute_ells.max()+1), v_freq, color=darken(linecolors[1],1.2), lw=2, label='Windowed velocity + noise')
	ax.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_dndz_min%dGHz'%freq], errors['v_dndz_max%dGHz'%freq], color=linecolors[1], alpha=0.75)
	ax.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_total_min%dGHz'%freq], errors['v_total_max%dGHz'%freq], color=linecolors[1], alpha=0.35)
	ax.set_title('Reconstructions at %d GHz' % freq)
	ax.set_xlim([2, 27])
	ax.set_xlabel(r'$\ell$')
	ax.set_ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
ymin = ax100.get_ylim()[0]
for ax in (ax100, ax143, ax217):
	ax.set_ylim([ymin, 1.8e-6])
	order = [2,3,4,5,1,0]
	handles, labels = ax.get_legend_handles_labels()
	ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.tight_layout()
plt.savefig(outdir+'freq_recon_fiducialmask')

# 2pt statistics at 217 GHz for the two different masks
plt.figure()
plt.errorbar(x_ells, bandpowers(recon_Cls['217GHz']), yerr=bandpowers_errbar(errors['217GHz']), capsize=3, label='Fiducial mask', ls='None', marker='x', zorder=100, color=linecolors[0])
plt.errorbar(x_ells, bandpowers(recon_Cls['217GHz_CIBmask']), yerr=bandpowers_errbar(errors['217GHz_CIBmask']), capsize=3, label='Large mask', ls='None', marker='x', zorder=100, color=linecolors[2])
plt.semilogy(np.arange(100), np.repeat(noises['217GHz'],100), c='k',label='Theory Noise', ls='--', zorder=10, lw=2)
plt.semilogy(np.arange(velocity_compute_ells.max()+1), v_217GHz_interp, color=darken(linecolors[1],1.2), lw=2, label='Windowed velocity + noise')
plt.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_dndz_min217GHz'], errors['v_dndz_max217GHz'], color=linecolors[1], alpha=0.75)
plt.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_total_min217GHz'], errors['v_total_max217GHz'], color=linecolors[1], alpha=0.35)
plt.xlim([2, 27])
order = [2,3,1,0]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$     ',rotation=0,fontsize=16)
plt.title('Reconstructions at 217 GHz')
plt.tight_layout()
plt.savefig(outdir+'217_recon_masks')

# 2pt statistics at 353 GHz for the different foregrounds
plt.figure()
plt.errorbar(x_ells, bandpowers(recon_Cls['353GHz_CIBmask']), yerr=bandpowers_errbar(errors['353GHz_CIBmask']), capsize=3, label='353 GHz x unWISE', ls='None', marker='x', zorder=100, color=linecolors[0])
plt.errorbar(x_ells, bandpowers(recon_Cls['353GHz_thermaldust_CIBmask']), yerr=bandpowers_errbar(errors['353GHz_thermaldust_CIBmask']), capsize=3, label='thermal dust x unWISE', ls='None', marker='x', zorder=100, color=linecolors[2])
plt.errorbar(x_ells, bandpowers(recon_Cls['353GHz_CIB_CIBmask']), yerr=bandpowers_errbar(errors['353GHz_CIB_CIBmask']), capsize=3, label='CIB x unWISE', ls='None', marker='x', zorder=100, color=linecolors[3])
plt.semilogy(np.arange(100), np.repeat(noises['353GHz'],100), c='k',label='Theory Noise', ls='--', zorder=10, lw=2)
plt.semilogy(np.arange(velocity_compute_ells.max()+1), v_353GHz_interp, color=darken(linecolors[1],1.2), lw=2, label='Windowed velocity + noise')
plt.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_dndz_min353GHz'], errors['v_dndz_max353GHz'], color=linecolors[1], alpha=0.75)
plt.fill_between(np.arange(velocity_compute_ells.max()+1), errors['v_total_min353GHz'], errors['v_total_max353GHz'], color=linecolors[1], alpha=0.35)
plt.xlim([2, 27])
order = [2,3,4,1,0]
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$     ',rotation=0,fontsize=16)
plt.title('Reconstructions at 353 GHz with large mask')
plt.tight_layout()
plt.savefig(outdir+'353_recon_masks')

# dN/dz
plt.figure()
plt.plot(redshifts, dNdz_fiducial / simps(dNdz_fiducial), lw=2, label=r'Best-fit $\frac{d\mathrm{N}}{dz}$',zorder=200,color='red')
plt.plot(redshifts, dNdz_realization[0,:] / simps(dNdz_realization[0,:]), lw=0.5, color='gray', alpha=0.75, label=r'Individual $\frac{d\mathrm{N}}{dz}$ realizations')
for i in np.arange(1,100):
	plt.plot(redshifts, dNdz_realization[i,:] / simps(dNdz_realization[i,:]), lw=0.5, color='gray', alpha=0.75)
plt.title('unWISE redshift distribution')
plt.xlabel(r'$z$')
plt.xlim([redshifts.min(), redshifts.max()])
leg = plt.legend()
for lh in leg.legendHandles[1:]:
	lh.set_alpha(1.)
	lh.set_linewidth(1.)
plt.tight_layout()
plt.savefig(outdir+'dndz')

# Cltaug
plt.figure()
plt.semilogy(cltaug_fiducial, lw=2, color='red', zorder=200, label=r'$C_\ell^{\tau\mathrm{g}}$ for best-fit $\frac{d\mathrm{N}}{dz}$')
plt.semilogy(cltaug_dndz[0,:], lw=0.5, color='gray', alpha=0.75, label=r'$C_\ell^{\tau\mathrm{g}}$ for $\frac{d\mathrm{N}}{dz}$ realizations')
for i in np.arange(1,100):
	plt.semilogy(cltaug_dndz[i,:], lw=0.5, color='gray', alpha=0.75)
plt.title('unWISE blue modelled tau-galaxy cross-spectrum')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\tau\mathrm{g}}$')
plt.xlim([101,4000])
leg = plt.legend()
for lh in leg.legendHandles[1:]:
	lh.set_alpha(1.)
	lh.set_linewidth(1.)
plt.tight_layout()
plt.savefig(outdir+'cltaug')

# Fiducial mask map
plt.figure()
hp.mollview(maplist.mask, cmap=c, cbar=False, title=r'Fiducial Reconstruction Mask $\left(f_{\mathrm{sky}}=%.2f\right)$' % maplist.fsky)
plt.savefig(outdir+'Mask input')

# Large mask map
plt.figure()
hp.mollview(maplist.mask_huge, cmap=c, cbar=False, title=r'Large Reconstruction Mask $\left(f_{\mathrm{sky}}=%.2f\right)$' % maplist.fsky_huge)
plt.savefig(outdir+'hugemask_unwise')

# 1pt statistics for the fiducial reconstruction
pdiff_expect = pixel_scaling_masked(expect_normprod_COMMANDER, maplist.fsky)
pdiff_measure = pixel_scaling_masked(n_out_normprod_COMMANDER,maplist.fsky)
percent_diff = 200*np.abs(pdiff_measure - pdiff_expect) / (pdiff_measure + pdiff_expect)
plt.figure()
plt.fill_between(centres(bins_out_normprod_COMMANDER), np.zeros(n_out_normprod_COMMANDER.size), pixel_scaling_masked(n_out_normprod_COMMANDER,maplist.fsky)/1e3, label='Velocity reconstruction')
plt.plot(centres(bins_out_normprod_COMMANDER), pixel_scaling_masked(expect_normprod_COMMANDER, maplist.fsky)/1e3,color='k', ls='--',label='Normal product distribution')
ax_inset = plt.gca().axes.inset_axes([0.1, 0.62, 0.3, 0.25])
ax_inset.plot(centres(bins_out_normprod_COMMANDER), percent_diff, label='Velocity reconstruction',lw=0.1)
ax_inset.set_xlim([-.075,.075])
ax_inset.set_ylim([0,40])
ax_inset.set_xticks([0.,-.06,.06],['0','-0.06','0.06'])
ax_inset.set_title('% difference',fontsize=10)
plt.ylim([0, 5.2]) 
plt.xlim([-.075,.075])
plt.xlabel(r'$\frac{v}{c}$', fontsize=16)
plt.ylabel(r'$N_{\mathrm{pix}}\ \left[\times 10^3\right]$')
plt.title('Planck x unWISE pixel value distribution')
plt.legend()
plt.tight_layout()
plt.savefig(outdir+'recon_1pt')

# 1pt statistics across multiple frequencies and masks
plt.figure()
l1, = plt.plot(centres(bins_freqhist), n_out_100/simps(n_out_100), label='100 GHz', zorder=95)
l2, = plt.plot(centres(bins_freqhist), n_out_143/simps(n_out_143), label='143 GHz', zorder=96)
l3, = plt.plot(centres(bins_freqhist), n_out_217/simps(n_out_217), label='217 GHz (fiducial mask)', zorder=97)
l4, = plt.plot(centres(bins_freqhist), n_out_217_huge/simps(n_out_217_huge), label='217 GHz (large mask)', zorder=98)
l5, = plt.plot(centres(bins_freqhist), n_out_353_huge/simps(n_out_353_huge), label='353 GHz (large mask)', zorder=99)
plt.fill_between(centres(bins_freqhist), np.min(n_out_dndz['100GHz'],axis=0)/simps(n_out_100), np.max(n_out_dndz['100GHz'],axis=0)/simps(n_out_100), alpha=0.5, color=l1.get_c(), lw=0.)
plt.fill_between(centres(bins_freqhist), np.min(n_out_dndz['143GHz'],axis=0)/simps(n_out_143), np.max(n_out_dndz['143GHz'],axis=0)/simps(n_out_143), alpha=0.5, color=l2.get_c(), lw=0.)
plt.fill_between(centres(bins_freqhist), np.min(n_out_dndz['217GHz'],axis=0)/simps(n_out_217), np.max(n_out_dndz['217GHz'],axis=0)/simps(n_out_217), alpha=0.5, color=l3.get_c(), lw=0.)
plt.fill_between(centres(bins_freqhist), np.min(n_out_dndz['217GHz_CIBmask'],axis=0)/simps(n_out_217_huge), np.max(n_out_dndz['217GHz_CIBmask'],axis=0)/simps(n_out_217_huge), alpha=0.5, color=l4.get_c(), lw=0.)
plt.fill_between(centres(bins_freqhist), np.min(n_out_dndz['353GHz_CIBmask'],axis=0)/simps(n_out_353_huge), np.max(n_out_dndz['353GHz_CIBmask'],axis=0)/simps(n_out_353_huge), alpha=0.5, color=l5.get_c(), lw=0.)
plt.ylim([0.00001,plt.ylim()[1]*0.85])
plt.xlabel(r'$\frac{v}{c}$', fontsize=16)
plt.ylabel(r'Normalized $\mathrm{N}_{\mathrm{pix}}$')
plt.title('Frequency map reconstruction pixel values')
plt.xlim([-3000/299792.458,1000/299792.458])
plt.legend()
plt.tight_layout()
plt.savefig(outdir+'1ptdrift')

# 1pt statistics for 353 GHz for different foregrounds
plt.figure()
plt.plot(centres(bins_353hist), n_out_353_353hist/simps(n_out_353_353hist), label='353 GHz x unWISE', zorder=97)
plt.plot(centres(bins_353hist), n_out_353_noSMICA/simps(n_out_353_noSMICA), label='353 GHz - SMICA x unWISE', zorder=95)
plt.plot(centres(bins_353hist), n_out_353_noCOMMANDER/simps(n_out_353_noCOMMANDER), label='353 GHz - COMMANDER x unWISE', zorder=96)
plt.plot(centres(bins_353hist), n_out_353_thermaldust/simps(n_out_353_thermaldust), label='Thermal dust x unWISE', zorder=99)
plt.plot(centres(bins_353hist), n_out_353_CIB/simps(n_out_353_CIB), label='CIB x unWISE', zorder=98)
plt.ylim([0.00001, plt.ylim()[1]*.98])
plt.xlabel(r'$\frac{v}{c}$', fontsize=16)
plt.ylabel(r'Normalized $\mathrm{N}_{\mathrm{pix}}$')
plt.title('353 GHz large-masked reconstruction pixel values   ')
plt.xlim([bins_353hist.min(),bins_353hist.max()])
plt.legend()
plt.tight_layout()
plt.savefig(outdir+'hist_353')

# P_me plot
plt.figure()
plt.semilogy(ls,Pme_at_ells[zbar_index,:],lw=2,c='k', ls='--', label=r'$z = \bar{z}$', zorder=100)
for s, sliced in enumerate(slice_indices):
	plt.semilogy(ls,Pme_at_ells[sliced,:], label=r'$z = %.1f$' % results.redshift_at_comoving_radial_distance(chis[sliced]))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$P_{me}\left(\frac{\ell+\frac{1}{2}}{\chi},\chi\right)$')
plt.title('Matter-electron power spectrum')
plt.xlim([2,4000])
plt.legend()
plt.savefig(outdir+'Pme')

# Dipole
plt.figure()
hp.mollview(np.where(np.arange(plotmask.size) < 20000, 0, 1),norm='hist',coord='G',notext=True,cbar=None,cmap=cmap_dipole,title='Planck x unWISE reconstruction dipole')  # Base map
hp.projplot(thetas['SMICA'],phis['SMICA'],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),181.15-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
plt.savefig(outdir+'dipole_Planck_circle')

plt.figure()
hp.mollview(np.nan_to_num(plotmask),norm='hist',coord='G',notext=True,cbar=None,cmap=cmap_dipole,title='Planck x unWISE reconstruction dipole')  # Base map
hp.projplot(hp.vec2ang(mean_dipole['COMMANDER']),'bx',ms=12,mew=4,zorder=100)  # Mean dipole point
hp.projplot(hp.vec2ang(mean_dipole['SMICA']),'gx',ms=12,mew=4,zorder=100)  # Mean dipole point
hp.projplot(hp.vec2ang(mean_dipole['100GHz']),'mx',ms=12,mew=4,zorder=100)  # Mean dipole point
hp.projplot(263.99-360, 48.26, 'ro', lonlat=True,ms=12)  # Primary CMB dipole
hp.projplot(263.99-360+180, -48.26, 'ro', lonlat=True,ms=12)  # Primary CMB dipole antipode
hp.projtext(263.99-360, 48.26, r' $+\beta$', lonlat=True, color='r',fontsize=21,va='top',weight='extra bold')  # Primary CMB dipole text label
hp.projtext(263.99-360+180, -48.26, r' $-\beta$', lonlat=True, color='r',fontsize=21,va='bottom',weight='extra bold')  # Primary CMB dipole antipode text label
hp.projplot(np.repeat(np.pi/2,100),np.linspace(0,2*np.pi,100),c='#DFDF00',coord='E',lw=5)  # Ecliptic line
hp.projplot(thetas['COMMANDER'],phis['COMMANDER'],'-',lw=5,color='b',lonlat=True,rot=(np.degrees(lon_mean['COMMANDER']),180-np.degrees(lat_mean['COMMANDER']),0))   # rotate: south pole counts as 0 latitude for rot keyword
#hp.projplot(thetas['SMICA'][:44],phis['SMICA'][:44],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),180-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
#hp.projplot(thetas['SMICA'][54:],phis['SMICA'][54:],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),180-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
hp.projplot(thetas['SMICA'],phis['SMICA'],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),181.15-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
hp.projplot(thetas['100GHz'],phis['100GHz'],'-',lw=5,color='m',lonlat=True,rot=(np.degrees(lon_mean['100GHz']),180-np.degrees(lat_mean['100GHz']),0))   # rotate: south pole counts as 0 latitude for rot keyword
for i in np.arange(1000):
	print(i)
	_ = hp.projplot(hp.vec2ang(dipoles['COMMANDER'][i,:])[0][0],hp.vec2ang(dipoles['COMMANDER'][i,:])[1][0], 'bx',ms=8,mew=.5,zorder=101,alpha=0.5)
	_ = hp.projplot(hp.vec2ang(dipoles['SMICA'][i,:])[0][0],hp.vec2ang(dipoles['SMICA'][i,:])[1][0], 'gx',ms=8,mew=.5,zorder=101,alpha=0.5)
	_ = hp.projplot(hp.vec2ang(dipoles['100GHz'][i,:])[0][0],hp.vec2ang(dipoles['100GHz'][i,:])[1][0], 'mx',ms=8,mew=.5,zorder=101,alpha=0.5)

plt.savefig(outdir+'dipole_Planck_nocircle')










plt.figure()
hp.mollview(np.nan_to_num(plotmask),norm='hist',coord='G',notext=True,cbar=None,cmap=cmap_dipole,title='Planck x unWISE reconstruction dipole')  # Base map
hp.projplot(hp.vec2ang(mean_dipole['COMMANDER']),'bx',ms=12,mew=4,zorder=100)  # Mean dipole point
hp.projplot(hp.vec2ang(mean_dipole['SMICA']),'gx',ms=12,mew=4,zorder=100)  # Mean dipole point
hp.projplot(hp.vec2ang(mean_dipole['100GHz']),'mx',ms=12,mew=4,zorder=100)  # Mean dipole point
hp.projplot(263.99-360, 48.26, 'ro', lonlat=True,ms=12)  # Primary CMB dipole
hp.projplot(263.99-360+180, -48.26, 'ro', lonlat=True,ms=12)  # Primary CMB dipole antipode
hp.projtext(263.99-360, 48.26, r' $+\beta$', lonlat=True, color='r',fontsize=21,va='top',weight='extra bold')  # Primary CMB dipole text label
hp.projtext(263.99-360+180, -48.26, r' $-\beta$', lonlat=True, color='r',fontsize=21,va='bottom',weight='extra bold')  # Primary CMB dipole antipode text label
hp.projplot(np.repeat(np.pi/2,100),np.linspace(0,2*np.pi,100),c='#DFDF00',coord='E',lw=5)  # Ecliptic line
hp.projplot(thetas['COMMANDER'],phis['COMMANDER'],'-',lw=5,color='b',lonlat=True,rot=(np.degrees(lon_mean['COMMANDER']),180-np.degrees(lat_mean['COMMANDER']),0))   # rotate: south pole counts as 0 latitude for rot keyword
#hp.projplot(thetas['SMICA'][:44],phis['SMICA'][:44],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),180-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
#hp.projplot(thetas['SMICA'][54:],phis['SMICA'][54:],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),180-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
hp.projplot(thetas['SMICA'],phis['SMICA'],'-',lw=5,color='g',lonlat=True,rot=(np.degrees(lon_mean['SMICA']),181.15-np.degrees(lat_mean['SMICA']),0))   # rotate: south pole counts as 0 latitude for rot keyword
hp.projplot(thetas['100GHz'],phis['100GHz'],'-',lw=5,color='m',lonlat=True,rot=(np.degrees(lon_mean['100GHz']),180-np.degrees(lat_mean['100GHz']),0))   # rotate: south pole counts as 0 latitude for rot keyword
hp.projplot(thetas['143GHz'],phis['143GHz'],'-',lw=5,color='c',lonlat=True,rot=(np.degrees(lon_mean['143GHz']),180-np.degrees(lat_mean['143GHz']),0))   # rotate: south pole counts as 0 latitude for rot keyword
hp.projplot(thetas['217GHz'],phis['217GHz'],'-',lw=5,color='orange',lonlat=True,rot=(np.degrees(lon_mean['217GHz']),180-np.degrees(lat_mean['217GHz']),0))   # rotate: south pole counts as 0 latitude for rot keyword
for i in np.arange(1000):
	print(i)
	_ = hp.projplot(hp.vec2ang(dipoles['100GHz'][i,:])[0][0],hp.vec2ang(dipoles['100GHz'][i,:])[1][0], 'mx',ms=8,mew=.5,zorder=101,alpha=0.5)
	_ = hp.projplot(hp.vec2ang(dipoles['143GHz'][i,:])[0][0],hp.vec2ang(dipoles['143GHz'][i,:])[1][0], 'cx',ms=8,mew=.5,zorder=101,alpha=0.5)
	_ = hp.projplot(hp.vec2ang(dipoles['217GHz'][i,:])[0][0],hp.vec2ang(dipoles['217GHz'][i,:])[1][0], 'x',color='orange',ms=8,mew=.5,zorder=101,alpha=0.5)

plt.savefig(outdir+'dipole_Planck_all')


print('\n\nCompleted successfully!\n\n')



# means = {key : np.zeros(40) for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']}
# for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']:
# 	plt.figure()
# 	dimap = reconstructions[key] * plotmask
# 	hp.mollview(dimap)
# 	for i, angle in enumerate(np.linspace(0, np.pi/2, 40)):
# 		arf_above = dimap.copy()
# 		arf_below = dimap.copy()
# 		arf_above[np.where(thetas > np.pi - (np.pi/2 + angle*np.sin(phis)))] = np.nan
# 		arf_below[np.where(thetas < np.pi - (np.pi/2 + angle*np.sin(phis)))] = np.nan
# 		means[key][i] = np.nanmean(arf_above) - np.nanmean(arf_below)
# 		hp.projplot(np.pi - (np.pi/2 + angle*np.sin(np.linspace(0,2*np.pi,100))), np.linspace(0,2*np.pi,100))
# 	plt.savefig(outdir+'test_%s'%key)


# plt.figure()
# for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']:
# 	plt.plot(np.linspace(0, np.pi/2, 40), means[key], label=key)

# plt.title('Difference of unmasked mean hemisphere pixel values')
# plt.xlabel(r'Rotation angle $\alpha$')
# plt.ylabel('Difference of hemispherical means')
# plt.legend()
# plt.savefig(outdir+'map_dipole_means')





# key = 'COMMANDER'
# plt.figure()
# dimap = reconstructions[key] * plotmask
# hp.mollview(dimap)
# for i, angle in enumerate([0,np.pi/4,np.pi/2]):
# 	hp.projplot(np.pi - (np.pi/2 + angle*np.sin(np.linspace(0,2*np.pi,100))), np.linspace(0,2*np.pi,100))

# plt.savefig(outdir+'test_small')



# # subtract the mean of unmasked COMMANDER pixels from the masked COMMANDER reconstruction
# # do inpainting with mean-subtracted statistics
# # measure the dipole
# mean_subtracted_dipoles = { key : 0 for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']}
# recons_inpaint_store = { key : 0 for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']}
# for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']:
# 	ClTT_filter = maplist.Cls[key].copy()[:maplist.lmax+1]
# 	ClTT_filter[:100] = 1e15
# 	Tlms   = hp.synalm(maplist.Cls[key], lmax=maplist.lmax)
# 	lsslms = hp.synalm(maplist.Cls['unWISE'], lmax=maplist.lmax)
# 	Tmap_filtered   = hp.alm2map(hp.almxfl(Tlms,   np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0)), lmax=maplist.lmax, nside=maplist.nside)
# 	lssmap_filtered = hp.alm2map(hp.almxfl(lsslms, np.divide(cltaug_fiducial,           Clgg_filter, out=np.zeros_like(Clgg_filter),               where=Clgg_filter!=0)), lmax=maplist.lmax, nside=maplist.nside)
# 	outmap_noiserecon = -Tmap_filtered * lssmap_filtered * inverse_fiducial_mask * noises[key]
# 	recon_inpainted = reconstructions[key]  + outmap_noiserecon - np.nanmean(reconstructions[key]*plotmask)
# 	recons_inpaint_store[key] = recon_inpainted.copy()
# 	mean_subtracted_dipoles[key] = hp.vec2ang(hp.fit_dipole(recon_inpainted)[1])




# plt.figure()
# hp.mollview(np.nan_to_num(plotmask),norm='hist',coord='G',notext=True,cbar=None,cmap=cmap_dipole,title='Planck x unWISE reconstruction dipole')  # Base map
# for i, key in enumerate(['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']):
# 	hp.projplot(hp.vec2ang(mean_dipole[key]),'x',color=linecolors[i],ms=12,mew=4,zorder=100, label=key)  # Mean dipole point
# 	hp.projplot(mean_subtracted_dipoles[key], 'o', color=darken(linecolors[i],1.2),ms=12, mew=4, zorder=101)  # mean subtracted dipole

# hp.projplot(np.repeat(np.pi/2,100),np.linspace(0,2*np.pi,100),c='#DFDF00',coord='E',lw=5)  # Ecliptic line
# plt.legend(loc='lower right')
# plt.savefig(outdir+'dipole_meansubtracted')








# means = {key : np.zeros(10) for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']}
# for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']:
# 	dimap = recons_inpaint_store[key] * plotmask
# 	for i, angle in enumerate(np.linspace(0, np.pi/2, 10)):
# 		arf_above = dimap.copy()
# 		arf_below = dimap.copy()
# 		arf_above[np.where(thetas > np.pi - (np.pi/2 + angle*np.sin(phis)))] = np.nan
# 		arf_below[np.where(thetas < np.pi - (np.pi/2 + angle*np.sin(phis)))] = np.nan
# 		means[key][i] = np.nanmean(arf_above) - np.nanmean(arf_below)


# plt.figure()
# for key in ['COMMANDER', 'SMICA', '100GHz', '143GHz', '217GHz']:
# 	plt.plot(np.linspace(0, np.pi/2, 10), means[key], label=key)

# plt.title('Difference of unmasked mean hemisphere pixel values')
# plt.xlabel(r'Rotation angle $\alpha$')
# plt.ylabel('Difference of hemispherical means')
# plt.legend()
# plt.savefig(outdir+'map_dipole_means_meansubtracted')

