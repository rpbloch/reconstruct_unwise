#### Get frequency maps to match theory noise which we take as correct
#### Scale window velocity, your cltaug is just at zbar so it's times that equation thing and W(v) ^2. W(v) is not normalized to 1, normalization is fixed to choice of zbar

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
    outmap = outmap_filtered * Noise * mask
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

taud_window  = (-thomson_SI * ne_0 * (1+results.redshift_at_comoving_radial_distance(chis))**2 * m_per_Mpc)[zbar_index]  # Units of 1/Mpc
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
velocity_compute_ells = np.unique(np.geomspace(1,30,10).astype(int))
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

print('    Computing windowed velocity')
clv_windowed, clv_windowed_mm = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['SMICA'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed_100GHz, clv_windowed_mm_100GHz = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['100GHz'] / pars.TCMB**2, clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed_143GHz, clv_windowed_mm_143GHz = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['143GHz'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed_217GHz, clv_windowed_mm_217GHz = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['217GHz'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)
clv_windowed_353GHz, clv_windowed_mm_353GHz = compute_windowed_velocity(results, Pmm_full, galaxy_window, clv, cltt=maplist.Cls['353GHz'], clgg=maplist.Cls['unWISE'], cltaug=cltaug_fiducial, sample_ells=velocity_compute_ells, chis=chis, chibar=chibar, zbar_index=zbar_index, bin_width=bin_width)

print('    Computing fiducial reconstructions')
noises['SMICA'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['SMICA'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
noises['COMMANDER'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['COMMANDER'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
reconstructions['SMICA'] = combine_alm(maplist.processed_alms['SMICA'], maplist.processed_alms['unWISE'], maplist.mask, maplist.Cls['SMICA'], maplist.Cls['unWISE'], cltaug_fiducial, noises['SMICA'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
reconstructions['COMMANDER'] = combine_alm(maplist.processed_alms['COMMANDER'], maplist.processed_alms['unWISE'], maplist.mask, maplist.Cls['COMMANDER'], maplist.Cls['unWISE'], cltaug_fiducial, noises['COMMANDER'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
recon_Cls['SMICA'] = maplist.alm2cl(hp.map2alm(reconstructions['SMICA'], lmax=recon_lmax), maplist.fsky)
recon_Cls['COMMANDER'] = maplist.alm2cl(hp.map2alm(reconstructions['COMMANDER'], lmax=recon_lmax), maplist.fsky)

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
	if '353GHz' not in key:
		convert_K_flag = True if '100GHz' in key else False
		reconstructions[key] = combine_alm(maplist.processed_alms[key], maplist.processed_alms['unWISE'], maplist.mask, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial, noises[key], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=convert_K_flag)
		recon_Cls[key] = maplist.alm2cl(hp.map2alm(reconstructions[key], lmax=recon_lmax), maplist.fsky)
	if ('217GHz' in key) or ('353GHz' in key):
		reconstructions[key+'_CIBmask'] = combine_alm(maplist.processed_alms[key+'_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial, noises[key], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
		recon_Cls[key+'_CIBmask'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_CIBmask'], lmax=recon_lmax), maplist.fsky_huge)


maplist.stored_maps['recon_SMICA'] = maplist.lowpass_filter(reconstructions['SMICA'], lmax=25)
maplist.stored_maps['recon_COMMANDER'] = maplist.lowpass_filter(reconstructions['COMMANDER'], lmax=25)

### Plots
bandpowers = lambda spectrum : np.array([spectrum[1+(5*i):1+(5*(i+1))].mean() for i in np.arange(spectrum.size//5)])
x_ells = bandpowers(np.arange(recon_lmax+1))
linecolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

## Needs to be cut in outside figure, hp.mollview is uncooperative with matplotlib formatting for whitespace.
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,15))
plt.axes(ax1)
hp.mollview(maplist.stored_maps['recon_SMICA']*maplist.mask, title=r'SMICA x unWISE Reconstruction $\left(\ell_{\mathrm{max}}\leq 25\right)$', unit=r'$\frac{v}{c}$', hold=True)
plt.axes(ax2)
hp.mollview(maplist.stored_maps['recon_COMMANDER']*maplist.mask, title=r'COMMANDER x unWISE Reconstruction $\left(\ell_{\mathrm{max}}\leq 25\right)$',unit=r'$\frac{v}{c}$', hold=True)
plt.tight_layout()
plt.savefig(outdir+'recon_outputs')

plt.figure()
plt.semilogy(x_ells, bandpowers(recon_Cls['SMICA']), label='SMICA x unWISE Reconstruction', ls='None', marker='x', zorder=100,color=linecolors[0])
plt.semilogy(x_ells, bandpowers(recon_Cls['COMMANDER']),label='COMMANDER x unWISE Reconstruction', ls='None', marker='x', zorder=100,color=linecolors[3])
plt.semilogy(x_ells, np.repeat(noises['SMICA'],x_ells.size), c='k',label='Theory Noise', ls='--', zorder=10, lw=2)
plt.xlim([2, 25])
#plt.semilogy(velocity_compute_ells, clv_windowed+noises['SMICA'], color=linecolors[1],lw=2,label='Windowed velocity + noise')
plt.semilogy(bandpowers(np.arange(50)), bandpowers(interp1d(velocity_compute_ells,clv_windowed+noises['SMICA'],bounds_error=False,fill_value=0.)(np.arange(50))), color=linecolors[1],lw=2,label='Windowed velocity + noise')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)
plt.title('Planck x unWISE Reconstruction')
plt.tight_layout()
plt.savefig(outdir+'signal_noise_gauss.png')


fig, ((ax100, ax143), (ax217, ax217huge), (ax353huge, axnull)) = plt.subplots(3,2,figsize=(12,18))
axnull.remove()
for freq, ax in zip([100, 143, 217], [ax100, ax143, ax217]):
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz' % freq]), label='%d GHz x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[0])
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz_noSMICA' % freq]), label='%d GHz - SMICA x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[2])
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz_noCOMMANDER' % freq]), label='%d GHz - COMMANDER x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[3])
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz_thermaldust' % freq]), label='thermal dust @ %d GHz x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[4])
	ax.set_title('Reconstructions at %d GHz, fsky = %.2f' % (freq, maplist.fsky))

for freq, ax in zip([217, 353], [ax217huge, ax353huge]):
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz_CIBmask' % freq]), label='%d GHz x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[0])
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz_noSMICA_CIBmask' % freq]), label='%d GHz - SMICA x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[2])
	ax.semilogy(x_ells, bandpowers(recon_Cls['%dGHz_noCOMMANDER_CIBmask' % freq]), label='%d GHz - COMMANDER x unWISE' % freq, ls='None', marker='x', zorder=100, color=linecolors[3])
	ax.set_title('Reconstructions at %d GHz, fsky = %.2f' % (freq, maplist.fsky_huge))


ax100.semilogy(velocity_compute_ells, clv_windowed_100GHz+noises['100GHz']/pars.TCMB**2, color=linecolors[1],lw=2,label='Windowed velocity + noise')
ax143.semilogy(velocity_compute_ells, clv_windowed_143GHz+noises['143GHz'], color=linecolors[1],lw=2,label='Windowed velocity + noise')
ax217.semilogy(velocity_compute_ells, clv_windowed_217GHz+noises['217GHz'], color=linecolors[1],lw=2,label='Windowed velocity + noise')
ax217huge.semilogy(velocity_compute_ells, clv_windowed_217GHz+noises['217GHz'], color=linecolors[1],lw=2,label='Windowed velocity + noise')
ax353huge.semilogy(velocity_compute_ells, clv_windowed_353GHz+noises['353GHz'], color=linecolors[1],lw=2,label='Windowed velocity + noise')

ax217huge.semilogy(x_ells, bandpowers(recon_Cls['217GHz_thermaldust_CIBmask']), label='thermal dust @ 217 GHz x unWISE', ls='None', marker='x', zorder=100, color=linecolors[4])
ax217huge.set_title('Reconstructions at 217 GHz, fsky = %.2f' % maplist.fsky_huge)
ax353huge.semilogy(x_ells, bandpowers(recon_Cls['353GHz_CIB_CIBmask']), label='CIB @ 353 GHz x unWISE', ls='None', marker='x', zorder=100, color=linecolors[4])
ax353huge.set_title('Reconstructions at 353 GHz, fsky = %.2f' % maplist.fsky_huge)

for ax in [ax100, ax143, ax217,ax217huge, ax353huge]:
	ax.set_xlim([2, 25])
	ax.legend()
	ax.set_xlabel(r'$\ell$')
	ax.set_ylabel(r'$\frac{v^2}{c^2}$',rotation=0,fontsize=16)

plt.tight_layout()
plt.savefig(outdir+'Foreground_Contributions')


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

plt.figure()
plt.semilogy(-cltaug_fiducial, lw=2, color='red', zorder=200, label=r'$C_\ell^{\tau\mathrm{g}}$ for best-fit $\frac{d\mathrm{N}}{dz}$')
plt.semilogy(-cltaug_dndz[0,:], lw=0.5, color='gray', alpha=0.75, label=r'$C_\ell^{\tau\mathrm{g}}$ for $\frac{d\mathrm{N}}{dz}$ realizations')
for i in np.arange(1,100):
	plt.semilogy(-cltaug_dndz[i,:], lw=0.5, color='gray', alpha=0.75)

plt.title('unWISE blue modelled tau-galaxy cross-spectrum')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\tau\mathrm{g}}$')
plt.xlim([101,4000])
#plt.ylim([plt.ylim()[0], csm.Clgg[0,0,101]*2])
leg = plt.legend()
for lh in leg.legendHandles[1:]:
	lh.set_alpha(1.)
	lh.set_linewidth(1.)

plt.gca().invert_yaxis()
plt.gca().set_yticks([1e-7,1e-8,1e-9,1e-10],labels=[r'$-10^{-7}$',r'$-10^{-8}$',r'$-10^{-9}$',r'$-10^{-10}$'])
plt.tight_layout()
plt.savefig(outdir+'cltaug')


c = mpl.colors.ListedColormap(['darkred', 'gold'])
plt.figure()
hp.mollview(maplist.mask, cmap=c, cbar=False, title=r'Fiducial Reconstruction Mask $\left(f_{\mathrm{sky}}=%.2f\right)$' % maplist.fsky)
plt.savefig(outdir+'Mask input')

plt.figure()
hp.mollview(maplist.mask_huge, cmap=c, cbar=False, title=r'Large Reconstruction Mask $\left(f_{\mathrm{sky}}=%.2f\right)$' % maplist.fsky_huge)
plt.savefig(outdir+'hugemask_unwise')



centres = lambda BIN : (BIN[:-1]+BIN[1:]) / 2
bessel = lambda bins, Tmap, lssmap : kn(0, np.abs(centres(bins)) / (np.std(Tmap)*np.std(lssmap)))
normal_product = lambda bins, Tmap, lssmap : bessel(bins,Tmap,lssmap) / (np.pi * np.std(Tmap) * np.std(lssmap))
pixel_scaling_masked = lambda distribution, FSKY : (12*2048**2) * FSKY * (distribution / simps(distribution))

nbin_hist = 1000000
n_out_normprod_COMMANDER, bins_out_normprod_COMMANDER = np.histogram(reconstructions['COMMANDER'][np.where(maplist.mask!=0)], bins=nbin_hist)
n_out_normprod_SMICA, bins_out_normprod_SMICA = np.histogram(reconstructions['SMICA'][np.where(maplist.mask!=0)], bins=nbin_hist)

normprod_COMMANDERmap_filtered = hp.alm2map(hp.almxfl(maplist.processed_alms['COMMANDER'], np.divide(np.ones_like(maplist.Cls['COMMANDER']), maplist.Cls['COMMANDER'], out=np.zeros_like(maplist.Cls['COMMANDER']), where=np.arange(cltaug_fiducial.size)>=100)), lmax=maplist.lmax, nside=maplist.nside)
normprod_SMICAmap_filtered = hp.alm2map(hp.almxfl(maplist.processed_alms['SMICA'], np.divide(np.ones_like(maplist.Cls['SMICA']), maplist.Cls['SMICA'], out=np.zeros_like(maplist.Cls['SMICA']), where=np.arange(cltaug_fiducial.size)>=100)), lmax=maplist.lmax, nside=maplist.nside)
normprod_lssmap_filtered = hp.alm2map(hp.almxfl(maplist.processed_alms['unWISE'], np.divide(cltaug_fiducial, maplist.Cls['unWISE'], out=np.zeros_like(cltaug_fiducial), where=np.arange(cltaug_fiducial.size)>=100)), lmax=maplist.lmax, nside=maplist.nside)

normprod_std_COMMANDERmap_filtered   = np.std(normprod_COMMANDERmap_filtered[np.where(maplist.mask!=0)])
normprod_std_SMICAmap_filtered   = np.std(normprod_SMICAmap_filtered[np.where(maplist.mask!=0)])
normprod_std_lssmap_filtered = np.std(normprod_lssmap_filtered[np.where(maplist.mask!=0)])

expect_normprod_COMMANDER = normal_product(bins_out_normprod_COMMANDER,normprod_COMMANDERmap_filtered[np.where(maplist.mask!=0)]*noises['COMMANDER'],normprod_lssmap_filtered[np.where(maplist.mask!=0)])
expect_normprod_SMICA = normal_product(bins_out_normprod_SMICA,normprod_SMICAmap_filtered[np.where(maplist.mask!=0)]*noises['SMICA'],normprod_lssmap_filtered[np.where(maplist.mask!=0)])


plt.figure()
plt.plot(centres(bins_out_normprod_COMMANDER), pixel_scaling_masked(n_out_normprod_COMMANDER,maplist.fsky)-pixel_scaling_masked(expect_normprod_COMMANDER, maplist.fsky),label='COMMANDER',lw=.5)
#plt.plot(centres(bins_out_normprod_SMICA), pixel_scaling_masked(n_out_normprod_SMICA,maplist.fsky)-pixel_scaling_masked(expect_normprod_SMICA, maplist.fsky),label='SMICA',lw=.25)
plt.xlim([-.25/1e2,.25/1e2])
plt.legend()
plt.savefig(outdir+'test')

plt.figure()
plt.fill_between(centres(bins_out_normprod_COMMANDER), np.zeros(n_out_normprod_COMMANDER.size), pixel_scaling_masked(n_out_normprod_COMMANDER,maplist.fsky)/1e5, label='Velocity reconstruction')
plt.plot(centres(bins_out_normprod_COMMANDER), pixel_scaling_masked(expect_normprod_COMMANDER, maplist.fsky)/1e5,color='k', ls='--', lw=2., label='Normal product distribution')
ax_inset_COM = plt.gca().axes.inset_axes([0.01, 0.65, 0.3, 0.25])
ax_inset_COM.fill_between(centres(bins_out_normprod_COMMANDER), np.zeros(n_out_normprod_COMMANDER.size), pixel_scaling_masked(n_out_normprod_COMMANDER,maplist.fsky)/1e5, label='Velocity reconstruction')
ax_inset_COM.plot(centres(bins_out_normprod_COMMANDER), pixel_scaling_masked(expect_normprod_COMMANDER, maplist.fsky)/1e5,color='k', ls='--', lw=2., label='Normal product distribution')
ax_inset_COM.set_xlim([-20/3e5,20/3e5])
ax_inset_COM.annotate(r'$\times 10^{-5}$', xy=(21/299792.458, 1.85), xycoords='data',annotation_clip=False)
ax_inset_COM.plot([0,0],[0,4],c='k',alpha=0.5)
ax_inset_COM.plot(np.repeat(centres(bins_out_normprod_COMMANDER)[np.where(n_out_normprod_COMMANDER==n_out_normprod_COMMANDER.max())][0],2),[0,4],c='k',alpha=0.5)
#ax_inset_COM.set_ylim([y1,y2])
ax_inset_COM.set_yticks([])
ax_inset_COM.set_xticks([0., centres(bins_out_normprod_COMMANDER)[np.where(n_out_normprod_COMMANDER==n_out_normprod_COMMANDER.max())][0]],['0', '%.1f' % (centres(bins_out_normprod_COMMANDER)[np.where(n_out_normprod_COMMANDER==n_out_normprod_COMMANDER.max())][0]*1e5)])
ax_inset_COM.set_title('COMMANDER x unWISE',fontsize=10)
ax_inset_COM.set_ylim([2,4])
ax_inset_SMICA = plt.gca().axes.inset_axes([0.01, 0.25, 0.3, 0.25])
ax_inset_SMICA.fill_between(centres(bins_out_normprod_SMICA), np.zeros(n_out_normprod_SMICA.size), pixel_scaling_masked(n_out_normprod_SMICA,maplist.fsky)/1e5, label='Velocity reconstruction')
ax_inset_SMICA.plot(centres(bins_out_normprod_SMICA), pixel_scaling_masked(expect_normprod_SMICA, maplist.fsky)/1e5,color='k', ls='--', lw=2., label='Normal product distribution')
ax_inset_SMICA.set_xlim([-80/3e5,50/3e5])
ax_inset_SMICA.annotate(r'$\times 10^{-5}$', xy=(52/299792.458, 1.85), xycoords='data',annotation_clip=False)
y1, y2 = ax_inset_SMICA.get_ylim()
ax_inset_SMICA.plot([0,0],[0,4],c='k',alpha=0.5)
ax_inset_SMICA.plot(np.repeat(centres(bins_out_normprod_SMICA)[np.where(n_out_normprod_SMICA==n_out_normprod_SMICA.max())][0],2),[0,4],c='k',alpha=0.5)
ax_inset_SMICA.set_ylim([y1,y2])
ax_inset_SMICA.set_yticks([])
ax_inset_SMICA.set_xticks([0., centres(bins_out_normprod_SMICA)[np.where(n_out_normprod_SMICA==n_out_normprod_SMICA.max())][0]],['0', '%.1f' % (centres(bins_out_normprod_SMICA)[np.where(n_out_normprod_SMICA==n_out_normprod_SMICA.max())][0]*1e5)])
ax_inset_SMICA.set_title('SMICA x unWISE',fontsize=10)
ax_inset_SMICA.set_ylim([2,4])
y1, y2 = plt.ylim()
plt.ylim([0, y2]) 
plt.xlim([-.3,.3])
plt.xlabel(r'$\frac{\Delta T}{T}$', fontsize=16)
plt.ylabel(r'$N_{\mathrm{pix}}\ \left[\times 10^5\right]$')
plt.title('Planck x unWISE pixel value distribution')
plt.legend()
plt.tight_layout()
plt.savefig(outdir+'recon_1pt')


maplist.stored_maps['recon_100GHz'] = maplist.lowpass_filter(reconstructions['100GHz'], lmax=25)
maplist.stored_maps['recon_143GHz'] = maplist.lowpass_filter(reconstructions['143GHz'], lmax=25)
maplist.stored_maps['recon_217GHz'] = maplist.lowpass_filter(reconstructions['217GHz'], lmax=25)
maplist.stored_maps['recon_217GHz_CIBmask'] = maplist.lowpass_filter(reconstructions['217GHz_CIBmask'], lmax=25)
maplist.stored_maps['recon_353GHz_CIBmask'] = maplist.lowpass_filter(reconstructions['353GHz_CIBmask'], lmax=25)

histbins = np.linspace(-3000,1000,nbin_hist) / 299792.458

n_out_100, _ = np.histogram(maplist.stored_maps['recon_100GHz'][np.where(maplist.mask!=0)], bins=histbins)
n_out_143, _ = np.histogram(maplist.stored_maps['recon_143GHz'][np.where(maplist.mask!=0)], bins=histbins)
n_out_217, _ = np.histogram(maplist.stored_maps['recon_217GHz'][np.where(maplist.mask!=0)], bins=histbins)
n_out_217_huge, _ = np.histogram(maplist.stored_maps['recon_217GHz_CIBmask'][np.where(maplist.mask_huge!=0)], bins=histbins)
n_out_353_huge, _ = np.histogram(maplist.stored_maps['recon_353GHz_CIBmask'][np.where(maplist.mask_huge!=0)], bins=histbins)


plt.figure()
plt.plot(centres(histbins)*299792.458, n_out_100/simps(n_out_100), label='100 GHz')
plt.plot(centres(histbins)*299792.458, n_out_143/simps(n_out_143), label='143 GHz')
plt.plot(centres(histbins)*299792.458, n_out_217/simps(n_out_217), label='217 GHz (fiducial mask)')
plt.plot(centres(histbins)*299792.458, n_out_217_huge/simps(n_out_217_huge), label='217 GHz (large mask)')
plt.plot(centres(histbins)*299792.458, n_out_353_huge/simps(n_out_353_huge), label='353 GHz (large mask)')
plt.xlabel('km/s')
plt.ylabel(r'Normalized $\mathrm{N}_{\mathrm{pix}}$')
plt.title('Frequency map reconstruction pixel values')
plt.xlim([-2000,1000])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(outdir+'1ptdrift')


Pme_at_ells = np.zeros((chis.size, ls.size))
for l, ell in enumerate(ls):
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))
	m_to_e = np.diagonal(np.flip(bias_e2(results.redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1]), axis=1))
	Pme_at_ells[:,l] = Pmms[:,l] * m_to_e



slice_indices = np.arange(chis.size)[10:][::20]

plt.figure()
plt.semilogy(ls,Pme_at_ells[zbar_index,:],lw=2,c='k', ls='--', label=r'$z = \bar{z}$', zorder=100)
for s, sliced in enumerate(slice_indices):
	plt.semilogy(ls,Pme_at_ells[sliced,:], label=r'$z = %.1f$' % results.redshift_at_comoving_radial_distance(chis[sliced]))

plt.xlabel(r'$\ell$')
plt.ylabel(r'$P_{me}\left(\frac{\ell+\frac{1}{2}}{\chi},\chi\right)$')
plt.title('Matter-electron power spectrum')
plt.legend()
plt.savefig(outdir+'Pme')



print(kaskalamnikat)



# Plot P_me thing to see how different we get from what we expect, i.e. if independence from ell prime is a good assumption?



# Then we want errors for dN/dz and P_mm
# Writing: We also want to mention the unWISE stuff where we considered correlation with Nhits and adjusted for extinction, but nothing significant resulted









## Combine dN/dz and Pme/Pmm errors in quadrature for measurement error bars
recon_143_yerr = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[143]['full'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_143[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[143]['full'],axis=0)[:ells_plot.size]/fsky)])
recon_143_yerr_foregrounds = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[143]['noCMB'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_143_foregrounds[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[143]['noCMB'],axis=0)[:ells_plot.size]/fsky)])
recon_143_yerr_thermaldust = np.array([bandpowers(np.sqrt(np.std(recon_Cls_dndz_freq[143]['thermaldust'],axis=0)[:ells_plot.size]**2+error_mm_me_recon_143_thermaldust[:ells_plot.size]**2)/fsky),bandpowers(np.std(recon_Cls_dndz_freq[143]['thermaldust'],axis=0)[:ells_plot.size]/fsky)])

lowpass_SMICA_plot = hp.alm2map(hp.almxfl(hp.map2alm(outmap_SMICA), [0 if l > 25 else 1 for l in np.arange(6144)]), 2048)

plt.figure()
hp.mollview(lowpass_SMICA_plot*total_mask, title=r'SMICA x unWISE Reconstruction $\left(\ell_{\mathrm{max}}\leq 25\right)$',unit=r'$\frac{v}{c}$')
plt.savefig(outdir+'SMICA_out')


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



histogram_bins = np.linspace(-1500,1000,nbin_hist) / 299792.458
n_100, bins_100 = np.histogram(lowpass_output_100[np.where(total_mask!=0)], bins=histogram_bins)
n_143, bins_143 = np.histogram(lowpass_output_143[np.where(total_mask!=0)], bins=histogram_bins)
n_217, bins_217 = np.histogram(lowpass_output_217[np.where(total_mask!=0)], bins=histogram_bins)
n_100_huge, bins_100_huge = np.histogram(lowpass_output_100_huge[np.where(hugemask_unwise!=0)], bins=histogram_bins)
n_143_huge, bins_143_huge = np.histogram(lowpass_output_143_huge[np.where(hugemask_unwise!=0)], bins=histogram_bins)
n_217_huge, bins_217_huge = np.histogram(lowpass_output_217_huge[np.where(hugemask_unwise!=0)], bins=histogram_bins)


darken = lambda color, amount :  colorsys.hls_to_rgb(colorsys.rgb_to_hls(*mc.to_rgb(color))[0], 1 - amount * (1 - colorsys.rgb_to_hls(*mc.to_rgb(color))[1]), colorsys.rgb_to_hls(*mc.to_rgb(color))[2])

plt.figure()
l1, = plt.plot(centres(histogram_bins)*299792.458, n_100/simps(n_100), label='100 GHz')
l2, = plt.plot(centres(histogram_bins)*299792.458, n_143/simps(n_143), label='143 GHz')
l3, = plt.plot(centres(histogram_bins)*299792.458, n_217/simps(n_217), label='217 GHz (fiducial mask)')
plt.plot(centres(histogram_bins)*299792.458, n_100_huge/simps(n_100_huge), label='100 GHz (large mask)', c=darken(l1.get_c(),1.4))
plt.plot(centres(histogram_bins)*299792.458, n_143_huge/simps(n_143_huge), label='143 GHz (large mask)', c=darken(l2.get_c(),1.5))
plt.plot(centres(histogram_bins)*299792.458, n_217_huge/simps(n_217_huge), label='217 GHz (large mask)', c=darken(l3.get_c(),1.3))
plt.xlabel('km/s')
plt.ylabel(r'Normalized $\mathrm{N}_{\mathrm{pix}}$')
plt.xlim([-1500,1000])
plt.legend()
plt.savefig(outdir+'1ptdrift')





















# Linear Pmm 
print('\n\nCompleted successfully!\n\n')



