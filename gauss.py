import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})
import numpy as np
import camb
from camb import model, initialpower
import scipy

maskfile = 'data/mask_unWISE_thres_v10.npy'
Tmapfile = 'data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits'
lssmapfile = 'data/unWISE/numcounts_map1_2048-r1-v2_flag.fits'
outdir = 'plots/analysis/sims_mask_unwise/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

### Cosmology and velocity calculations from jupyter notebook
redshifts = np.linspace(0.0,2.5,100)
spectra_lmax = 4000
ls = np.unique(np.append(np.geomspace(1,spectra_lmax-1,200).astype(int), spectra_lmax-1))
zbar_index = 30
zbar = redshifts[zbar_index]

# Set up the growth function by computing ratios of linear matter power spectra 
# and compute the radial comoving distance functions
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(redshifts=redshifts, kmax=2.0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 50)

Dratio = pk[:,1]/pk[zbar_index,1]

chis = results.comoving_radial_distance(redshifts)

# Set up the electron window function from the kSZ term:
const = 2.3e-5
ombh2=.0224
tbar=.054
h = results.h_of_z(redshifts)
W_e = const*ombh2*(1+redshifts)**2/h/tbar

# Set up the galaxy window function:
b_g = 0.8 + 1.2*redshifts

dndz_data = np.transpose(np.loadtxt('data/unWISE/blue.txt', dtype=float))
dndz = np.interp(redshifts,dndz_data[0,:],dndz_data[1,:])
W_g = dndz/np.trapz(dndz,redshifts)
W_g[0] = 0.0

# Assemble the velocity window function:
W_v = W_e * W_g * (b_g/b_g[zbar_index]) * Dratio * (chis[zbar_index]/chis)**(1.5)
W_v[0] = 0.0 


#Trying to get velocities out. From here https://camb.readthedocs.io/en/stable/transfer_variables.html the variable 
#'v_newtonian_cdm' is actually k*v/curlyH where curlyH is comoving Hubble I presume.
nks = 2000
ks = np.logspace(-4,1,nks)

### Spectra calculations and estimator
from paper_analysis import Estimator, Cosmology
from astropy.io import fits
import healpy as hp
from scipy.interpolate import interp1d

mask_map = np.load(maskfile)
#mask_map = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL020'],n2r=True)
fsky = np.where(mask_map!=0)[0].size / mask_map.size

SMICAinp = hp.reorder(fits.open(Tmapfile)[1].data['I_STOKES'], n2r=True) / 2.725  # Remove K_CMB units
SMICAbeam = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143)
SMICAmap_real = hp.alm2map(hp.almxfl(hp.map2alm(SMICAinp), 1/SMICAbeam), 2048)  # De-beamed unitless SMICA map
ClTT = hp.anafast(SMICAmap_real)
ClTT[ls.max()+1:] = 0.  # Zero Cls above our lmax

unWISEmap = fits.open(lssmapfile)[1].data['T'].flatten()
ngbar = unWISEmap.sum() / unWISEmap.size  # Needed to scale delta to galaxy

csm = Cosmology(nbin=1,zmin=redshifts.min(), zmax=redshifts.max(), redshifts=redshifts, ks=ks)  # Set up cosmology
estim = Estimator(nbin=1)  # Set up estimator

# Ensure csm object has same cosmology as what we used in our velocity calculation earlier
csm.cambpars = pars
csm.cosmology_data = camb.get_background(pars)
csm.bin_width = chis[-1] - chis[0]
csm.chi_bin_boundaries = np.array([chis[0],chis[-1]])

csm.compute_Cls(ngbar=ngbar)  # These are the integrated Cls over the entire bin


fullspectrum_ls = np.unique(np.append(np.geomspace(1,6144-1,200).astype(int), 6144-1))
# Now we compute the same Cls we just did, but at zbar instead of over the entire bin
Pmms = np.zeros((chis.size,fullspectrum_ls.size))
Pmm_full = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=100., zmax=redshifts.max())
for l, ell in enumerate(fullspectrum_ls):  # Do limber approximation: P(z,k) -> P(z, (ell+0.5)/chi )
	Pmms[:,l] = np.diagonal(np.flip(Pmm_full.P(camb.get_background(pars).redshift_at_comoving_radial_distance(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))

Pem_bin1_chi = np.zeros((fullspectrum_ls.size))
for l, ell in enumerate(fullspectrum_ls):
	Pem_bin1_chi[l] = Pmms[zbar_index,l] * np.diagonal(np.flip(csm.bias_e2(csm.chi_to_z(chis), (ell+0.5)/chis[::-1]), axis=1))[zbar_index]  # Convert Pmm to Pem

galaxy_window_binned = csm.get_limber_window('g', chis, avg=False)[zbar_index]  # Units of 1/Mpc
taud1_window  = csm.get_limber_window('taud', chis, avg=False)[zbar_index]  # Units of 1/Mpc
chibar = csm.z_to_chi(redshifts[zbar_index])

# Manual Cls at zbar. Same form as in csm.compute_Cls but instead of integrating over chi we multiply the integrand by deltachi, where the integrand is evaluated at zbar
Cltaug_at_zbar = interp1d(fullspectrum_ls, (Pem_bin1_chi * galaxy_window_binned * taud1_window   / chibar**2) * csm.bin_width * ngbar, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

# The two instances where Estimator functions are called: one for the noise, one for the reconstruction
noise_SMICA_unWISE = estim.Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=1, cltt=ClTT.copy(), clgg_binned=csm.Clgg[0,0,:].copy(), cltaudg_binned=Cltaug_at_zbar.copy())

#foregrounds_masking = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL060'],n2r=True)
#fsky_masking = np.where(foregrounds_masking!=0)[0].size/foregrounds_masking.size
# Generate realizations
#SMICA_Cls = hp.anafast(SMICAinp*foregrounds_masking)/fsky_masking
#LSS_Cls = hp.anafast(unWISEmap)
#print(kassksnksan)
#for i in np.arange(20):
#	print('Generating realization %d' % (i+1))
	#if not os.path.exists(outdir+'SMICA_gauss_%0d.npy' % i):
	#	newmap = hp.synfast(SMICA_Cls, 2048)
	#	newmap_real = hp.alm2map(hp.almxfl(hp.map2alm(newmap), 1/SMICAbeam), 2048)
	#	np.save('SMICA_gauss_%0d.npy' % i, newmap_real)
	#if not os.path.exists(outdir+'unWISE_gauss_%0d.npy' % i):
	#	newlssmap = hp.synfast(LSS_Cls, 2048)
	#	np.save(outdir+'unWISE_gauss_%0d.npy' % i, newlssmap)
	

Tmaps = np.load('data/gauss_reals/SMICA_gauss_100_reals.npy')[:100]
for i in np.arange(100):
	print('Reconstructing cases for realization %d     fsky=%.2f'% ((i+1), fsky))
	#Tmap_gauss = np.load('SMICA_gauss_%0d.npy' % i)
	Tmap_gauss = Tmaps[i]
	#lssmap_gauss = np.load(outdir+'unWISE_gauss_%0d.npy' % i)
	#_, _, outmap_Treal_greal   = estim.combine(SMICAmap_real, unWISEmap,    mask_map, ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA_unWISE,6144), convert_K=False)
	_, _, outmap_Tgauss_greal  = estim.combine(Tmap_gauss,    unWISEmap,    mask_map, ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA_unWISE,6144), convert_K=False)
	#_, _, outmap_Treal_ggauss  = estim.combine(SMICAmap_real, lssmap_gauss, mask_map, ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA_unWISE,6144), convert_K=False)
	#_, _, outmap_Tgauss_ggauss = estim.combine(Tmap_gauss,    lssmap_gauss, mask_map, ClTT, csm.Clgg[0,0,:], Cltaug_at_zbar.copy(), np.repeat(noise_SMICA_unWISE,6144), convert_K=False)
	#recon_Cl_Treal_greal   = hp.anafast(outmap_Treal_greal)
	recon_Cl_Tgauss_greal  = hp.anafast(outmap_Tgauss_greal)
	#recon_Cl_Treal_ggauss  = hp.anafast(outmap_Treal_ggauss)
	#recon_Cl_Tgauss_ggauss = hp.anafast(outmap_Tgauss_ggauss)
	#np.savez('reconstruction_%0d.npz'%i,Cl_Treal_greal=recon_Cl_Treal_greal,Cl_Tgauss_greal=recon_Cl_Tgauss_greal,Cl_Treal_ggauss=recon_Cl_Treal_ggauss,Cl_Tgauss_ggauss=recon_Cl_Tgauss_ggauss)
	np.savez('rreconstruction_%0d.npz'%i,Cl_Tgauss_greal=recon_Cl_Tgauss_greal)


#avg_Cl_Treal_greal   = np.zeros((100, 6144))
avg_Cl_Tgauss_greal  = np.zeros((100, 6144))
#avg_Cl_Treal_ggauss  = np.zeros((100, 6144))
#avg_Cl_Tgauss_ggauss = np.zeros((100, 6144))
for i in np.arange(100):
	data = np.load('rreconstruction_%0d.npz'%i)
	#avg_Cl_Treal_greal[i,:]   = data['Cl_Treal_greal']
	avg_Cl_Tgauss_greal[i,:]  = data['Cl_Tgauss_greal']
	#avg_Cl_Treal_ggauss[i,:]  = data['Cl_Treal_ggauss']
	#avg_Cl_Tgauss_ggauss[i,:] = data['Cl_Tgauss_ggauss']

#np.savez(outdir+'reconstructions',Cl_Treal_greal=avg_Cl_Treal_greal,Cl_Tgauss_greal=avg_Cl_Tgauss_greal,Cl_Treal_ggauss=avg_Cl_Treal_ggauss,Cl_Tgauss_ggauss=avg_Cl_Tgauss_ggauss)
np.savez('rreconstructions',Cl_Tgauss_greal=avg_Cl_Tgauss_greal)


print(kaskalamnikat)



Tmaps = np.zeros((100,12*2048**2))
lssmaps = np.zeros((100,12*2048**2))
for i in np.arange(100):
	Tmaps[i] = np.load('SMICA_gauss_%0d.npy' % i)
	lssmaps[i] = np.load(outdir+'unWISE_gauss_%0d.npy' % i)
	


np.save('SMICA_gauss_100_reals',Tmaps)
np.save('unWISE_gauss_100_reals', lssmaps)
