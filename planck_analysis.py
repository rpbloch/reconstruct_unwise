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
outdir = 'plots/analysis/planck_unwise_analysis_2/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Setting up maps and Cls')
print('   loading maps')
### Load maps (bin 0)
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
#planck_galcut_40 = hp.reorder(np.load('data/planckgal40.npy'), n2r=True)
#mask_map *= planck_galcut_40
fsky = np.where(mask_map!=0)[0].size/mask_map.size
Tmap_noise100 = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/100ghz/ffp10_noise_100_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
Tmap_noise = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/143ghz/ffp10_noise_143_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
Tmap_noise217 = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/217ghz/ffp10_noise_217_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
Tmap_100_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_143_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_217_inp = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_SMICA = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True) / 2.725
Tmap_COMMANDER = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-commander_2048_R3.00_full.fits')[1].data['I_STOKES_INP'],n2r=True) / 2.725
Tmap_NILC = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-nilc_2048_R3.00_full.fits')[1].data['I_STOKES_INP'],n2r=True) / 2.725
Tmap_SEVEM = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-sevem_2048_R3.01_full.fits')[1].data['I_STOKES_INP'],n2r=True) / 2.725
lssmap_unwise = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
vrmap = c.load(basic_conf_dir, 'vr_full_fine_2048_64_real=0_bin=0', dir_base=estim_dir+'sims')

lssmap_gauss = hp.synfast(hp.anafast(lssmap_unwise), 2048)

Tcl_noise = hp.anafast(Tmap_noise)
Tmap_gauss = hp.synfast(Tcl_noise, 2048)

Tmap_SMICA_100 = Tmap_SMICA + hp.synfast(hp.anafast(Tmap_noise100), 2048)
Tmap_SMICA_217 = Tmap_SMICA + hp.synfast(hp.anafast(Tmap_noise217), 2048)

Tmap_SMICA += Tmap_gauss
Tmap_COMMANDER += Tmap_gauss
Tmap_NILC += Tmap_gauss
Tmap_SEVEM += Tmap_gauss
# how well do our spectra agree precisely, not just by eye! We may be looking at 1% or 10% effects
# cut off beams see if it helps
# make sure we understand our input T spectra and they agree (with what lmao)
### DO OUR HOLES MATTER?? rando CMB x unWISE, mask 70 vs mask_map
# All CMB products have effective beam of 5 arcmin as per https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_maps
# Frequency effective beams assigned as per https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Effective_Beams#Statistics_of_the_effective_beams_computed_using_FEBeCoP
beam_SMICA = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6144)
beam_100   = hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6144)
beam_143   = hp.gauss_beam(fwhm=np.radians(7.27/60), lmax=6144)
beam_217   = hp.gauss_beam(fwhm=np.radians(5.01/60), lmax=6144)

# Numerical trouble for healpix for beams that blow up at large ell. Sufficient to flatten out at high ell above where the 1/TT filter peaks
beam_100[4001:] = beam_100[4000]
beam_143[4001:] = beam_143[4000]
beam_217[4001:] = beam_217[4000]

# # healpix hates beam_100 lmao it rings if it debeams at the map level. Works fine at the cl level. Example:
# from scipy.integrate import simps
# map_debeam = hp.anafast(hp.alm2map(hp.almxfl(hp.map2alm(Tmap_100), 1/beam_100), 2048))
# cl_debeam  = hp.anafast(Tmap_100) / beam_100[:6144]**2
# almxfl_debeam  = hp.alm2cl(        hp.almxfl(hp.map2alm(Tmap_100), 1/beam_100))
# fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
# ax1.loglog(cl_debeam[101:121], label='Cl debeam')
# ax1.loglog(almxfl_debeam[101:121], label='alm2cl debeam', ls='--')
# ax1.loglog(map_debeam[101:121], label='map debeam')
# ax2.loglog(cl_debeam[101:121]/simps(cl_debeam[101:121]), label='Cl debeam')
# ax2.loglog(almxfl_debeam[101:121]/simps(almxfl_debeam[101:121]), label='alm2cl debeam', ls='--')
# ax2.loglog(map_debeam[101:121]/simps(map_debeam[101:121]), label='map debeam')
# ax1.legend()
# ax2.legend()
# ax2.get_yaxis().set_visible(False)
# plt.tight_layout()
# plt.savefig(outdir + 'beam_100_issue')


#### DIAGNOSTICS    ####
## This plot demonstrates that for frequency maps:
## The full sky Cl differs from the masked sky Cl by more than just a factor of fsky, both in magnitude and shape
## Therefore we cannot reconstruct on the full sky and mask the output and treat this as the same as
## masking the inputs and getting a masked output. Although each pixel is independent, the agreement between
## theory Cl and map Cl for temperature can exist for either the masked case or the unmasked case, but 
## the theory Cl of one case does not match that of the other even accounting for fsky.
T100_inp_cl = hp.anafast(Tmap_100_inp)
T100_debeam_cl        = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp         ), 1/beam_100))
T100_masked_debeam_cl = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100))
plt.figure()
plt.loglog(T100_inp_cl, label='Planck 100GHz Sky', c='k')
plt.loglog(T100_inp_cl[-1]/beam_100[:-1]**2, label='Beam', ls=':', c='k')
plt.loglog(T100_debeam_cl, label='Sky > Debeamed')
plt.loglog(T100_masked_debeam_cl / fsky, label='[(Sky*mask) > Debeamed] / fsky')
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{TT}}\ \left[K_{\mathrm{CMB}}^2\right]$')
plt.title(r'Mismatch beyond $f_{\mathrm{sky}}$ for masking Planck'+'\n' + r'frequency maps (100GHz)')
plt.tight_layout()
plt.savefig(outdir+'TT_frequency_mismatch')



## This is probably why we see that as we mask more and more of the output
## we get closer agreement to the theory noise. We may see that effect is due
## to TT agreement. Let's check it out:
gal20 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL020'],n2r=True)
gal40 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL040'],n2r=True)
gal60 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL060'],n2r=True)
gal70 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL070'],n2r=True)
gal80 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL080'],n2r=True)
gal90 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL090'],n2r=True)

fsky20 = np.where(gal20*mask_map!=0)[0].size/gal20.size
fsky40 = np.where(gal40*mask_map!=0)[0].size/gal40.size
fsky60 = np.where(gal60*mask_map!=0)[0].size/gal60.size
fsky70 = np.where(gal70*mask_map!=0)[0].size/gal70.size
fsky80 = np.where(gal80!=0)[0].size/gal80.size
fsky90 = np.where(gal90!=0)[0].size/gal90.size

T100_inp_cl_90cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal90), 1/beam_100))
T100_inp_cl_80cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal80), 1/beam_100))
T100_inp_cl_70cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal70), 1/beam_100))
T100_inp_cl_unwise_70cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal70*mask_map), 1/beam_100))
T100_inp_cl_unwise_60cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal60*mask_map), 1/beam_100))
T100_inp_cl_unwise_40cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal40*mask_map), 1/beam_100))
T100_inp_cl_unwise_20cut = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*gal20*mask_map), 1/beam_100))

plt.figure()
plt.loglog(T100_inp_cl,label='unmasked')
plt.loglog(T100_inp_cl_90cut/fsky90,label='fsky=0.90 galcut')
plt.loglog(T100_inp_cl_80cut/fsky80,label='fsky=0.80 galcut')
plt.loglog(T100_inp_cl_70cut/fsky70,label='fsky=0.70 galcut')
plt.loglog(T100_inp_cl_unwise_70cut/fsky,label='unWISE mask (70 galcut)')
plt.loglog(T100_inp_cl_unwise_60cut/fsky60,label='unWISE mask (60 galcut)')
plt.loglog(T100_inp_cl_unwise_40cut/fsky40,label='unWISE mask (40 galcut)')
plt.loglog(T100_inp_cl_unwise_20cut/fsky20,label='unWISE mask (20 galcut)')
plt.legend(ncol=2,loc='upper left')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{TT}}\ \left[K_{\mathrm{CMB}}^2\right]$')
plt.title(r'Mismatch beyond $f_{\mathrm{sky}}$ for masking Planck'+'\n' + r'frequency maps (100GHz)')
plt.tight_layout()
plt.savefig(outdir+'TT_frequency_mismatch_mask-cascade')


########################

Tmap_100 = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100), 2048)
Tmap_143 = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_143_inp*mask_map), 1/beam_143), 2048)
Tmap_217 = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_217_inp*mask_map), 1/beam_217), 2048)

Tmap_SMICA = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_SMICA), 1/beam_SMICA), 2048)
Tmap_COMMANDER = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_COMMANDER), 1/beam_SMICA), 2048)
Tmap_NILC = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_NILC), 1/beam_SMICA), 2048)
Tmap_SEVEM = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_SEVEM), 1/beam_SMICA), 2048)

### Load / compute Cls (bin 0)
# Temperature
print('   computing ClTT')
Tcl_SMICA = hp.anafast(Tmap_SMICA)
Tcl_COMMANDER = hp.anafast(Tmap_COMMANDER)
Tcl_NILC = hp.anafast(Tmap_NILC)
Tcl_SEVEM = hp.anafast(Tmap_SEVEM)
Tcl_100 = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)) / fsky
Tcl_143 = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_143_inp*mask_map), 1/beam_143)) / fsky
Tcl_217 = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_217_inp*mask_map), 1/beam_217)) / fsky
ClTT_noise = np.append(Tcl_noise, Tcl_noise[-1])
ClTT_SMICA = np.append(Tcl_SMICA, Tcl_SMICA[-1])
ClTT_COMMANDER = np.append(Tcl_COMMANDER, Tcl_COMMANDER[-1])
ClTT_NILC = np.append(Tcl_NILC, Tcl_NILC[-1])
ClTT_SEVEM = np.append(Tcl_SEVEM, Tcl_SEVEM[-1])
ClTT_100 = np.append(Tcl_100, Tcl_100[-1])
ClTT_143 = np.append(Tcl_143, Tcl_143[-1])
ClTT_217 = np.append(Tcl_217, Tcl_217[-1])
ClTT_noise[:100] = ClTT_SMICA[:100] = ClTT_COMMANDER[:100] = ClTT_NILC[:100] = ClTT_SEVEM[:100] = ClTT_100[:100] = ClTT_143[:100] = ClTT_217[:100] = 1e15

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

print('   computing noise')
ells = np.array([1,2,3,4,5,6,10,50,100,500,1200,2400,3900,6100])
if os.path.exists('data/cache/Noise_Tnoise_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max()):
    filedata = np.load('data/cache/Noise_Tnoise_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max())
    Noise_Tnoise_gunwise = interp1d(filedata['ells'], filedata['Noise_Tnoise_gunwise'], bounds_error=False, fill_value='extrapolate')(np.arange(6144))
else:
    Noise_Tnoise_gunwise = np.zeros(ells.size)
    for l, ell in enumerate(ells):
        print('      @ ell = %d' % ell)
        Noise_Tnoise_gunwise[l] = Noise_vr_diag(6144, 0, 0, ell, ClTT_noise, Cldd)
    np.savez('data/cache/Noise_Tnoise_gunwise_debeam_tau=g_lmax=%d.npz'%ells.max(), ells=ells, Noise_Tnoise_gunwise=Noise_Tnoise_gunwise)
    Noise_Tnoise_gunwise = interp1d(ells, Noise_Tnoise_gunwise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

if os.path.exists('data/cache/Noise_TSMICA_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max()):
    filedata = np.load('data/cache/Noise_TSMICA_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max())
    Noise_TSMICA_gunwise = interp1d(filedata['ells'], filedata['Noise_TSMICA_gunwise'], bounds_error=False, fill_value='extrapolate')(np.arange(6144))
else:
    Noise_TSMICA_gunwise = np.zeros(ells.size)
    #ClTT_SMICA_zeroed = ClTT_SMICA.copy()
    #ClTT_SMICA_zeroed[2501:] = 0.
    for l, ell in enumerate(ells):
        print('      @ ell = %d' % ell)
        Noise_TSMICA_gunwise[l] = Noise_vr_diag(6144, 0, 0, ell, ClTT_SMICA, Cldd)
    np.savez('data/cache/Noise_TSMICA_gunwise_debeam_tau=g_lmax=%d.npz'%ells.max(), ells=ells, Noise_TSMICA_gunwise=Noise_TSMICA_gunwise)
    Noise_TSMICA_gunwise = interp1d(ells, Noise_TSMICA_gunwise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

if os.path.exists('data/cache/Noise_TCOMMANDER_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max()):
    filedata = np.load('data/cache/Noise_TCOMMANDER_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max())
    Noise_TCOMMANDER_gunwise = interp1d(filedata['ells'], filedata['Noise_TCOMMANDER_gunwise'], bounds_error=False, fill_value='extrapolate')(np.arange(6144))
else:
    Noise_TCOMMANDER_gunwise = np.zeros(ells.size)
    for l, ell in enumerate(ells):
        print('      @ ell = %d' % ell)
        Noise_TCOMMANDER_gunwise[l] = Noise_vr_diag(6144, 0, 0, ell, ClTT_COMMANDER, Cldd)
    np.savez('data/cache/Noise_TCOMMANDER_gunwise_debeam_tau=g_lmax=%d.npz'%ells.max(), ells=ells, Noise_TCOMMANDER_gunwise=Noise_TCOMMANDER_gunwise)
    Noise_TCOMMANDER_gunwise = interp1d(ells, Noise_TCOMMANDER_gunwise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

if os.path.exists('data/cache/Noise_TNILC_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max()):
    filedata = np.load('data/cache/Noise_TNILC_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max())
    Noise_TNILC_gunwise = interp1d(filedata['ells'], filedata['Noise_TNILC_gunwise'], bounds_error=False, fill_value='extrapolate')(np.arange(6144))
else:
    Noise_TNILC_gunwise = np.zeros(ells.size)
    for l, ell in enumerate(ells):
        print('      @ ell = %d' % ell)
        Noise_TNILC_gunwise[l] = Noise_vr_diag(6144, 0, 0, ell, ClTT_NILC, Cldd)
    np.savez('data/cache/Noise_TNILC_gunwise_debeam_tau=g_lmax=%d.npz'%ells.max(), ells=ells, Noise_TNILC_gunwise=Noise_TNILC_gunwise)
    Noise_TNILC_gunwise = interp1d(ells, Noise_TNILC_gunwise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

if os.path.exists('data/cache/Noise_TSEVEM_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max()):
    filedata = np.load('data/cache/Noise_TSEVEM_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max())
    Noise_TSEVEM_gunwise = interp1d(filedata['ells'], filedata['Noise_TSEVEM_gunwise'], bounds_error=False, fill_value='extrapolate')(np.arange(6144))
else:
    Noise_TSEVEM_gunwise = np.zeros(ells.size)
    for l, ell in enumerate(ells):
        print('      @ ell = %d' % ell)
        Noise_TSEVEM_gunwise[l] = Noise_vr_diag(6144, 0, 0, ell, ClTT_SEVEM, Cldd)
    np.savez('data/cache/Noise_TSEVEM_gunwise_debeam_tau=g_lmax=%d.npz'%ells.max(), ells=ells, Noise_TSEVEM_gunwise=Noise_TSEVEM_gunwise)
    Noise_TSEVEM_gunwise = interp1d(ells, Noise_TSEVEM_gunwise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

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

hp.mollview(mask_map, title='Input Map Mask')
plt.savefig(outdir + 'maps_mask')

hp.mollview(Tmap_gauss, title='Gaussian Realization of Planck CMB Noise')
plt.savefig(outdir + 'maps_Tmap_gauss')

hp.mollview(Tmap_noise, title='Planck Noise Simulation')
plt.savefig(outdir + 'maps_Tmap_noise')

hp.mollview(Tmap_100, title='Planck 100GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_100')

hp.mollview(Tmap_143, title='Planck 143GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_143')

hp.mollview(Tmap_217, title='Planck 217GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_217')

hp.mollview(Tmap_SMICA, title='Planck SMICA CMB', norm='hist')
plt.savefig(outdir+'maps_Tmap_SMICA')

hp.mollview(Tmap_COMMANDER, title='Planck COMMANDER CMB', norm='hist')
plt.savefig(outdir+'maps_Tmap_COMMANDER')

hp.mollview(Tmap_NILC, title='Planck NILC CMB', norm='hist')
plt.savefig(outdir+'maps_Tmap_NILC')

hp.mollview(Tmap_SEVEM, title='Planck SEVEM CMB', norm='hist')
plt.savefig(outdir+'maps_Tmap_SEVEM')

hp.mollview(lssmap_gauss,title='Gaussian Realization of unWISE gg Spectrum')
plt.savefig(outdir + 'maps_lssmap_gauss')

hp.mollview(lssmap_unwise, title='unWISE Blue Sample')
plt.savefig(outdir + 'maps_lssmap_unwise')

plt.close('all')

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
Noise_T143_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_143, Cldd)]*6144)
Noise_T217_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd)]*6144)

Tmap_SMICA_filtered,     lssmap_unwise_filtered, outmap_TSMICA_gunwise     = combine(Tmap_SMICA, lssmap_unwise, ClTT_SMICA, Noise_TSMICA_gunwise)
Tmap_COMMANDER_filtered, lssmap_unwise_filtered, outmap_TCOMMANDER_gunwise = combine(Tmap_COMMANDER, lssmap_unwise, ClTT_COMMANDER, Noise_TCOMMANDER_gunwise)
Tmap_NILC_filtered,      lssmap_unwise_filtered, outmap_TNILC_gunwise      = combine(Tmap_NILC, lssmap_unwise, ClTT_NILC, Noise_TNILC_gunwise)
Tmap_SEVEM_filtered,     lssmap_unwise_filtered, outmap_TSEVEM_gunwise     = combine(Tmap_SEVEM, lssmap_unwise, ClTT_SEVEM, Noise_TSEVEM_gunwise)
Tmap_100_filtered,       lssmap_gauss_filtered,  outmap_T100_ggauss        = combine(Tmap_100, lssmap_gauss, ClTT_100, Noise_T100_gunwise)
Tmap_143_filtered,       _,                      outmap_T143_ggauss        = combine(Tmap_143, lssmap_gauss, ClTT_143, Noise_T143_gunwise)
Tmap_217_filtered,       _,                      outmap_T217_ggauss        = combine(Tmap_217, lssmap_gauss, ClTT_217, Noise_T217_gunwise)

twopt(outmap_TSMICA_gunwise, Noise_TSMICA_gunwise, r'Power Spectrum of T[SMICA] $\times$ lss[unWISE]', 'twopt_TSMICA_gunwise')
twopt(outmap_TCOMMANDER_gunwise, Noise_TCOMMANDER_gunwise, r'Power Spectrum of T[COMMANDER] $\times$ lss[unWISE]', 'twopt_TCOMMANDER_gunwise')
twopt(outmap_TNILC_gunwise, Noise_TNILC_gunwise, r'Power Spectrum of T[NILC] $\times$ lss[unWISE]', 'twopt_TNILC_gunwise')
twopt(outmap_TSEVEM_gunwise, Noise_TSEVEM_gunwise, r'Power Spectrum of T[SEVEM] $\times$ lss[unWISE]', 'twopt_TSEVEM_gunwise')
twopt(outmap_T100_ggauss, Noise_T100_gunwise, r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss')
twopt(outmap_T143_ggauss, Noise_T143_gunwise, r'Power Spectrum of T[143GHz] $\times$ lss[gauss]', 'twopt_T143_ggauss')
twopt(outmap_T217_ggauss, Noise_T217_gunwise, r'Power Spectrum of T[217GHz] $\times$ lss[gauss]', 'twopt_T217_ggauss')


hp.mollview(outmap_T100_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T100_ggauss')

hp.mollview(outmap_T143_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T143_ggauss')

hp.mollview(outmap_T217_ggauss,norm='hist')
plt.savefig(outdir+'outmap_T217_ggauss')



#### Test unit to get it working, 100GHz:
## this part works
Tcl_100 = hp.anafast(Tmap_100)
ClTT_100 = np.append(Tcl_100, Tcl_100[-1])
ClTT_100[:100]=1e15
Noise_T100_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100, Cldd)]*6144)  #calculate theory noise based on unmasked debeamed Cls
Tmap_100_filtered,       lssmap_gauss_filtered,  outmap_T100_ggauss        = combine(Tmap_100, lssmap_gauss, ClTT_100, Noise_T100_gunwise)  # Reconstruct using unmasked debeamed Cls 

twopt(outmap_T100_ggauss, np.array([Noise_vr_diag(6143, 0, 0, 5, hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100)) / fsky, Cldd[:-1])/fsky**2]*6144), r'Power Spectrum of T[100GHz] $\times$ lss[gauss]', 'twopt_T100_ggauss_nomaskedexceptintoNtheory-fskymissing')

## does it work for 217
Tcl_217 = hp.anafast(Tmap_217)
ClTT_217 = np.append(Tcl_217, Tcl_217[-1])
ClTT_217[:217]=1e15
Noise_T217_gunwise = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_217, Cldd)]*6144)  #calculate theory noise based on unmasked debeamed Cls
Tmap_217_filtered,       lssmap_gauss_filtered,  outmap_T217_ggauss        = combine(Tmap_217, lssmap_gauss, ClTT_217, Noise_T217_gunwise)  # Reconstruct using unmasked debeamed Cls 

twopt(outmap_T217_ggauss, np.array([Noise_vr_diag(6143, 0, 0, 5, hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_217_inp*mask_map), 1/beam_217)) / fsky, Cldd[:-1])/fsky**2]*6144), r'Power Spectrum of T[217GHz] $\times$ lss[gauss]', 'twopt_T217_ggauss_nomaskedexceptintoNtheory-fskymissing')



### Can we build and justify it:
Tmap_100_refresh = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_100_inp), 1/beam_100), 2048)
unmasked_debeamed_Cls_100 = hp.anafast(Tmap_100_refresh)
ClTT_100_unmasked_debeamed = np.append(unmasked_debeamed_Cls_100, unmasked_debeamed_Cls_100[-1])
ClTT_100_unmasked_debeamed[:100] = 1e15


## We can't interchange the mask and debeaming because of numerical issues, it must be masked -> debeamed.
## But we cannot mask before the inverse TT filter or the harmonic transform mixes the mask across all modes.
## So the mask is basically applied after, meaning we reconstruct on the full sky.
## Since we showed that for frequency maps the masked/unmasked sky Cls differ by more than a factor of fsky
## (probably from cutting out the galactic plane and losing overall power), we have to use a theory
## noise that is consistent with a full sky reconstruction: that is, the theory Cls going into the theory
## noise are those of the unmasked > debeamed sky.
## Later down a plot shows that the masked/unmasked output differ by more than a factor of fsky
Noise_T100_gunwise_unmasked_debeamed = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100_unmasked_debeamed, Cldd)]*6144)  #calculate theory noise based on unmasked debeamed Cls

## A masked reconstruction is one where the power is determined by the unmasked regions,
## which in our case means regions outside the galactic plane. The contribution of these
## regions to the power spectrum:
masked_debeamed_Cls_100 = hp.alm2cl(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100))
ClTT_100_masked_debeamed = np.append(masked_debeamed_Cls_100, masked_debeamed_Cls_100[-1])
ClTT_100_masked_debeamed[:100] = 1e15

# compute the theory noise and pretend it is that of a full sky map since our inputs had an fsky: i.e. divide by fsky
Noise_T100_gunwise_masked_debeamed = np.array([Noise_vr_diag(6144, 0, 0, 5, ClTT_100_masked_debeamed, Cldd)]*6144) / fsky


Tmap_100_ref_mask = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_100_inp*mask_map), 1/beam_100), 2048)*mask_map
Tmap_100_ref = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_100_inp), 1/beam_100), 2048)

# print('   filtering and plotting maps')
dTlm = hp.map2alm(Tmap_100_ref_mask)
dlm  = hp.map2alm(lssmap_gauss)
cltaudg = Cltaudd.copy() * bin_width
dTlm_xi_fullsky = hp.almxfl(dTlm, np.divide(np.ones(ClTT_100_masked_debeamed.size), ClTT_100_masked_debeamed, out=np.zeros_like(np.ones(ClTT_100_masked_debeamed.size)), where=ClTT_100_masked_debeamed!=0))
dlm_zeta_fullsky = hp.almxfl(dlm, np.divide(cltaudg, Cldd, out=np.zeros_like(cltaudg), where=Cldd!=0))
Tmap_filtered_fullsky = hp.alm2map(dTlm_xi_fullsky, 2048)
lssmap_filtered_fullsky = hp.alm2map(dlm_zeta_fullsky, 2048)
outmap_filtered_fullsky = Tmap_filtered_fullsky*lssmap_filtered_fullsky
const_noise_fullsky = np.median(Noise_T100_gunwise_masked_debeamed[10:100])
outmap_fullsky = outmap_filtered_fullsky * const_noise_fullsky * mask_map
fullsky_outmap = hp.anafast(outmap_fullsky)

## This plot shows that the masked/unmasked outputs differ by more than a factor of fsky,
## as we saw with the TT inputs. This is probably because the galactic plane is cut out
## and there is just less overall power as a result even accounting for fsky.
maskedsky_outmap = hp.anafast(outmap_fullsky*mask_map)

plt.figure()
plt.loglog(fullsky_outmap,label='fullsky outmap')
plt.loglog(maskedsky_outmap,label='maskedsky outmap')
plt.loglog(Noise_T100_gunwise_unmasked_debeamed*fsky,label='noise used for recon')
plt.loglog(Noise_T100_gunwise_masked_debeamed * fsky,label='noise from masked inputs / fsky')
plt.legend()
plt.savefig(outdir+'outs100')

### The total power of the masked reconstruction = power of unmasked reconstruction regions spread out over full sky (has fsky in it)
### power of theory noise based on unmasked regions spread out over full sky (divide by fsky)
### means hp.anafast(mask*recon) power should be equivalent to the full sky noise extrapolated from unmasked regions.
### this is what we are plotting, but we are a factor of fsky too small on theory noise


# TASK 3: Reconstruction noise for frequency maps x unWISE is strongly affected by galactic foregrounds.
#         Show the f_sky vs frequency for which the reconstruction noise agrees best with the
#         theoretical expectations. We would expect the necessary f_sky to grow with frequency among
#         the 100, 143, and 217 GHz maps.

print('100GHz theory noise higher factor: ', hp.anafast(outmap_T100_ggauss*mask_map,lmax=200).mean()/fsky/Noise_T100_gunwise.mean())
print('143GHz theory noise higher factor: ', hp.anafast(outmap_T143_ggauss*mask_map,lmax=200).mean()/fsky/Noise_T143_gunwise.mean())
print('217GHz theory noise higher factor: ', hp.anafast(outmap_T217_ggauss*mask_map,lmax=200).mean()/fsky/Noise_T217_gunwise.mean())


# clsmica = hp.anafast(outmap_TNILC_gunwise)
# clsmica90 = hp.anafast(outmap_TNILC_gunwise*gal90)
# clsmica80 = hp.anafast(outmap_TNILC_gunwise*gal80)
# clsmicaunwise = hp.anafast(outmap_TNILC_gunwise*mask_map)

# plt.figure()
# plt.loglog(clsmica,label='unmasked')
# plt.loglog(clsmica90/fsky90,label='fsky=0.90 galcut')
# plt.loglog(clsmica80/fsky80,label='fsky=0.80 galcut')
# plt.loglog(clsmicaunwise/fsky,label='unWISE mask (70 galcut)')
# plt.loglog(Noise_TNILC_gunwise,label='Theory')
# plt.legend()
# plt.savefig(outdir+'NILC')

# cl100 = hp.anafast(outmap_T100_ggauss)
# cl10090 = hp.anafast(outmap_T100_ggauss*gal90)
# cl10080 = hp.anafast(outmap_T100_ggauss*gal80)
# cl10070 = hp.anafast(outmap_T100_ggauss*gal70)
# cl100unwise = hp.anafast(outmap_T100_ggauss*mask_map)
# cl10060unwise = hp.anafast(outmap_T100_ggauss*gal60*mask_map)
# cl10040unwise = hp.anafast(outmap_T100_ggauss*gal40*mask_map)
# cl10020unwise = hp.anafast(outmap_T100_ggauss*gal20*mask_map)

# plt.figure()
# plt.loglog(cl100,label='unmasked')
# plt.loglog(cl10090/fsky90,label='fsky=0.90 galcut')
# plt.loglog(cl10080/fsky80,label='fsky=0.80 galcut')
# plt.loglog(cl10070/fsky70,label='fsky=0.70 galcut')
# plt.loglog(cl100unwise/fsky,label='unWISE mask (70 galcut)')
# plt.loglog(cl10060unwise/fsky60,label='unWISE mask (60 galcut)')
# plt.loglog(cl10040unwise/fsky40,label='unWISE mask (40 galcut)')
# plt.loglog(cl10020unwise/fsky20,label='unWISE mask (20 galcut)')
# plt.loglog(Noise_T100_gunwise,label='Theory')
# plt.legend()
# plt.savefig(outdir+'T100')


# cl143 = hp.anafast(outmap_T143_ggauss)
# cl14390 = hp.anafast(outmap_T143_ggauss*gal90)
# cl14380 = hp.anafast(outmap_T143_ggauss*gal80)
# cl14370 = hp.anafast(outmap_T143_ggauss*gal70)
# cl143unwise = hp.anafast(outmap_T143_ggauss*mask_map)
# cl14360unwise = hp.anafast(outmap_T143_ggauss*gal60*mask_map)
# cl14340unwise = hp.anafast(outmap_T143_ggauss*gal40*mask_map)
# cl14320unwise = hp.anafast(outmap_T143_ggauss*gal20*mask_map)

# plt.figure()
# plt.loglog(cl143,label='unmasked')
# plt.loglog(cl14390/fsky90,label='fsky=0.90 galcut')
# plt.loglog(cl14380/fsky80,label='fsky=0.80 galcut')
# plt.loglog(cl14370/fsky70,label='fsky=0.70 galcut')
# plt.loglog(cl143unwise/fsky,label='unWISE mask (70 galcut)')
# plt.loglog(cl14360unwise/fsky60,label='unWISE mask (60 galcut)')
# plt.loglog(cl14340unwise/fsky40,label='unWISE mask (40 galcut)')
# plt.loglog(cl14320unwise/fsky20,label='unWISE mask (20 galcut)')
# plt.loglog(Noise_T143_gunwise,label='Theory')
# plt.legend()
# plt.savefig(outdir+'T143')



# cl217 = hp.anafast(outmap_T217_ggauss)
# cl21790 = hp.anafast(outmap_T217_ggauss*gal90)
# cl21780 = hp.anafast(outmap_T217_ggauss*gal80)
# cl21770 = hp.anafast(outmap_T217_ggauss*gal70)
# cl217unwise = hp.anafast(outmap_T217_ggauss*mask_map)
# cl21760unwise = hp.anafast(outmap_T217_ggauss*gal60*mask_map)
# cl21740unwise = hp.anafast(outmap_T217_ggauss*gal40*mask_map)
# cl21720unwise = hp.anafast(outmap_T217_ggauss*gal20*mask_map)

# plt.figure()
# plt.loglog(cl217,label='unmasked')
# plt.loglog(cl21790/fsky90,label='fsky=0.90 galcut')
# plt.loglog(cl21780/fsky80,label='fsky=0.80 galcut')
# plt.loglog(cl21770/fsky70,label='fsky=0.70 galcut')
# plt.loglog(cl217unwise/fsky,label='unWISE mask (70 galcut)')
# plt.loglog(cl21760unwise/fsky60,label='unWISE mask (60 galcut)')
# plt.loglog(cl21740unwise/fsky40,label='unWISE mask (40 galcut)')
# plt.loglog(cl21720unwise/fsky20,label='unWISE mask (20 galcut)')
# plt.loglog(Noise_T217_gunwise,label='Theory')
# plt.legend()
# plt.savefig(outdir+'T217')

