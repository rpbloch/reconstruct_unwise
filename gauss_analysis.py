### Things are okay up to and including T143 and gaussian unWISE, but are weird for T143 and unWISE. What about unWISE and gaussian T143?



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
binning = 10000  # Number of bins to use
outdir = 'plots/analysis/plots_%dx%d/Tmap_prog/binning_%d/'% (8,32,binning)
if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Setting up maps and Cls')
print('   loading maps')
### Load maps (bin 0)
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
#planck_galcut_40 = hp.reorder(np.load('data/planckgal40.npy'), n2r=True)
#mask_map *= planck_galcut_40
fsky = np.where(mask_map!=0)[0].size/mask_map.size
Tmap_noise = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/143ghz/ffp10_noise_143_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
Tmap_143 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
Tmap_SMICA = hp.reorder(fits.open('data/planck_data_testing/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True) / 2.725e6
lssmap_unwise = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
#lssmap_gauss = c.load(basic_conf_dir, 'lss_vr_2048_real=0_bin=0', dir_base=estim_dir+'sims')[0,:]
lssmap_gauss = hp.synfast(hp.anafast(lssmap_unwise), 2048)
#vrmap = c.load(basic_conf_dir, 'vr_full_fine_2048_64_real=0_bin=0', dir_base=estim_dir+'sims')



Tcl_noise = hp.anafast(Tmap_noise)
Tmap_gauss = hp.synfast(Tcl_noise, 2048)

###
Tmap_SMICA += (Tmap_gauss)
#Tmap_143 += Tmap_gauss
###

beam = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6144)

Tmap_SMICA = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_SMICA), 1/beam), 2048)
Tmap_143 = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_143), 1/beam), 2048)

### Load / compute Cls (bin 0)
# Temperature
print('   computing ClTT')
Tcl_143 = hp.anafast(Tmap_143)
Tcl_SMICA = hp.anafast(Tmap_SMICA)
ClTT_noise = np.append(Tcl_noise, Tcl_noise[-1])
ClTT_143 = np.append(Tcl_143, Tcl_143[-1])
ClTT_SMICA = np.append(Tcl_SMICA, Tcl_SMICA[-1])
ClTT_noise[:100] = ClTT_143[:100] = ClTT_SMICA[:100] = 1e15

#done above at the map level
#ClTT_SMICA /= beam**2
#ClTT_143 /= beam**2

print('   loading and interpolating Clgg')
# Galaxy
# with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
#     lines = FILE.readlines()
# ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float')
# clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')
# Cldd = interp1d(ells,clgg, bounds_error=False,fill_value='extrapolate')(np.arange(6145))
# Cldd[:100] = 1e15
Cldd = hp.anafast(lssmap_unwise)
Cldd = np.append(Cldd, Cldd[-1])
Cltaudd = Cldd / bin_width
Cldd[:100] = 1e15

print('   loading Cltaudg and generating gaussian CMB')
# Cross (bin 0)
#Cltaudd = loginterp.log_interpolate_matrix(c.load(basic_conf_dir,'Cl_taud_g_lmax=6144', dir_base = 'Cls/'+c.direc('taud','g',conf)), c.load(basic_conf_dir,'L_sample_lmax=6144', dir_base = 'Cls'))[:,0,0]


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
#ells = np.array([1,2,3,4,5,6,10,50,100,500,1200,2400,3900,6100])
ells = np.array([1,2,3,4,5,6,10,50])
if os.path.exists('data/cache/Noise_T143_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max()):
    filedata = np.load('data/cache/Noise_T143_gunwise_debeam_tau=g_lmax=%d.npz' % ells.max())
    Noise_T143_gunwise = interp1d(filedata['ells'], filedata['Noise_T143_gunwise'], bounds_error=False, fill_value='extrapolate')(np.arange(6144))
else:
    Noise_T143_gunwise = np.zeros(ells.size)
    for l, ell in enumerate(ells):
        print('      @ ell = %d' % ell)
        Noise_T143_gunwise[l] = Noise_vr_diag(6144, 0, 0, ell, ClTT_143, Cldd)
    np.savez('data/cache/Noise_T143_gunwise_debeam_tau=g_lmax=%d.npz'%ells.max(), ells=ells, Noise_T143_gunwise=Noise_T143_gunwise)
    Noise_T143_gunwise = interp1d(ells, Noise_T143_gunwise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

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


print('   filtering and plotting maps')
def combine(Tmap, lssmap, ClTT, Noise, mask_tag=False):
    if mask_tag:
        mask = mask_map.copy()
    else:
        mask = np.ones(mask_map.size)
    dTlm = hp.map2alm(Tmap)
    dlm  = hp.map2alm(lssmap)
    cltaudg = Cltaudd.copy() * bin_width
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
    dlm_zeta = hp.almxfl(dlm, np.divide(cltaudg, Cldd, out=np.zeros_like(cltaudg), where=Cldd!=0))
    Tmap_filtered = hp.alm2map(dTlm_xi, 2048)
    lssmap_filtered = hp.alm2map(dlm_zeta, 2048)
    outmap_filtered = Tmap_filtered*lssmap_filtered
    const_noise = np.median(Noise[10:100])
    outmap = outmap_filtered * const_noise * mask
    return Tmap_filtered * const_noise, lssmap_filtered, outmap

# def combine(Tmap, lssmap, ClTT, Noise, mask_tag=False):
#     if mask_tag:
#         mask = mask_map.copy()
#     else:
#         mask = np.ones(mask_map.size)
#     dTlm = hp.map2alm(Tmap)
#     dlm  = hp.map2alm(lssmap)
#     cltaudg = Cltaudd.copy() * bin_width
#     dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
#     dlm_zeta = hp.almxfl(dlm, np.divide(cltaudg, Cldd, out=np.zeros_like(cltaudg), where=Cldd!=0))
#     Tmap_filtered = hp.alm2map(dTlm_xi, 2048)    * mask
#     lssmap_filtered = hp.alm2map(dlm_zeta, 2048) * mask
#     outmap_filtered = Tmap_filtered*lssmap_filtered
#     const_noise = np.median(Noise[:500])
#     outmap = outmap_filtered * const_noise
#     return Tmap_filtered * const_noise, lssmap_filtered, outmap

def combine_harmonic(Tmap, lssmap, ClTT, Noise, mask_tag):
    if mask_tag:
        mask = mask_map.copy()
    else:
        mask = np.ones(12*2048**2)
    dTlm = hp.map2alm(Tmap)
    dlm  = hp.map2alm(lssmap)
    cltaudg = Cltaudd.copy() * bin_width
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
    dlm_zeta = hp.almxfl(dlm, np.divide(cltaudg, Cldd, out=np.zeros_like(cltaudg), where=Cldd!=0))
    Tmap_filtered = hp.alm2map(dTlm_xi, 2048)
    lssmap_filtered = hp.alm2map(dlm_zeta, 2048)
    outmap_filtered = Tmap_filtered*lssmap_filtered
    outmap = hp.alm2map(hp.almxfl(hp.map2alm(outmap_filtered), Noise), 2048) * mask
    return Tmap_filtered, lssmap_filtered, outmap



hp.mollview(mask_map, title='Input Map Mask')
plt.savefig(outdir + 'maps_mask')

hp.mollview(Tmap_gauss, title='Gaussian Realization of Planck CMB Noise')
plt.savefig(outdir + 'maps_Tmap_gauss')

hp.mollview(Tmap_noise, title='Planck Noise Simulation')
plt.savefig(outdir + 'maps_Tmap_noise')

hp.mollview(Tmap_143, title='Planck 143GHz Sky', norm='hist')
plt.savefig(outdir + 'maps_Tmap_143')

hp.mollview(Tmap_SMICA, title='Planck SMICA CMB', norm='hist')
plt.savefig(outdir+'maps_Tmap_SMICA')

hp.mollview(lssmap_gauss,title='Gaussian Realization of unWISE gg Spectrum')
plt.savefig(outdir + 'maps_lssmap_gauss')

hp.mollview(lssmap_unwise, title='unWISE Blue Sample')
plt.savefig(outdir + 'maps_lssmap_unwise')

plt.close('all')

Tmap_gauss_filtered_const, lssmap_gauss_filtered_const, outmap_Tgauss_ggauss_const = combine(Tmap_gauss, lssmap_gauss, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=False)
Tmap_gauss_filtered_harmonic, lssmap_gauss_filtered_harmonic, outmap_Tgauss_ggauss_harmonic = combine_harmonic(Tmap_gauss, lssmap_gauss, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=False)

plt.figure()
plt.loglog(hp.anafast(outmap_Tgauss_ggauss_const), label=r'N=const')
plt.loglog(hp.anafast(outmap_Tgauss_ggauss_harmonic), label=r'N=N$_\ell$')
plt.loglog(Noise_Tnoise_gunwise, label='Theory', c='k', ls='--')
plt.legend(title='Filtered Inputs Times:')
plt.savefig(outdir+'analysis_impact-of-noise-application')

Tmap_gauss_filtered = Tmap_gauss_filtered_const.copy()
lssmap_gauss_filtered = lssmap_gauss_filtered_const.copy()
outmap_Tgauss_ggauss = outmap_Tgauss_ggauss_const.copy()

Tmap_gauss_filtered_mask, lssmap_gauss_filtered_mask, outmap_Tgauss_ggauss_mask = combine(Tmap_gauss, lssmap_gauss, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=True)
Tmap_noise_filtered, _, outmap_Tnoise_ggauss = combine(Tmap_noise, lssmap_gauss, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=False)
_, _, outmap_Tnoise_gunwise_mask = combine(Tmap_noise, lssmap_unwise, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=True)
Tmap_143_filtered, _, outmap_T143_ggauss = combine(Tmap_143, lssmap_gauss, ClTT_143, Noise_T143_gunwise, mask_tag=False)
Tmap_noise_filtered_mask, _, outmap_Tnoise_ggauss_mask = combine(Tmap_noise, lssmap_gauss, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=True)
Tmap_143_filtered_mask, _, outmap_T143_ggauss_mask = combine(Tmap_143, lssmap_gauss, ClTT_143, Noise_T143_gunwise, mask_tag=True)
Tmap_SMICA_filtered_mask, _, outmap_TSMICA_ggauss_mask = combine(Tmap_SMICA, lssmap_gauss, ClTT_SMICA, Noise_TSMICA_gunwise, mask_tag=True)
_, lssmap_unwise_filtered_mask, outmap_Tgauss_gunwise_mask = combine(Tmap_gauss, lssmap_unwise, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=True)
_, _, outmap_T143_gunwise_mask = combine(Tmap_143, lssmap_unwise, ClTT_143, Noise_T143_gunwise, mask_tag=True)
_, _, outmap_TSMICA_gunwise = combine(Tmap_SMICA, lssmap_unwise, ClTT_SMICA, Noise_TSMICA_gunwise, mask_tag=False)

hp.mollview(outmap_Tgauss_ggauss, title=r'Output T[gauss] $\times$ lss[gauss]')
plt.savefig(outdir + 'outmap_Tgauss_ggauss')

hp.mollview(outmap_Tnoise_ggauss, title=r'Output T[noise] $\times$ lss[gauss]')
plt.savefig(outdir + 'outmap_Tnoise_ggauss')

hp.mollview(outmap_T143_ggauss, title=r'Output T[143GHz] $\times$ lss[gauss]')
plt.savefig(outdir + 'outmap_T143_ggauss')

hp.mollview(outmap_T143_ggauss_mask * mask_map, title=r'Output of Masked T[143GHz] $\times$ lss[gauss]')
plt.savefig(outdir + 'outmap_T143_ggauss_mask')

hp.mollview(outmap_Tgauss_gunwise_mask * mask_map, title=r'Output of Masked T[gauss] $\times$ lss[unWISE]')
plt.savefig(outdir + 'outmap_Tgauss_gunWISE_mask')

hp.mollview(outmap_T143_gunwise_mask * mask_map, title=r'Output of Masked T[143GHz] $\times$ lss[unWISE]')
plt.savefig(outdir + 'outmap_T143_gunWISE_mask')

hp.mollview(outmap_TSMICA_gunwise * mask_map, title=r'Masked Output of T[SMICA] $\times$ lss[unWISE]')
plt.savefig(outdir + 'outmap_TSMICA_gunwise_mask')

plt.close('all')


print('Computing histograms of map cases')
print('   computing histograms of input maps')
n_Tgauss, bins_Tgauss, _ = plt.hist(Tmap_gauss, bins=binning)
n_Tgauss_filtered_mask, bins_Tgauss_filtered_mask, _ = plt.hist(Tmap_gauss_filtered_mask[np.where(mask_map!=0)], bins=binning)
n_Tnoise, bins_Tnoise, _ = plt.hist(Tmap_noise, bins=binning)
n_Tnoise_mask, bins_Tnoise_mask, _ = plt.hist(Tmap_noise[np.where(mask_map!=0)], bins=binning)
n_T143, bins_T143, _ = plt.hist(Tmap_143, bins=binning)
n_T143_mask, bins_T143_mask, _ = plt.hist(Tmap_143[np.where(mask_map != 0)], bins=binning)
n_Tgauss_filtered, bins_Tgauss_filtered, _ = plt.hist(Tmap_gauss_filtered, bins=binning)
n_Tnoise_filtered, bins_Tnoise_filtered, _ = plt.hist(Tmap_noise_filtered, bins=binning)
n_Tnoise_filtered_mask, bins_Tnoise_filtered_mask, _ = plt.hist(Tmap_noise_filtered[np.where(mask_map!=0)], bins=binning)
n_T143_filtered, bins_T143_filtered, _ = plt.hist(Tmap_143_filtered, bins=binning)
n_T143_filtered_mask, bins_T143_filtered_mask, _ = plt.hist(Tmap_143_filtered[np.where(mask_map!=0)], bins=binning)
n_TSMICA_mask, bins_TSMICA_mask, _ = plt.hist(Tmap_SMICA_filtered_mask[np.where(mask_map!=0)], bins=binning)
n_TSMICA_filtered_mask, bins_TSMICA_filtered_mask, _ = plt.hist(Tmap_SMICA_filtered_mask[np.where(mask_map!=0)], bins=binning)

n_ggauss, bins_ggauss, _ = plt.hist(lssmap_gauss, bins=binning)
n_gunwise_mask, bins_gunwise_mask, _ = plt.hist(lssmap_unwise[np.where(mask_map!=0)], bins=binning)
n_ggauss_mask, bins_ggauss_mask, _ = plt.hist(lssmap_gauss[np.where(mask_map!=0)], bins=binning)
n_ggauss_filtered, bins_ggauss_filtered, _ = plt.hist(lssmap_gauss_filtered, bins=binning)
n_ggauss_filtered_mask, bins_ggauss_filtered_mask, _ = plt.hist(lssmap_gauss_filtered[np.where(mask_map!=0)], bins=binning)
n_gunwise_filtered_mask, bins_gunwise_filtered_mask, _ = plt.hist(lssmap_unwise_filtered_mask[np.where(mask_map!=0)], bins=binning)


#vrn, vrbins, vrpatches = plt.hist(vrmap, bins=binning)

print('   plotting histograms')
def histplot(n_T, n_T_filtered, n_g, n_g_filtered, plottitle, filename):
    plt.figure()
    plt.plot(n_T/simps(n_T),label='Temperature')
    plt.plot(n_T_filtered/simps(n_T_filtered), label='Temperature (filtered)')
    plt.plot(n_g/simps(n_g),label='LSS')
    plt.plot(n_g_filtered/simps(n_g_filtered), label='LSS (filtered)')
    y1, y2 = plt.ylim()
    plt.plot([(binning/2)-1,(binning/2)-1],[-1,1],ls='--',c='gray')
    plt.ylim([y1, y2])
    plt.title(plottitle)
    plt.xlabel('Bin Number')
    plt.ylabel('Normalized # of Pixels')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

# plt.figure()
# plt.plot(bin_centres(vrbins), vrn)
# plt.title('Signal velocity field in bin 0')
# plt.xlabel('v/c')
# plt.ylabel('Normalized # of Pixels')
# plt.savefig(outdir + 'vr_hist')

histplot(n_Tgauss, n_Tgauss_filtered, n_ggauss, n_ggauss_filtered, r'Pixel Distribution of Input Maps (T[gauss] $\times$ lss[gauss])', 'histogram_inputs_Tgauss_lssgauss')
histplot(n_Tnoise, n_Tnoise_filtered, n_ggauss, n_ggauss_filtered, r'Pixel Distribution of Input Maps (T[noise] $\times$ lss[gauss])', 'histogram_inputs_Tnoise_lssgauss')
histplot(n_Tnoise_mask, n_Tnoise_filtered_mask, n_ggauss_mask, n_ggauss_filtered_mask, r'Pixel Distribution of Masked Input Maps (T[noise] $\times$ lss[gauss])', 'histogram_inputs_Tnoise_lssgauss_mask')
histplot(n_T143, n_T143_filtered, n_ggauss, n_ggauss_filtered, r'Pixel Distribution of Input Maps (T[143] $\times$ lss[gauss])', 'histogram_inputs_T143_lssgauss')
histplot(n_T143_mask, n_T143_filtered_mask, n_ggauss_mask, n_ggauss_filtered_mask, r'Pixel Distribution of Masked Input Maps (T[143] $\times$ lss[gauss])', 'histogram_inputs_T143_lssgauss_masked')
histplot(n_TSMICA_mask, n_TSMICA_filtered_mask, n_ggauss_mask, n_ggauss_filtered_mask, r'Pixel Distribution of Masked Input Maps (T[SMICA] $\times$ lss[gauss])', 'histogram_inputs_TSMICA_lssgauss_mask')
histplot(n_T143, n_T143_filtered, n_gunwise_mask, n_gunwise_filtered_mask, r'Pixel Distribution of Masked Input Maps (T[143] $\times$ lss[unWISE])', 'histogram_inputs_T143_lssunwise_masked')
histplot(n_Tnoise_mask, n_Tnoise_filtered_mask, n_gunwise_mask, n_gunwise_filtered_mask, r'Pixel Distribution of Masked Input Maps (T[noise] $\times$ lss[unWISE])', 'histogram_inputs_Tnoise_lssunwise_masked')
histplot(n_TSMICA_mask, n_TSMICA_filtered_mask, n_gunwise_mask, n_gunwise_filtered_mask, r'Pixel Distribution of Masked Input Maps (T[SMICA] $\times$ lss[unWISE])', 'histogram_inputs_TSMICA_lssunwise_mask')
plt.close('all')

print('   computing histograms of outputs and normal product distribution')
bessel = lambda bins, Tmap, lssmap : kn(0, np.abs(bin_centres(bins)) / (np.std(Tmap)*np.std(lssmap)))
normal_product = lambda bins, Tmap, lssmap : bessel(bins,Tmap,lssmap) / (np.pi * np.std(Tmap) * np.std(lssmap))
pixel_scaling = lambda distribution : (12*2048**2) * (distribution / simps(distribution))
pixel_scaling_masked = lambda distribution : np.where(mask_map!=0)[0].size * (distribution / simps(distribution))

def histplot_normprod(bins, n_Tg, normprod_Tg, plottitle, filename, tolerance=0.01):
    # Tolerance is the % of the max y value that defines the x axis limits so we can zoom on the interesting part.
    #x,y,z=plt.hist(vrmap,bins=bins,histtype='step')
    plt.close()
    plt.figure()
    if 'mask' in plottitle.lower():
        plt.plot(bin_centres(bins), pixel_scaling_masked(n_Tg), label='Map Pixels')
        plt.plot(bin_centres(bins), pixel_scaling_masked(normprod_Tg), ls='--', label='Normal Product Distribution')       
        #plt.plot(bin_centres(bins), pixel_scaling_masked(interp1d(bin_centres(vrbins),vrn,bounds_error=False,fill_value=0.)(bin_centres(bins))), label='Signal Map', alpha=0.5)
        #plt.plot(bin_centres(bins), pixel_scaling_masked(x))
    else:
        plt.plot(bin_centres(bins), pixel_scaling(n_Tg), label='Map Pixels')
        plt.plot(bin_centres(bins), pixel_scaling(normprod_Tg), ls='--', label='Normal Product Distribution')
        #plt.plot(bin_centres(bins), pixel_scaling(interp1d(bin_centres(vrbins),vrn,bounds_error=False,fill_value=0.)(bin_centres(bins))), label='Signal Map', alpha=0.5)
        #plt.plot(bin_centres(bins), pixel_scaling(x))
    plt.title(plottitle)
    plt.xlabel('v/c')
    plt.ylabel('# of Pixels')
    #plt.xlim([-.05,.05])
    plt.xlim([bin_centres(bins)[np.where(normprod_Tg>(tolerance*normprod_Tg.max()))][0], bin_centres(bins)[np.where(normprod_Tg>(tolerance*normprod_Tg.max()))][-1]])
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

n_output_Tgauss_ggauss, bins_output_Tgauss_ggauss, _ = plt.hist(outmap_Tgauss_ggauss, bins=binning)
n_output_Tgauss_ggauss_mask, bins_output_Tgauss_ggauss_mask, _ = plt.hist(outmap_Tgauss_ggauss_mask[np.where(mask_map!=0)], bins=binning)
n_output_Tnoise_ggauss, bins_output_Tnoise_ggauss, _ = plt.hist(outmap_Tnoise_ggauss, bins=binning)
n_output_Tnoise_ggauss_mask, bins_output_Tnoise_ggauss_mask, _ = plt.hist(outmap_Tnoise_ggauss_mask[np.where(mask_map!=0)], bins=binning)
n_output_T143_ggauss, bins_output_T143_ggauss, _ = plt.hist(outmap_T143_ggauss, bins=binning)
n_output_T143_ggauss_mask, bins_output_T143_ggauss_mask, _ = plt.hist(outmap_T143_ggauss_mask[np.where(mask_map!=0)], bins=binning)
n_output_TSMICA_ggauss_mask, bins_output_TSMICA_ggauss_mask, _ = plt.hist(outmap_TSMICA_ggauss_mask[np.where(mask_map!=0)], bins=binning)
n_output_Tgauss_gunwise_mask, bins_output_Tgauss_gunwise_mask, _ = plt.hist(outmap_Tgauss_gunwise_mask[np.where(mask_map!=0)], bins=binning)
n_output_T143_gunwise_mask, bins_output_T143_gunwise_mask, _ = plt.hist(outmap_T143_gunwise_mask[np.where(mask_map!=0)], bins=binning)
n_output_Tnoise_gunwise_mask, bins_output_Tnoise_gunwise_mask, _ = plt.hist(outmap_Tnoise_gunwise_mask[np.where(mask_map!=0)], bins=binning)
n_output_TSMICA_gunwise_mask, bins_output_TSMICA_gunwise_mask, _ = plt.hist(outmap_TSMICA_gunwise[np.where(mask_map!=0)], bins=binning)

normal_product_Tgauss_ggauss = normal_product(bins_output_Tgauss_ggauss, Tmap_gauss_filtered, lssmap_gauss_filtered)
normal_product_Tgauss_ggauss_mask = normal_product(bins_output_Tgauss_ggauss_mask, Tmap_gauss_filtered_mask[np.where(mask_map!=0)], lssmap_gauss_filtered_mask[np.where(mask_map!=0)])
normal_product_Tnoise_ggauss = normal_product(bins_output_Tnoise_ggauss, Tmap_noise_filtered, lssmap_gauss_filtered)
normal_product_Tnoise_ggauss_mask = normal_product(bins_output_Tnoise_ggauss_mask, Tmap_noise_filtered_mask[np.where(mask_map!=0)], lssmap_gauss_filtered_mask[np.where(mask_map!=0)])
normal_product_T143_ggauss = normal_product(bins_output_T143_ggauss, Tmap_143_filtered, lssmap_gauss_filtered)
normal_product_T143_ggauss_mask = normal_product(bins_output_T143_ggauss_mask, Tmap_143_filtered_mask[np.where(mask_map!=0)], lssmap_gauss_filtered_mask[np.where(mask_map!=0)])
normal_product_TSMICA_ggauss_mask = normal_product(bins_output_TSMICA_ggauss_mask, Tmap_SMICA_filtered_mask[np.where(mask_map!=0)], lssmap_gauss_filtered_mask[np.where(mask_map!=0)])
normal_product_Tgauss_gunwise_mask = pixel_scaling_masked(normal_product(bins_output_Tgauss_gunwise_mask, Tmap_gauss_filtered_mask[np.where(mask_map!=0)], lssmap_unwise_filtered_mask[np.where(mask_map!=0)]))
normal_product_T143_gunwise_mask = pixel_scaling_masked(normal_product(bins_output_T143_gunwise_mask, Tmap_143_filtered_mask[np.where(mask_map!=0)], lssmap_unwise_filtered_mask[np.where(mask_map!=0)]))
normal_product_Tnoise_gunwise_mask = pixel_scaling_masked(normal_product(bins_output_Tnoise_gunwise_mask, Tmap_noise_filtered_mask[np.where(mask_map!=0)], lssmap_unwise_filtered_mask[np.where(mask_map!=0)]))
normal_product_TSMICA_gunwise_mask = pixel_scaling_masked(normal_product(bins_output_TSMICA_gunwise_mask, Tmap_SMICA_filtered_mask[np.where(mask_map!=0)], lssmap_unwise_filtered_mask[np.where(mask_map!=0)]))

histplot_normprod(bins_output_Tgauss_ggauss, n_output_Tgauss_ggauss, normal_product_Tgauss_ggauss, r'Pixel Distribution of Output Map (T[gauss] $\times$ lss[gauss])', 'norm_prod_Tgauss_ggauss')
histplot_normprod(bins_output_Tgauss_ggauss_mask, n_output_Tgauss_ggauss_mask, normal_product_Tgauss_ggauss_mask, r'Pixel Distribution of (Masked) Output Map (T[gauss] $\times$ lss[gauss])', 'norm_prod_Tgauss_ggauss_mask')
histplot_normprod(bins_output_Tnoise_ggauss, n_output_Tnoise_ggauss, normal_product_Tnoise_ggauss, r'Pixel Distribution of Output Map (T[noise] $\times$ lss[gauss])', 'norm_prod_Tnoise_ggauss')
histplot_normprod(bins_output_Tnoise_ggauss_mask, n_output_Tnoise_ggauss_mask, normal_product_Tnoise_ggauss_mask, r'Pixel Distribution of Masked Input Maps (T[noise] $\times$ lss[gauss])', 'norm_prod_Tnoise_ggauss_mask')
histplot_normprod(bins_output_T143_ggauss, n_output_T143_ggauss, normal_product_T143_ggauss, r'Pixel Distribution of Output Map (T[143] $\times$ lss[gauss])', 'norm_prod_T143_ggauss')
histplot_normprod(bins_output_T143_ggauss_mask, n_output_T143_ggauss_mask, normal_product_T143_ggauss_mask, r'Pixel Distribution of Masked Input Maps (T[143] $\times$ lss[gauss])', 'norm_prod_T143_ggauss_mask')
histplot_normprod(bins_output_TSMICA_ggauss_mask, n_output_TSMICA_ggauss_mask, normal_product_TSMICA_ggauss_mask, r'Pixel Distribution of Masked Input Maps (T[SMICA] $\times$ lss[gauss])', 'norm_prod_TSMICA_ggauss_mask')
histplot_normprod(bins_output_Tgauss_gunwise_mask, n_output_Tgauss_gunwise_mask, normal_product_Tgauss_gunwise_mask, r'Pixel Distribution of Masked Input Maps (T[gauss] $\times$ lss[unWISE])', 'norm_prod_Tgauss_gunWISE_mask')
histplot_normprod(bins_output_T143_gunwise_mask, n_output_T143_gunwise_mask, normal_product_T143_gunwise_mask, r'Pixel Distribution of Masked Input Maps (T[143] $\times$ lss[unWISE])', 'norm_prod_T143_gunWISE_mask')
histplot_normprod(bins_output_Tnoise_gunwise_mask, n_output_Tnoise_gunwise_mask, normal_product_Tnoise_gunwise_mask, r'Pixel Distribution of Masked Input Maps (T[noise] $\times$ lss[unWISE])', 'norm_prod_Tnoise_gunWISE_mask')
histplot_normprod(bins_output_TSMICA_gunwise_mask, n_output_TSMICA_gunwise_mask, normal_product_TSMICA_gunwise_mask, r'Masked Pixel Distribution of Input Maps (T[SMICA] $\times$ lss[unWISE])', 'norm_prod_TSMICA_gunwise_mask')

plt.close('all')





def twopt(outmap, theory_noise, plottitle, filename):
    fsky = np.where(mask_map!=0)[0].size / mask_map.size
    recon_noise_masked = hp.anafast(outmap * mask_map)
    recon_noise = hp.anafast(outmap)
    plt.figure()
    plt.loglog(np.arange(2,6144), recon_noise[2:], label='Reconstruction')
    plt.loglog(np.arange(2,6144), recon_noise_masked[2:] / fsky, label='Masked Reconstruction / fsky') 
    plt.loglog(np.arange(2,6144), theory_noise[2:], label='Theory')
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

def twopt_maskonly(outmap, theory_noise, plottitle, filename):
    fsky = np.where(mask_map!=0)[0].size / mask_map.size
    recon_noise = hp.anafast(outmap * mask_map)
    plt.figure()
    plt.loglog(np.arange(2,6144), recon_noise[2:], label='Masked Reconstruction')
    plt.loglog(np.arange(2,6144), theory_noise[2:] * fsky, label='Theory * fsky')
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')


#twopt(outmap_Tnoise_ggauss, Noise_Tnoise_gunwise, r'Power Spectrum of T[noise] $\times$ lss[gauss]', 'twopt_Tnoise_ggauss')
#twopt(outmap_Tnoise_gunwise_mask, Noise_Tnoise_gunwise, r'Power Spectrum of T[noise] $\times$ lss[unWISE]', 'twopt_Tnoise_gunwise')
twopt(outmap_Tgauss_ggauss, Noise_Tnoise_gunwise, r'Power Spectrum of T[gauss] $\times$ lss[gauss]', 'twopt_Tgauss_ggauss')
twopt_maskonly(outmap_Tgauss_gunwise_mask, Noise_Tnoise_gunwise, r'Power Spectrum of T[gauss] $\times$ lss[unWISE]', 'twopt_Tgauss_gunwise')
twopt_maskonly(outmap_TSMICA_ggauss_mask, Noise_TSMICA_gunwise, r'Power Spectrum of T[SMICA] $\times$ lss[gauss]', 'twopt_TSMICA_ggauss')
twopt(outmap_TSMICA_gunwise, Noise_TSMICA_gunwise, r'Power Spectrum of T[SMICA] $\times$ lss[unWISE]', 'twopt_TSMICA_gunwise')
twopt_maskonly(outmap_T143_gunwise_mask, Noise_T143_gunwise, r'Power Spectrum of T[143GHz] $\times$ lss[unWISE]', 'twopt_T143_gunwise')


gauss_unmasked = hp.anafast(outmap_Tgauss_ggauss)
gauss_masked = hp.anafast(outmap_Tgauss_ggauss_mask)

print('unmasked: %.2f' % np.median(Noise_Tnoise_gunwise[101:501] / gauss_unmasked[101:501]))
print('masked: %.2f' % np.median(Noise_Tnoise_gunwise[101:501] / (gauss_masked[101:501]/fsky)))

plt.figure()
plt.loglog(gauss_unmasked, label='Unmasked')
plt.loglog(gauss_masked/fsky, label='Masked')
plt.loglog(Noise_Tnoise_gunwise, label='Theory')
plt.legend(title='Gauss-Gauss Case')
plt.savefig(outdir+'masked_vs_unmasked-gaussgauss')


_,_,smicaharmonic = combine_harmonic(Tmap_SMICA,lssmap_gauss,ClTT_SMICA,Noise_TSMICA_gunwise,mask_tag=True)

smica_const = hp.anafast(outmap_TSMICA_ggauss_mask*mask_map)
smica_harmonic = hp.anafast(smicaharmonic*mask_map)


plt.figure()
plt.loglog(smica_const, label='Const')
plt.loglog(smica_harmonic, label='Harmonic')
plt.loglog(Noise_TSMICA_gunwise*fsky, label='Theory')
plt.legend(title='SMICA Case')
plt.savefig(outdir+'harmonic_vs_const-smica')



Tmap_SMICA_gaussified = hp.synfast(hp.anafast(Tmap_SMICA), 2048)

_,_,outmap_TSMICAgaussified_ggauss_mask = combine(Tmap_SMICA_gaussified,lssmap_gauss,ClTT_SMICA,Noise_TSMICA_gunwise,mask_tag=True)

twopt_maskonly(outmap_TSMICAgaussified_ggauss_mask, Noise_TSMICA_gunwise, r'Power Spectrum of gauss(T[SMICA]) $\times$ lss[gauss]', 'twopt_TSMICAgaussified_ggauss')




_, _, outmap_TSMICA_ggauss = combine(Tmap_SMICA, lssmap_gauss, ClTT_SMICA, Noise_TSMICA_gunwise, mask_tag=False)
_, _, outmap_TSMICA_ggauss_mask = combine(Tmap_SMICA, lssmap_gauss, ClTT_SMICA, Noise_TSMICA_gunwise, mask_tag=True)

twopt(outmap_TSMICA_ggauss, Noise_TSMICA_gunwise, r'Power Spectrum of T[SMICA] $\times$ lss[gauss]', 'twopt_TSMICA_ggauss_nomask')
twopt_maskonly(outmap_TSMICA_ggauss, Noise_TSMICA_gunwise, r'Power Spectrum of Post-Reconstruction Masked\nT[SMICA] $\times$ lss[gauss]', 'twopt_TSMICA_ggauss_postmask')
twopt_maskonly(outmap_TSMICA_ggauss_mask, Noise_TSMICA_gunwise, r'Power Spectrum of Pre-Reconstruction Masked\nT[SMICA] $\times$ lss[gauss]', 'twopt_TSMICA_ggauss_premask')


_, _, outmap_T143_ggauss = combine(Tmap_143, lssmap_gauss, ClTT_143, Noise_T143_gunwise, mask_tag=False)
_, _, outmap_T143_ggauss_mask = combine(Tmap_143, lssmap_gauss, ClTT_143, Noise_T143_gunwise, mask_tag=True)

twopt(outmap_T143_ggauss, Noise_T143_gunwise, r'Power Spectrum of T[143] $\times$ lss[gauss]', 'twopt_T143_ggauss_nomask')
twopt_maskonly(outmap_T143_ggauss, Noise_T143_gunwise, r'Power Spectrum of Post-Reconstruction Masked\nT[143] $\times$ lss[gauss]', 'twopt_T143_ggauss_postmask')
twopt_maskonly(outmap_T143_ggauss_mask, Noise_T143_gunwise, r'Power Spectrum of Pre-Reconstruction Masked\nT[143] $\times$ lss[gauss]', 'twopt_T143_ggauss_premask')



_, _, outmap_T143_gunwise = combine(Tmap_143, lssmap_unwise, ClTT_143, Noise_T143_gunwise, mask_tag=False)
_, _, outmap_T143_gunwise_mask = combine(Tmap_143, lssmap_unwise, ClTT_143, Noise_T143_gunwise, mask_tag=True)

twopt(outmap_T143_gunwise, Noise_T143_gunwise, r'Power Spectrum of T[143] $\times$ lss[unWISE]', 'twopt_T143_gunwise_nomask')
twopt_maskonly(outmap_T143_gunwise, Noise_T143_gunwise, r'Power Spectrum of Post-Reconstruction Masked\nT[143] $\times$ lss[unWISE]', 'twopt_T143_gunwise_postmask')
twopt_maskonly(outmap_T143_gunwise_mask, Noise_T143_gunwise, r'Power Spectrum of Pre-Reconstruction Masked\nT[143] $\times$ lss[unWISE]', 'twopt_T143_gunwise_premask')


#Then do different frequency maps and plot expected signal on 2pt plots

gal20 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL020'],n2r=True)
gal40 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL040'],n2r=True)
gal60 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL060'],n2r=True)
gal70 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL070'],n2r=True)
gal80 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL080'],n2r=True)
gal90 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL090'],n2r=True)
gal97 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL097'],n2r=True)

fsky20 = np.where(gal20*mask_map!=0)[0].size/gal20.size
fsky40 = np.where(gal40*mask_map!=0)[0].size/gal40.size
fsky60 = np.where(gal60*mask_map!=0)[0].size/gal60.size
fsky70 = np.where(gal70!=0)[0].size/gal70.size
fsky80 = np.where(gal80!=0)[0].size/gal80.size
fsky90 = np.where(gal90!=0)[0].size/gal90.size
fsky97 = np.where(gal97!=0)[0].size/gal97.size


clsmica = hp.anafast(outmap_TSMICA_gunwise)
clsmica97 = hp.anafast(outmap_TSMICA_gunwise*gal97)
clsmica90 = hp.anafast(outmap_TSMICA_gunwise*gal90)
clsmica80 = hp.anafast(outmap_TSMICA_gunwise*gal80)
clsmica70 = hp.anafast(outmap_TSMICA_gunwise*gal70)
clsmica70unwise = hp.anafast(outmap_TSMICA_gunwise*mask_map)

plt.figure()
plt.loglog(clsmica,label='unmasked')
plt.loglog(clsmica97/fsky97,label='fsky=0.97 galcut')
plt.loglog(clsmica90/fsky90,label='fsky=0.90 galcut')
plt.loglog(clsmica80/fsky80,label='fsky=0.80 galcut')
plt.loglog(clsmica70/fsky70,label='fsky=0.70 galcut')
plt.loglog(clsmica70unwise/fsky,label='unWISE mask (70 galcut)')
plt.loglog(Noise_TSMICA_gunwise,label='Theory')
plt.legend()
plt.savefig(outdir+'SMICA')

cl143 = hp.anafast(outmap_T143_ggauss)
cl14397 = hp.anafast(outmap_T143_ggauss*gal97)
cl14390 = hp.anafast(outmap_T143_ggauss*gal90)
cl14380 = hp.anafast(outmap_T143_ggauss*gal80)
cl14370 = hp.anafast(outmap_T143_ggauss*gal70)
cl14370unwise = hp.anafast(outmap_T143_ggauss*mask_map)
cl14360unwise = hp.anafast(outmap_T143_ggauss*gal60*mask_map)
cl14340unwise = hp.anafast(outmap_T143_ggauss*gal40*mask_map)
cl14320unwise = hp.anafast(outmap_T143_ggauss*gal20*mask_map)

plt.figure()
plt.loglog(cl143,label='unmasked')
plt.loglog(cl14397/fsky97,label='fsky=0.97 galcut')
plt.loglog(cl14390/fsky90,label='fsky=0.90 galcut')
plt.loglog(cl14380/fsky80,label='fsky=0.80 galcut')
plt.loglog(cl14370/fsky70,label='fsky=0.70 galcut')
plt.loglog(cl14370unwise/fsky,label='unWISE mask (70 galcut)')
plt.loglog(cl14360unwise/fsky60,label='unWISE mask (60 galcut)')
plt.loglog(cl14340unwise/fsky40,label='unWISE mask (40 galcut)')
plt.loglog(cl14320unwise/fsky20,label='unWISE mask (20 galcut)')
plt.loglog(Noise_TSMICA_gunwise,label='Theory')
plt.legend()
plt.savefig(outdir+'T143')

'''




Tmap_SMICA = hp.reorder(fits.open('data/planck_data_testing/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True) / 2.725e6



debeamed_SMICA = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_SMICA), 1/beam),2048)
ClTT = hp.anafast(debeamed_SMICA)
filtered_debeamed_SMICA = hp.alm2map(hp.almxfl(hp.map2alm(debeamed_SMICA), 1/ClTT),2048)
masked_filtered_debeamed_SMICA = filtered_debeamed_SMICA * mask_map
ClTT_mfdSMICA = hp.anafast(masked_filtered_debeamed_SMICA)


plt.figure()
plt.loglog(ClTT,label='SMICA')
for i in np.linspace(0.000001,0.0001,2):
    debeamed_SMICA_noise = hp.alm2map(hp.almxfl(hp.map2alm(Tmap_SMICA+(Tmap_gauss*i)), 1/beam),2048)
    ClTT_noise = hp.anafast(debeamed_SMICA_noise)
    plt.loglog(ClTT_noise,label='SMICA+%.1e noise'%i)

plt.legend()
plt.xlim([1,2500])
plt.savefig(outdir+'a')








'''
# def histplot_smoothed(bins, n_Tg, normprod_Tg, Tmap, gmap, noise, plottitle, filename, tolerance=0.01):
#     # Tolerance is the % of the max y value that defines the x axis limits so we can zoom on the interesting part.
#     #x,y,z=plt.hist(vrmap,bins=bins,histtype='step')
#     plt.close()
#     smoothing_factor = {'T[noise]' : 9.5, 'T[gauss]' : 11, 'T[SMICA]' : 41, 'T[143GHz]' : 150}
#     for case in smoothing_factor:
#         if case.lower() in plottitle.lower():
#             smoothing_factor = smoothing_factor[case]
#             break
#     if 'mask' in plottitle.lower():
#         equiv_outmap = Tmap * gmap * mask_map  * sum(noise[1:] / ((np.arange(1,6144)*(np.arange(1,6144)+1))/4/np.pi))
#         equiv_outmap_smoothed = hp.smoothing(equiv_outmap, fwhm=hp.nside2resol(2048)*smoothing_factor)
#         n_smoothed, _, _ = plt.hist(equiv_outmap_smoothed[np.where(mask_map!=0)], bins=bins)
#     else:
#         equiv_outmap = Tmap * gmap * sum(noise[1:] / ((np.arange(1,6144)*(np.arange(1,6144)+1))/4/np.pi))
#         equiv_outmap_smoothed = hp.smoothing(equiv_outmap, fwhm=hp.nside2resol(2048)*11)
#         n_smoothed, _, _ = plt.hist(equiv_outmap_smoothed, bins=bins)
#     plt.figure()
#     if 'mask' in plottitle.lower():
#         plt.plot(bin_centres(bins), pixel_scaling_masked(n_Tg), label='Reconstruction')
#         plt.plot(bin_centres(bins), pixel_scaling_masked(normprod_Tg), ls='--', label='Normal Product')       
#         plt.plot(bin_centres(bins), pixel_scaling_masked(n_smoothed), ls=':', label='Normal Product\nSmoothed:%.1f'%smoothing_factor)
#     else:
#         plt.plot(bin_centres(bins), pixel_scaling(n_Tg), label='Reconstruction')
#         plt.plot(bin_centres(bins), pixel_scaling(normprod_Tg), ls='--', label='Normal Product')
#         plt.plot(bin_centres(bins), pixel_scaling(n_smoothed), ls=':', label='Normal Product\nSmoothed:%.1f'%smoothing_factor)
#     plt.title(plottitle)
#     plt.xlabel('v/c')
#     plt.ylabel('# of Pixels')
#     #plt.xlim([-.05,.05])
#     plt.xlim([bin_centres(bins)[np.where(n_smoothed>(tolerance*n_smoothed.max()))][0], bin_centres(bins)[np.where(n_smoothed>(tolerance*n_smoothed.max()))][-1]])
#     plt.legend()
#     plt.savefig(outdir + filename)
#     plt.close('all')


# histplot_smoothed(bins_output_Tgauss_ggauss, n_output_Tgauss_ggauss, normal_product_Tgauss_ggauss, Tmap_gauss_filtered, lssmap_gauss_filtered, Noise_Tnoise_gunwise, r'Pixel Distribution of Output Map (T[gauss] $\times$ lss[gauss])', 'analysis_smoothing_Tgauss_ggauss')
# histplot_smoothed(bins_output_Tgauss_ggauss_mask, n_output_Tgauss_ggauss_mask, normal_product_Tgauss_ggauss_mask, Tmap_gauss_filtered_mask, lssmap_gauss_filtered_mask, Noise_Tnoise_gunwise, r'Pixel Distribution of Masked Output Map (T[gauss] $\times$ lss[gauss])', 'analysis_smoothing_Tgauss_ggauss_mask')

# histplot_smoothed(bins_output_Tnoise_ggauss, n_output_Tnoise_ggauss, normal_product_Tnoise_ggauss, Tmap_noise_filtered, lssmap_gauss_filtered, Noise_Tnoise_gunwise, r'Pixel Distribution of Output Map (T[noise] $\times$ lss[gauss])', 'analysis_smoothing_Tnoise_ggauss')
# histplot_smoothed(bins_output_Tnoise_ggauss_mask, n_output_Tnoise_ggauss_mask, normal_product_Tnoise_ggauss_mask, Tmap_noise_filtered_mask, lssmap_gauss_filtered_mask, Noise_Tnoise_gunwise, r'Pixel Distribution of Masked Output Map (T[noise] $\times$ lss[gauss])', 'analysis_smoothing_Tnoise_ggauss_mask')

# histplot_smoothed(bins_output_TSMICA_ggauss_mask, n_output_TSMICA_ggauss_mask, normal_product_TSMICA_ggauss_mask, Tmap_SMICA_filtered_mask, lssmap_gauss_filtered_mask, Noise_TSMICA_gunwise, r'Pixel Distribution of Masked Output Map (T[SMICA] $\times$ lss[gauss])', 'analysis_smoothing_TSMICA_ggauss_mask')
# histplot_smoothed(bins_output_Tgauss_gunwise_mask, n_output_Tgauss_gunwise_mask, normal_product_Tgauss_gunwise_mask, Tmap_gauss_filtered_mask, lssmap_unwise_filtered_mask, Noise_Tnoise_gunwise, r'Pixel Distribution of Masked Output Map (T[gauss] $\times$ lss[unWISE])', 'analysis_smoothing_Tgauss_gunwise_mask')
# histplot_smoothed(bins_output_T143_gunwise_mask, n_output_T143_gunwise_mask, normal_product_T143_gunwise_mask, Tmap_143_filtered_mask, lssmap_unwise_filtered_mask, Noise_T143_gunwise, r'Pixel Distribution of Masked Output Map (T[143GHz] $\times$ lss[unWISE])', 'analysis_smoothing_T143_gunwise_mask')


# smoothing_scales = [0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# smoothed_outmaps = {smoothing_scales[i] : hp.smoothing(Tmap_gauss_filtered*2.529e-6*lssmap_gauss_filtered,fwhm=hp.nside2resol(2048)*smoothing_scales[i]) for i in range(len(smoothing_scales))}

# smoothed_Tg = {smoothing_scales[i] : plt.hist(smoothed_outmaps[x], bins=bins_output_Tgauss_ggauss) for i,x in enumerate(smoothing_scales)}


# plt.figure()
# plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(n_output_Tgauss_ggauss), label='Map Pixels')
# plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(normal_product_Tgauss_ggauss), ls='--', label='Normal Product Distribution')
# for i in smoothing_scales[::3]:
#     plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(smoothed_Tg[i][0]),label='Smoothing %.1fpx'%i)
# #plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(smoothed_Tg[11][0]),label='Smoothing %.1fpx'%11)

# plt.xlim([bin_centres(bins_output_Tgauss_ggauss)[np.where(normal_product_Tgauss_ggauss>(0.01*normal_product_Tgauss_ggauss.max()))][0], bin_centres(bins)[np.where(normal_product_Tgauss_ggauss>(0.01*normal_product_Tgauss_ggauss.max()))][-1]])
# plt.legend()
# plt.savefig(outdir+'analysis_smoothing_examples')




# noisemap = hp.synfast(Noise_Tnoise_gunwise, 2048)
# n_noise, b_noise, _ = plt.hist(noisemap, bins=bins_output_Tgauss_ggauss)

# hn_noise = n_noise / simps(n_noise)
# hn_outmap = normal_product_Tgauss_ggauss / simps(normal_product_Tgauss_ggauss)

# if sum(hn_noise) != 1.:
#     hn_noise[np.where(hn_noise==hn_noise.max())] += (1-sum(hn_noise))

# if sum(hn_outmap) != 1.:
#     hn_outmap[np.where(hn_outmap==hn_outmap.max())] += (1-sum(hn_outmap))

# noise_sample = np.random.choice(bin_centres(bins_output_Tgauss_ggauss), size=12*2048**2, p=hn_noise)
# outmap_sample = np.random.choice(bin_centres(bins_output_Tgauss_ggauss), size=12*2048**2, p=hn_outmap)

# n_noise_outmap, _, _ = plt.hist(noise_sample*outmap_sample, bins=bins_output_Tgauss_ggauss)
# n_noise_samp, _, _ = plt.hist(noise_sample, bins=bins_output_Tgauss_ggauss)
# n_outmap_samp, _, _ = plt.hist(outmap_sample, bins=bins_output_Tgauss_ggauss)

# plt.figure()
# plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(n_noise_samp), label='Noise')
# plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(n_outmap_samp), label='Pre-noise Reconstruction')
# plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(n_noise_outmap), label='Combination')
# plt.plot(bin_centres(bins_output_Tgauss_ggauss), pixel_scaling(n_output_Tgauss_ggauss), label='Post-noise Reconstruction')
# plt.legend()
# plt.savefig(outdir+'show')


'''
def Poisson(lamb, xpoints):
    distro = np.zeros(xpoints.size)
    for xid in np.arange(xpoints.size):
        distro[xid] = ((lamb**xpoints[xid]) * np.exp(-lamb))  /  factorial(xpoints[xid])
    return distro

plt.close('all')


lssbins = 50
smudge = 6.5
n_unwise, b_unwise, _ = plt.hist(lssmap_unwise, bins=lssbins)
n_ggauss_filtered_mask_lowbin, bins_ggauss_filtered_mask_lowbin, _ = plt.hist(lssmap_unwise_filtered_mask[np.where(mask_map!=0)],bins=lssbins)
lss_poisson_sample = Poisson(np.mean(lssmap_unwise), np.arange(b_unwise.size))
lss_filtered_poisson_sample = Poisson(np.mean(lssmap_unwise)*smudge, np.arange(bins_ggauss_filtered_mask_lowbin.size))

plt.figure()
plt.bar(bin_centres(np.arange(b_unwise.size)), n_unwise, align='center', label='unWISE Map', alpha=0.75)
plt.plot(np.arange(b_unwise.size), pixel_scaling(lss_poisson_sample), label='P(%.2f,bin#)'%np.mean(lssmap_unwise))
plt.bar(bin_centres(np.arange(bins_ggauss_filtered_mask_lowbin.size)), n_ggauss_filtered_mask_lowbin, align='center', label='Filtered unWISE Map', alpha=0.75)
plt.plot(np.arange(bins_ggauss_filtered_mask_lowbin.size), pixel_scaling_masked(lss_filtered_poisson_sample), label='P(%.2f,bin#)'%(np.mean(lssmap_unwise)*smudge))
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,3,1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
plt.title('unWISE map distributions')
plt.xlabel('Bin #')
plt.ylabel('# of Pixels')
plt.savefig(outdir+'analysis_lssfilter_nonpoisson')

Tmap_gauss_filtered_mask, lssmap_gauss_filtered_mask, outmap_Tgauss_ggauss_mask = combine(Tmap_gauss, lssmap_gauss, ClTT_noise, Noise_Tnoise_gunwise, mask_tag=True)
n_Tgauss_filtered_mask, bins_Tgauss_filtered_mask, _ = plt.hist(Tmap_gauss_filtered_mask[np.where(mask_map!=0)], bins=binning)

hist_norm_ggauss_filtered_mask = n_ggauss_filtered_mask_lowbin/simps(n_ggauss_filtered_mask_lowbin)
hist_norm_Tgauss_filtered_mask = n_Tgauss_filtered_mask/simps(n_Tgauss_filtered_mask)

if sum(hist_norm_ggauss_filtered_mask) != 1.:
    hist_norm_ggauss_filtered_mask[np.where(hist_norm_ggauss_filtered_mask==hist_norm_ggauss_filtered_mask.max())] += (1-sum(hist_norm_ggauss_filtered_mask))

if sum(hist_norm_Tgauss_filtered_mask) != 1.:
    hist_norm_Tgauss_filtered_mask[np.where(hist_norm_Tgauss_filtered_mask==hist_norm_Tgauss_filtered_mask.max())] += (1-sum(hist_norm_Tgauss_filtered_mask))

lss_filtered_sample = np.random.choice(bin_centres(np.arange(bins_ggauss_filtered_mask_lowbin.size)), size=np.where(mask_map!=0)[0].size, p=hist_norm_ggauss_filtered_mask)
T_filtered_sample = np.random.choice(bin_centres(np.arange(bins_Tgauss_filtered_mask.size)), size=np.where(mask_map!=0)[0].size, p=hist_norm_Tgauss_filtered_mask)



t,z,y = plt.hist(lss_filtered_sample,bins=lssbins)
tt,zz,yy = plt.hist(T_filtered_sample, bins=binning)

plt.figure()
plt.bar(bin_centres(np.arange(bins_ggauss_filtered_mask_lowbin.size)), n_ggauss_filtered_mask_lowbin, align='center', label='Filtered unWISE Map', alpha=0.75)
plt.bar(bin_centres(z), t, align='center',label='random selection', alpha=0.75)
plt.legend()
plt.xlabel('Bin #')
plt.ylabel('# of pixels')
plt.title('Filtered lss map and random draws')
plt.savefig(outdir+'analysis_ggaussmatch')

plt.figure()
plt.bar(bin_centres(np.arange(bins_Tgauss_filtered_mask.size)), n_Tgauss_filtered_mask, align='center', label='Filtered T Map', alpha=0.75)
plt.bar(bin_centres(zz), tt, align='center', label='random selection', alpha=0.75)
plt.legend()
plt.xlabel('Bin #')
plt.ylabel('# of pixels')
plt.title('Filtered T map and random draws')
plt.savefig(outdir+'analysis_Tgaussmatch')

#n_combo, bins_combo, _ = plt.hist(T_filtered_sample*lss_filtered_sample, bins=binning)

plt.figure()
plt.hist(T_filtered_sample*lss_filtered_sample, bins=50)
plt.xlabel('lss bin #   x   T bin #')
plt.ylabel('# of pixels')
plt.title('Product of filtered T and lss pixel distributions')
plt.savefig(outdir+'analysis_combo')

'''


'''


lssbins = 50

n_gunwise_filtered_mask_lowbin, bins_gunwise_filtered_mask_lowbin, _ = plt.hist(lssmap_unwise_filtered_mask[np.where(mask_map!=0)], bins=lssbins)

hist_norm_gunwise_filtered_mask = n_gunwise_filtered_mask_lowbin/simps(n_gunwise_filtered_mask_lowbin)
hist_norm_T143_filtered_mask = n_T143_filtered_mask/simps(n_T143_filtered_mask)


if sum(hist_norm_gunwise_filtered_mask) != 1.:
    hist_norm_gunwise_filtered_mask[np.where(hist_norm_gunwise_filtered_mask==hist_norm_gunwise_filtered_mask.max())] += (1-sum(hist_norm_gunwise_filtered_mask))

if sum(hist_norm_T143_filtered_mask) != 1.:
    hist_norm_T143_filtered_mask[np.where(hist_norm_T143_filtered_mask==hist_norm_T143_filtered_mask.max())] += (1-sum(hist_norm_T143_filtered_mask))


lss_filtered_sample = np.random.choice(bin_centres(bins_gunwise_filtered_mask_lowbin), size=np.where(mask_map!=0)[0].size, p=hist_norm_gunwise_filtered_mask)
T_filtered_sample = np.random.choice(bin_centres(bins_T143_filtered_mask), size=np.where(mask_map!=0)[0].size, p=hist_norm_T143_filtered_mask)

t,z,y = plt.hist(lss_filtered_sample,bins=lssbins)
tt,zz,yy = plt.hist(T_filtered_sample, bins=binning)

plt.figure()
plt.plot(bin_centres(bins_gunwise_filtered_mask_lowbin), n_gunwise_filtered_mask_lowbin, label='Filtered unWISE Map', alpha=0.75)
plt.plot(bin_centres(z), t, label='random selection', alpha=0.75)
plt.legend()
plt.xlabel('Bin #')
plt.ylabel('# of pixels')
plt.title('Filtered lss map and random draws')
plt.savefig(outdir+'analysis_gunwisematch')

plt.figure()
plt.plot(bin_centres(bins_T143_filtered_mask), n_T143_filtered_mask, label='Filtered T Map', alpha=0.75)
plt.plot(bin_centres(zz), tt, label='random selection', alpha=0.75)
plt.legend()
plt.xlabel('Bin #')
plt.ylabel('# of pixels')
plt.title('Filtered T map and random draws')
plt.savefig(outdir+'analysis_T143match')


plt.figure()
n_comboreal, _, _ = plt.hist(T_filtered_sample*lss_filtered_sample, bins=bins_output_T143_gunwise_mask)
plt.xlabel('v/c')
plt.ylabel('# of pixels')
plt.title('Product of filtered T and lss pixel distributions')
plt.xlim([bin_centres(bins_output_T143_gunwise_mask)[np.where(n_comboreal>(0.01*n_comboreal.max()))][0], bin_centres(bins_output_T143_gunwise_mask)[np.where(n_comboreal>(0.01*n_comboreal.max()))][-1]])
plt.savefig(outdir+'analysis_combo_real')


plt.figure()
n_estimreal, _, _ = plt.hist(outmap_T143_gunwise_mask[np.where(mask_map!=0)], bins=bins_output_T143_gunwise_mask)
plt.xlabel('v/c')
plt.ylabel('# of pixels')
plt.title('Reconstruction of T[143]    x   lss[unWISE]')
plt.xlim([bin_centres(bins_output_T143_gunwise_mask)[np.where(n_estimreal>(0.01*n_estimreal.max()))][0], bin_centres(bins_output_T143_gunwise_mask)[np.where(n_estimreal>(0.01*n_estimreal.max()))][-1]])
plt.savefig(outdir+'analysis_combo_estim_out')

planck_galcut = hp.reorder(fits.open('/home/richard/Desktop/ReCCO/code/data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL070'], n2r=True)
planck_galcut_grown = hp.smoothing(planck_galcut, fwhm=hp.nside2resol(2048)*50)
planck_galcut_grown[np.where(planck_galcut_grown<.95)]=0

mask_map = np.load('data/mask_unWISE_thres_v10.npy')

plt.figure()
hp.mollview(mask_map*outmap_T143_gunwise_mask)
plt.savefig(outdir+'before')
plt.figure()
mask_map[np.where(planck_galcut_grown==0)] = 0
hp.mollview(mask_map*outmap_T143_gunwise_mask)
plt.savefig(outdir+'after')


x=hp.smoothing(np.load('data/mask_unWISE_thres_v10.npy'), fwhm=hp.nside2resol(2048)*.1)
x[np.where(x<.95)]=0
plt.figure()
hp.mollview(x)
plt.savefig(outdir+'99')

twopt_real = hp.anafast(outmap_T143_gunwise_mask*mask_map)


xstart = 100

plt.figure()
plt.plot(np.arange(6144)[xstart:],(Noise_T143_gunwise * fsky)[xstart:], label='Theory')
plt.plot(np.arange(6144)[xstart:],twopt_real[xstart:], label='Reconstruction')
plt.scatter(ells[np.where(ells>=xstart)], (Noise_T143_gunwise*fsky)[ells[np.where(ells>=xstart)]],s=20,marker='x',c='k')
plt.xlabel(r'$\ell$')
plt.ylabel('Noise')
plt.title('Theory vs Reconstruction Noise')
plt.legend()
plt.xscale('log')
plt.savefig(outdir+'analysis_2pt_vs_theory')


twopt_gauss = hp.anafast(outmap_Tgauss_ggauss)
twopt_gauss_mask = hp.anafast(outmap_Tgauss_ggauss_mask*mask_map)

plt.figure()
plt.plot(np.arange(6144),[Noise_Tnoise_gunwise[5] for x in np.arange(6144)], label='Theory')
plt.plot(np.arange(6144),twopt_gauss, label='Reconstruction')
plt.xlabel(r'$\ell$')
plt.ylabel('Noise')
plt.title('Theory vs Reconstruction Noise')
plt.legend()
plt.xscale('log')
plt.savefig(outdir+'analysis_2pt_vs_theory_gauss')

plt.figure()
plt.plot(np.arange(6144),[Noise_Tnoise_gunwise[5] * fsky for x in np.arange(6144)], label='Theory')
plt.plot(np.arange(6144),twopt_gauss_mask, label='Reconstruction')
plt.xlabel(r'$\ell$')
plt.ylabel('Noise')
plt.title('Theory vs Reconstruction Noise')
plt.legend()
plt.xscale('log')
plt.savefig(outdir+'analysis_2pt_vs_theory_gauss_mask')


## MAKE X AXIS BINS AGREE FOR 1pt
#   > They agree. Now prediction of shape, actual shape, and normal product distribution are all in agreement.
## IT THE MASK OR THE DATA PRODUCT RUINING THE LOW ELL SHIT?
###USING THE SAME MASK, COMPARE THIS 2PT TO THE GAUSSIAN MOCKS. IF THEY HAVE THE SAME FEATURE ITS A MASK FEATURE AND NOT AN INTRINSIC DATA PROPERTY
## TRY USING A GALACTIC CUT ONLY, WILL THIS HELP?
## GROW PLANCK GALAXY CUT AND LEAVE SMALL HOLES ALONE

## 1pt random gaussian full sky. 1pt with Planck mask+grown. 1pt with full mask.
## 1pt data. 1pt with Planckgrowth+rest of mask






# xvals = bin_centres(np.arange(bins_Tgauss.size))
# xmean = xvals[np.where(n_Tgauss==n_Tgauss.max())][0]
# xstdmin = xvals[np.where(n_Tgauss>=np.std(n_Tgauss))][0]
# xstdmax = xvals[np.where(n_Tgauss>=np.std(n_Tgauss))][-1]

# gauss_samples = np.random.normal(loc=np.arange(n_Tgauss.size)[np.where(n_Tgauss==n_Tgauss.max())][0], scale=(xmean-xstdmin)/np.sqrt(2), size=12*2048**2)

# plt.figure()
# plt.plot(bin_centres(np.arange(bins_Tgauss.size)), n_Tgauss, label='Gaussian T Map', alpha=0.75)
# y1, y2 = plt.ylim()
# plt.plot([xstdmin]*2, [0,1e10], ls=':', c='black', alpha=.75, label=r'1$\sigma$')
# plt.plot([xmean]*2, [0,1e10], ls='--', c='black', alpha=.75)
# plt.plot([xstdmax]*2, [0,1e10], ls=':', c='black', alpha=.75)
# plt.ylim([y1,y2])
# plt.xlabel('Bin #')
# plt.ylabel('# of pixels')
# plt.title('Distribution of Gaussian Tmap')
# plt.legend()
# plt.savefig(outdir+'analysis_distribution_Tgauss')

# plt.figure()
# plt.hist(gauss_samples,bins=10000, label='Sampled Distribution')
# plt.plot(bin_centres(np.arange(bins_Tgauss.size)), n_Tgauss, label='Original Distribution')
# plt.xlabel('Bin #')
# plt.ylabel('# of pixels')
# plt.title('Sampled/Original T Distribution')
# plt.legend()
# plt.savefig(outdir+'analysis_distribution_Tgauss_sampled')
'''