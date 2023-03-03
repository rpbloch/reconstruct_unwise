'''
# ### Would be nice to do multiple realizations to see the cosmic variance error bars
# ### Might be too big a project, but we could consider what happens if we inpaint without destroying correlations
# ### As masking grows what is effect on reconstruction
'''

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
from math import lgamma
import pickle

class Maps(object):
    def get_alms(self, tag, gauss=False):
        if 'unWISE' in tag:
            beam = np.ones(6144)
            fsky = 1.
            mask = np.ones(12*2048**2)
        else:
            beam = 1 / self.beams[tag.split('-')[0]]
            fsky = self.fsky
            mask = self.mask_map
        if gauss:
            # Important order of operations. For Gaussian T maps, the power spectrum of the full sky and cut sky are significantly different.
            # Therefore to best match the input T map spectrum to the theory T spectrum the following order is required:
            # 1) Spectrum used must be (Tmap * mask) / fsky   to represent the full sky power of T
            # 2) Generate Gaussian realization
            # 3) Remask. The power spectrum matches slightly better if we don't remask, but we are using masked T inputs and the mask doesn't
            #            survive the Gaussification of the power spectrum. For this reason we also want a separate cache of Gaussian Tmap Cls
            # 4) Debeam
            return hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(self.input_maps[tag] * mask) / fsky, 2048) * mask), beam)
        else:
            return hp.almxfl(hp.map2alm(self.input_maps[tag] * mask), beam)
    def alm2map(self, alms, tag):
        return hp.alm2map(alms[tag], 2048)
    def get_Cls(self, alms, tag):
        return hp.alm2cl(alms[tag.split('-')[0]]) / self.fsky  # For CMB subtracted maps we only want to reconstruct the subtracted map, we want the full frequency sky for the theory Cls
    def __init__(self, allmaps=True, fullsky=False):
        print('Loading input maps')
        if fullsky:
            self.mask_map = np.ones(12*2048**2)
        else:
            self.mask_map = np.load('data/mask_unWISE_thres_v10.npy')
        self.fsky = np.where(self.mask_map!=0)[0].size / self.mask_map.size
        if allmaps:
            self.map_tags = ('SMICA', '100GHz', '143GHz', '217GHz', '100GHz-SMICA', '143GHz-SMICA', '217GHz-SMICA', 'unWISE')
        else:
            self.map_tags = ('SMICA', 'unWISE')
        self.input_maps = {'SMICA_input' : hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True),
                           '100GHz' : hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) if '100GHz' in self.map_tags else None,
                           '143GHz' : hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) if '143GHz' in self.map_tags else None,
                           '217GHz' : hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True) if '217GHz' in self.map_tags else None,
                           '100GHz-SMICA' : fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten() if '100GHz-SMICA' in self.map_tags else None,
                           '143GHz-SMICA' : fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits')[1].data['INTENSITY'].flatten() if '143GHz-SMICA' in self.map_tags else None,
                           '217GHz-SMICA' : fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits')[1].data['INTENSITY'].flatten() if '217GHz-SMICA' in self.map_tags else None,
                           'unWISE_input' : fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten() }
        for tag in list(self.input_maps.keys()):  # Clean tags of unloaded maps
            if self.input_maps[tag] is None:
                del self.input_maps[tag]
        self.beams = { 'SMICA' : hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143),
                       '100GHz' : hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6143),
                       '143GHz' : hp.gauss_beam(fwhm=np.radians(7.27/60), lmax=6143),
                       '217GHz' : hp.gauss_beam(fwhm=np.radians(5.01/60), lmax=6143) }
        self.beams['100GHz'][4001:] = self.beams['100GHz'][4000]  # Numerical trouble for healpix for beams that blow up at large ell. Sufficient to flatten out at high ell above where the 1/TT filter peaks
        print('Adding Gaussian noise to CMB maps')
        self.input_maps['SMICA'] = self.input_maps['SMICA_input'] + hp.synfast(np.repeat([hp.anafast(self.input_maps['SMICA_input'], lmax=2500)[-1]], 6144), 2048)
        print('Inpainting unWISE map in masked regions')
        self.input_maps['unWISE'] = self.input_maps['unWISE_input'].copy()
        if not fullsky:
            ngal_per_pix = self.input_maps['unWISE_input'][np.where(self.mask_map!=0)].sum() / np.where(self.mask_map!=0)[0].size
            ngal_fill = np.random.poisson(lam=ngal_per_pix, size=np.where(self.mask_map==0)[0].size)
            self.input_maps['unWISE'][np.where(self.mask_map==0)] = ngal_fill
        print('Masking and debeaming maps')  # Exclude unWISE from actual map processing
        self.map_alms = { tag : self.get_alms(tag) for tag in self.map_tags[:-1]}
        self.maps = { tag : self.alm2map(self.map_alms, tag) for tag in self.map_tags[:-1]}
        self.Cls = { tag : self.get_Cls(self.map_alms, tag) for tag in self.map_tags[:-1]}
        self.maps['unWISE'] = self.input_maps['unWISE'].copy()
        self.Cls['unWISE'] = hp.anafast(self.maps['unWISE'])
        print('Generating Gaussian realizations of maps')
        self.gaussian_alms = { tag : self.get_alms(tag, gauss=True) for tag in self.map_tags}
        self.gaussian_maps = { tag : self.alm2map(self.gaussian_alms, tag) for tag in self.map_tags}
        self.gaussian_Cls = { tag : self.get_Cls(self.gaussian_alms, tag) for tag in self.map_tags}


class Cosmology(object):
    def ne0(self):
        G_SI = 6.674e-11
        mProton_SI = 1.673e-27
        H100_SI = 3.241e-18
        chi = 0.86
        me = 1.14
        gasfrac = 0.9
        omgh2 = gasfrac* conf.ombh2
        return chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me  # Average electron density today in 1/m^3
    def fe(self, z):
        a = 0.475
        b = 0.703
        c = 3.19
        z0 = 5.42
        return a*(z+b)**0.02 * (1-np.tanh(c*(z-z0)))
    def bias_e2(self, z, k):  # Supposed b^2 = P_ee/P_mm(dmo) bias given in eq. 20, 21. # Verified that b^2 / fe(z)**.5 matches ReCCO em exactly (and squared matches ee exactly).
        bstar2 = lambda z : 0.971 - 0.013*z
        gamma = lambda z : 1.91 - 0.59*z + 0.10*z**2
        kstar = lambda z : 4.36 - 3.24*z + 3.10*z**2 - 0.42*z**3
        bias_squared = np.zeros((z.size, k.size))
        for zid, redshift in enumerate(z):
            bias_squared[zid, :] = bstar2(redshift) / ( 1 + (k/kstar(redshift))**gamma(redshift) ) / self.fe(redshift)**.5
        return bias_squared
    def __init__(self):
        self.zmin = conf.z_min
        self.zmax = conf.z_max
        self.Clmm = None
        self.cambpars = camb.CAMBparams()
        self.cambpars.set_cosmology(H0 = conf.H0, ombh2=conf.ombh2, \
                                                  omch2=conf.omch2, mnu=conf.mnu , \
                                                  omk=conf.Omega_K, tau=conf.tau,  \
                                                  TCMB =2.725 )
        self.cambpars.InitPower.set_params(As =conf.As*1e-9 ,ns=conf.ns, r=0)
        self.cambpars.NonLinear = model.NonLinear_both
        self.cambpars.max_eta_k = 14000.0*conf.ks_hm[-1]
        self.cambpars.set_matter_power(redshifts=conf.zs_hm.tolist(), kmax=conf.ks_hm[-1], k_per_logint=20)
        self.cosmology_data = camb.get_background(self.cambpars)
        self.bin_width = (self.cosmology_data.comoving_radial_distance(self.zmax)-self.cosmology_data.comoving_radial_distance(self.zmin))
    def chi_to_z(self, chi):
        return self.cosmology_data.redshift_at_comoving_radial_distance(chi)
    def z_to_chi(self, z):
        return self.cosmology_data.comoving_radial_distance(z)
    def sample_chis(self, N):
        chi_min = self.z_to_chi(self.zmin)
        chi_max = self.z_to_chi(self.zmax)
        return np.linspace(chi_min, chi_max, N)
    def get_limber_window(self, tag, avg=False, gwindow_zdep=1.2):
        # Return limber window function for observable in units of 1/Mpc
        thompson_SI = 6.6524e-29
        m_per_Mpc = 3.086e22
        chis = self.sample_chis(1000)
        if tag == 'm':
            window = np.repeat(1 / self.bin_width, chis.size)
        elif tag == 'g':
            with open('data/unWISE/blue.txt', 'r') as FILE:
                x = FILE.readlines()
            z = np.array([float(l.split(' ')[0]) for l in x])
            dndz = np.array([float(l.split(' ')[1]) for l in x])
            galaxy_bias = (0.8+gwindow_zdep*self.chi_to_z(chis))  # Changed from 0.8 + 1.2z to better fit inpainted unWISE map spectrum
            window = galaxy_bias * interp1d(z ,dndz, kind= 'linear', bounds_error=False, fill_value=0)(self.chi_to_z(chis)) * self.cosmology_data.h_of_z(self.chi_to_z(chis))
        elif tag == 'taud':
            window = (-thompson_SI * self.ne0() * (1+self.chi_to_z(chis))**2 * m_per_Mpc)
        if avg:  # Returns unitless window
            return simps(window, chis)
        else:    # Returns 1/Mpc window so that integral over Pkk will return unitless Cls
            return window
    def compute_Cls(self, ngbar, gwindow_zdep=1.2):
        chis = self.sample_chis(1000)
        Pmm_full = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True,  hubble_units=False, k_hunit=False, kmax=conf.ks_hm[-1], zmax=conf.zs_hm[-1])
        ells = np.unique(np.append(np.geomspace(1,6143,120).astype(int), 6143))
        matter_window = self.get_limber_window('m',    avg=False, gwindow_zdep=gwindow_zdep)
        galaxy_window = self.get_limber_window('g',    avg=False, gwindow_zdep=gwindow_zdep)
        taud_window   = self.get_limber_window('taud', avg=False, gwindow_zdep=gwindow_zdep)
        self.Clmm = np.zeros(ells.size)
        self.Clgg = np.zeros(ells.size)
        self.Cltaudg = np.zeros(ells.size)
        self.Cltaudtaud = np.zeros(ells.size)
        for l, ell in enumerate(ells):
            Pmm_full_chi = np.diagonal(np.flip(Pmm_full.P(self.chi_to_z(chis), (ell+0.5)/chis[::-1], grid=True), axis=1))
            self.Clmm[l]       = simps(Pmm_full_chi * matter_window**2                                   / chis**2, chis)
            self.Clgg[l]       = simps(Pmm_full_chi *                  galaxy_window**2                  / chis**2, chis)
            self.Cltaudg[l]    = simps(Pmm_full_chi *                  galaxy_window    * taud_window    / chis**2, chis)
            self.Cltaudtaud[l] = simps(Pmm_full_chi *                                     taud_window**2 / chis**2, chis)
        self.Clmm       =  interp1d(ells, self.Clmm,       bounds_error=False, fill_value='extrapolate')(np.arange(6144))
        self.Clgg       = (interp1d(ells, self.Clgg,       bounds_error=False, fill_value='extrapolate')(np.arange(6144))  + 9.2e-8) * ngbar**2
        self.Cltaudg    = (interp1d(ells, self.Cltaudg,    bounds_error=False, fill_value='extrapolate')(np.arange(6144))          ) * ngbar
        self.Cltaudtaud =  interp1d(ells, self.Cltaudtaud, bounds_error=False, fill_value='extrapolate')(np.arange(6144))


class Estimator(object):
    def __init__(self):
        self.reconstructions = {}
        self.Cls = {}
        self.noises = {}
        self.Tmaps_filtered = {}
        self.lssmaps_filtered = {}
    def get_recon_tag(self, Ttag, gtag, Tgauss, ggauss, notes):
        return Ttag + '_gauss=' + str(Tgauss) + '__' + gtag + '_gauss=' + str(ggauss) + '__notes=' + notes
    def wigner_symbol(self, ell, ell_1,ell_2):
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
    def Noise_vr_diag(self, lmax, alpha, gamma, ell, cltt, clgg_binned, cltaudg_binned):
        terms = 0
        for l2 in np.arange(lmax):
            for l1 in np.arange(np.abs(l2-ell),l2+ell+1):
                if l1 > lmax-1 or l1 <2:   #triangle rule
                    continue
                gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell+1)/(4*np.pi))*self.wigner_symbol(ell, l1, l2)*cltaudg_binned[l2]
                term_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg_binned[l2]))
                if np.isfinite(term_entry):
                    terms += term_entry
        return (2*ell+1) / terms
    def combine(self, Tmap, lssmap, mask, ClTT, Clgg, Cltaudg, Noise, convert_K=True):
        dTlm = hp.map2alm(Tmap)
        dlm  = hp.map2alm(lssmap)
        ClTT_filter = ClTT.copy()
        Clgg_filter = Clgg.copy()
        ClTT_filter[:100] = 1e15
        Clgg_filter[:100] = 1e15
        dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
        dlm_zeta = hp.almxfl(dlm, np.divide(Cltaudg, Clgg_filter, out=np.zeros_like(Cltaudg), where=Clgg_filter!=0))
        Tmap_filtered = hp.alm2map(dTlm_xi, 2048) * mask
        lssmap_filtered = hp.alm2map(dlm_zeta, 2048) * mask
        outmap_filtered = Tmap_filtered*lssmap_filtered
        const_noise = np.median(Noise[10:100])
        outmap = outmap_filtered * const_noise * mask
        if convert_K:  # output map has units of K
            outmap /= 2.725
        return Tmap_filtered * const_noise, lssmap_filtered, outmap
    def combine_harmonic(self, Tmap, lssmap, mask, ClTT, Clgg, Cltaudg, Noise, convert_K=True):
        dTlm = hp.map2alm(Tmap)
        dlm  = hp.map2alm(lssmap)
        ClTT_filter = ClTT.copy()
        Clgg_filter = Clgg.copy()
        ClTT_filter[:100] = 1e15
        Clgg_filter[:100] = 1e15
        dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
        dlm_zeta = hp.almxfl(dlm, np.divide(Cltaudg, Clgg_filter, out=np.zeros_like(Cltaudg), where=Clgg_filter!=0))
        Tmap_filtered = hp.alm2map(dTlm_xi, 2048)
        lssmap_filtered = hp.alm2map(dlm_zeta, 2048)
        outmap_filtered = Tmap_filtered*lssmap_filtered
        outmap = hp.alm2map(hp.almxfl(hp.map2alm(outmap_filtered), Noise), 2048) * mask
        if convert_K:  # output map has units of K
            outmap /= 2.725
        return Tmap_filtered, lssmap_filtered, outmap
    def reconstruct(self, Tmap, lssmap, Tcl, gcl, mask, taudgcl, recon_tag, useharmonic=False):
        if useharmonic:
            if recon_tag not in self.noises.keys():
                ells = np.unique(np.append(np.geomspace(1,6143,15).astype(int), 6143))
                noise = np.zeros(ells.size)
                for l, ell in enumerate(ells):
                    print('    computing noise @ ell = ' + str(ell))
                    noise[l] = self.Noise_vr_diag(6143, 0, 0, ell, Tcl, gcl, taudgcl)
                self.noises[recon_tag] = interp1d(ells, noise, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
            combination_method = self.combine_harmonic
        else:
            if recon_tag not in self.noises.keys():
                self.noises[recon_tag] = np.repeat(self.Noise_vr_diag(6143, 0, 0, 5, Tcl, gcl, taudgcl), 6144)
            combination_method = self.combine
        if recon_tag not in self.reconstructions.keys():
            self.Tmaps_filtered[recon_tag], self.lssmaps_filtered[recon_tag], self.reconstructions[recon_tag] = combination_method(Tmap, lssmap, mask, Tcl, gcl, taudgcl, self.noises[recon_tag])
        if recon_tag not in self.Cls.keys():
            self.Cls[recon_tag] = hp.anafast(self.reconstructions[recon_tag])


def twopt_bandpowers(recon_Cls, theory_noise, FSKY, plottitle, filename, lmaxplot=700, convert_K=True):
    if not filename.endswith('.png'):
        filename += '.png'
    ell = np.arange(2, lmaxplot)
    n_bands = (len(ell) - 1) // 5
    ell_bands = np.array([np.mean(ell[i*5:(i+1)*5]) for i in range(n_bands)])
    Cls_bands = np.zeros((3, n_bands))
    for i in range(n_bands):
        Cls_bands[0, i] = np.mean(recon_Cls[i*5:(i+1)*5])
        Cls_bands[1, i] = np.mean(theory_noise[i*5:(i+1)*5])
        #Cls_bands[2, i] = np.mean(Clvv[i*5:(i+1)*5])
    plt.figure()
    plt.loglog(ell_bands, Cls_bands[0], label='Reconstruction')
    if convert_K:
        plt.loglog(ell_bands, Cls_bands[1] * FSKY / 2.725**2, label='Theory * fsky')
    else:
        plt.loglog(ell_bands, Cls_bands[1] * FSKY, label='Theory * fsky')
    # plt.loglog(ell_bands[:2], Cls_bands[2, :2], label='Theory Signal') 
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')
  
def plot_gg_tg_comparison(rc1, rc2, tn1, tn2, FSKY, plottitle, filename, lmaxplot=700, convert_K=True):
    if not filename.endswith('.png'):
        filename += '.png'
    ell = np.arange(2, lmaxplot)
    n_bands = (len(ell) - 1) // 5
    ell_bands = np.array([np.mean(ell[i*5:(i+1)*5]) for i in range(n_bands)])
    Cls_bands = np.zeros((5, n_bands))
    for i in range(n_bands):
        Cls_bands[0, i] = np.mean(rc1[i*5:(i+1)*5])
        Cls_bands[1, i] = np.mean(rc2[i*5:(i+1)*5])
        Cls_bands[2, i] = np.mean(tn1[i*5:(i+1)*5])
        Cls_bands[3, i] = np.mean(tn2[i*5:(i+1)*5])
        #Cls_bands[4, i] = np.mean(Clvv[i*5:(i+1)*5])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.loglog(ell_bands, Cls_bands[0], label='Reconstruction')
    ax2.loglog(ell_bands, Cls_bands[1], label='Reconstruction')
    if convert_K:
        ax1.loglog(ell_bands, Cls_bands[2] * FSKY / 2.725**2, label='Theory * fsky')
        ax2.loglog(ell_bands, Cls_bands[3] * FSKY / 2.725**2, label='Theory * fsky')
    else:
        ax1.loglog(ell_bands, Cls_bands[2] * FSKY, label='Theory * fsky')
        ax2.loglog(ell_bands, Cls_bands[3] * FSKY, label='Theory * fsky')
    # plt.loglog(ell_bands[:2], Cls_bands[4, :2], label='Theory Signal') 
    fig.suptitle(plottitle)
    ax1.set_title('tg is gg')
    ax2.set_title('tg isn\'t gg')
    ax1.set_xlabel(r'$\ell$')
    ax2.set_xlabel(r'$\ell$')
    ax1.set_ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$')
    ax1.legend()
    ax2.legend()
    plt.savefig(outdir + filename)
    plt.close('all')

outdir = 'plots/analysis/paper_analysis_latest/'
if not os.path.exists(outdir):
    os.makedirs(outdir)


#maplist = Maps(allmaps=False, fullsky=False)
#delta_to_g = maplist.maps['unWISE'].sum() / maplist.maps['unWISE'].size
#maplist.gaussian_Cls['unWISE'] = hp.anafast(maplist.gaussian_maps['unWISE'])
#maplist.gaussian_Cls['SMICA']  = hp.anafast(maplist.gaussian_maps['SMICA'])
estim = Estimator()
csm = Cosmology()
csm.compute_Cls(ngbar=2.60)
###recon_tag = estim.get_recon_tag('SMICA', 'unWISE', True, True, True, False, 'nonhybrid')
#estim.reconstruct(maplist.gaussian_maps['SMICA'], maplist.gaussian_maps['unWISE'], maplist.gaussian_Cls['SMICA'], maplist.gaussian_Cls['unWISE'], maplist.mask_map, csm.Cltaudg, recon_tag)
#twopt_bandpowers(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, 'SMICA x unWISE   [gauss x gauss]', 'recon_SMICAxunWISE_fullsky_'+recon_tag, lmaxplot=1000)

mask_map = np.load('data/mask_unWISE_thres_v10.npy')
fullsky_mask = np.ones(12*2048**2)
mask_planck70 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL070'],n2r=True)

unWISEmap = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
#unWISEmap_inpainted = unWISEmap.copy()
unWISEmap_inpainted_planck = unWISEmap.copy()
ngal_per_pix = 2.60
#ngal_fill = np.random.poisson(lam=ngal_per_pix, size=np.where(mask_map==0)[0].size)
ngal_fill_planck = np.random.poisson(lam=ngal_per_pix, size=np.where(mask_planck70==0)[0].size)
#unWISEmap_inpainted[np.where(mask_map==0)] = ngal_fill
unWISEmap_inpainted_planck[np.where(mask_planck70==0)] = ngal_fill_planck
gcl_full = hp.anafast(unWISEmap)
#gcl_inpainted = hp.anafast(unWISEmap_inpainted)
gmap_full_gauss = hp.synfast(gcl_full, 2048)
gmap_full_real = unWISEmap.copy()
#gmap_inpainted_gauss = hp.synfast(gcl_inpainted, 2048)
#gmap_inpainted_real = unWISEmap_inpainted.copy()
gmap_inpainted_planck_gauss = hp.synfast(hp.anafast(unWISEmap_inpainted_planck), 2048)
gmap_inpainted_planck_real = unWISEmap_inpainted_planck.copy()

SMICAinp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True)
noisegen = hp.synfast(np.repeat([hp.anafast(SMICAinp, lmax=2500)[-1]], 6144), 2048)
SMICAmap_gauss = hp.synfast(hp.anafast(SMICAinp + noisegen),2048)
SMICAmap_real = SMICAinp + noisegen

SMICAbeam = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143)

#masks = {'unWISE mask' : mask_map, 'no mask' : fullsky_mask, 'Planck 0.7 galaxy cut' : mask_planck70}
masks = {}
skycuts = { key : np.where(masks[key]!=0)[0].size / masks[key].size for key in masks}

maskdebeam_alms = lambda mask : hp.almxfl(hp.map2alm(mask * SMICAmap_real), 1 / SMICAbeam)
maskdebeam_alms_gauss = lambda mask : hp.almxfl(hp.map2alm(mask * SMICAmap_gauss), 1 / SMICAbeam)

# Pre masking
Tlms_real = { key : maskdebeam_alms(masks[key]) for key in masks}
TCls_real = { key : hp.alm2cl(Tlms_real[key]) / skycuts[key] for key in Tlms_real}
Tmaps_real = { key : hp.alm2map(Tlms_real[key], 2048) for key in Tlms_real}
Tlms_gauss = { key : maskdebeam_alms_gauss(masks[key]) for key in masks}
TCls_gauss = { key : hp.alm2cl(Tlms_gauss[key]) / skycuts[key] for key in Tlms_gauss}
Tmaps_gauss = { key : hp.alm2map(Tlms_gauss[key], 2048) for key in Tlms_gauss}
# Post masking
masks={}
skycuts={}
masks['unWISE mask (post-recon)'] = mask_map
skycuts = { key : np.where(masks[key]!=0)[0].size / masks[key].size for key in masks}
#skycuts['unWISE mask (post-recon)'] = skycuts['unWISE mask']
Tlms_real['unWISE mask (post-recon)'] = hp.almxfl(hp.map2alm(SMICAmap_real), 1 / SMICAbeam)
Tlms_gauss['unWISE mask (post-recon)'] = hp.almxfl(hp.map2alm(SMICAmap_gauss), 1 / SMICAbeam)
TCls_real['unWISE mask (post-recon)'] = hp.alm2cl(Tlms_real['unWISE mask (post-recon)'])  # No fsky factor because it is full sky in the filter
TCls_gauss['unWISE mask (post-recon)'] = hp.alm2cl(Tlms_gauss['unWISE mask (post-recon)'])  # No fsky factor because it is full sky in the filter
Tmaps_real['unWISE mask (post-recon)'] = hp.alm2map(Tlms_real['unWISE mask (post-recon)'], 2048)
Tmaps_gauss['unWISE mask (post-recon)'] = hp.alm2map(Tlms_gauss['unWISE mask (post-recon)'], 2048)

Clvv = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

#for gcase in ('mapspectrum', 'fullmap', 'inpainted', 'inpainted_planck'):
for gcase in (['fullmap', 'inpainted_planck']):
    print('gcase: ' + gcase)
    gwindows = {'fullmap' : 1.2, 'inpainted' : 0.65}
    if gcase in ('fullmap', 'inpainted'):
        csm.compute_Cls(ngbar=2.60, gwindow_zdep=gwindows[gcase])
        clgg_gcase = csm.Clgg.copy()
    elif gcase == 'mapspectrum':
        csm.compute_Cls(ngbar=2.60, gwindow_zdep=gwindows['fullmap'])  # Optional, slightly better taudg scaling for map gg
        clgg_gcase = gcl_full.copy()
    elif gcase == 'inpainted_planck':
        csm.compute_Cls(ngbar=2.60)
        clgg_gcase = csm.Clgg.copy()
    for case in masks:
        print('   reconstruction case: ' + case)
        for Tgauss in (True, False):
            if Tgauss:
                Tmap = Tmaps_gauss[case].copy()
                Tcl  = TCls_gauss[case].copy()
            else:
                Tmap = Tmaps_real[case].copy()
                Tcl  = TCls_real[case].copy()
            for ggauss in (True, False):
                if ggauss:
                    if gcase == 'inpainted':
                        gmap = gmap_inpainted_gauss.copy()
                    elif gcase == 'inpainted_planck':
                        gmap = gmap_inpainted_planck_gauss.copy()
                    else:
                        gmap = gmap_full_gauss.copy()
                else:
                    if gcase == 'inpainted':
                        gmap = gmap_inpainted_real.copy()
                    elif gcase == 'inpainted_planck':
                        gmap = gmap_inpainted_planck_real.copy()
                    else:
                        gmap = gmap_full_real.copy()
                recon_tag = estim.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, notes=gcase+case)
                estim.reconstruct(Tmap, gmap, Tcl, clgg_gcase, masks[case], csm.Cltaudg, recon_tag)
        recon_tags = [estim.get_recon_tag('SMICA', 'unWISE', tg, gg, notes=gcase+case) for tg in (True, False) for gg in (True, False)]
        gauss_tags = ['T=%s x lss=%s' % (tg,gg) for tg in ('gauss','real') for gg in ('gauss','real')]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,12))
        for i, ax in zip(np.arange(4), (ax1, ax2, ax3, ax4)):
            ax.loglog(estim.Cls[recon_tags[i]], lw=2)
            ax.loglog(estim.noises[recon_tags[i]] * skycuts[case] / 2.725**2, lw=2)
            ax.loglog(Clvv[:10])
            ax.set_title('SMICA x unWISE  [%s]\nmask = %s' % (gauss_tags[i], case), fontsize=16)
            if i > 1:
                ax.set_xlabel(r'$\ell$', fontsize=16)
            if i % 2 == 0:
                ax.set_ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$', fontsize=16)
        fig.suptitle('SMICA x unWISE Gaussian vs Real Inputs', fontsize=22)
        plt.tight_layout()
        plt.savefig(outdir + 'SMICAxunWISE_real_vs_gauss_case=%s_gcase=%s.png' % (case, gcase))
    #case_tags = ['no mask', 'Planck 0.7 galaxy cut', 'unWISE mask', 'unWISE mask (post-recon)']
    case_tags = ['unWISE mask (post-recon)']
    for tg in (True, False):
        for gg in (True, False):
            recon_tags = [estim.get_recon_tag('SMICA', 'unWISE', tg, gg, notes=gcase+casetag) for casetag in case_tags]
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,12))
            #for i, ax in zip(np.arange(4), (ax1, ax2, ax3, ax4)):
            for i, ax in zip(np.arange(1), [(ax4)]):
                ax.loglog(estim.Cls[recon_tags[i]], lw=2)
                ax.loglog(estim.noises[recon_tags[i]] * skycuts[case_tags[i]] / 2.725**2, lw=2)
                ax.loglog(Clvv[:10])
                ax.set_title('SMICA x unWISE  [T=%s x lss=%s]\nmask = %s' % (tg, gg, case_tags[i]), fontsize=16)
                if i > 1:
                    ax.set_xlabel(r'$\ell$', fontsize=16)
                if i % 2 == 0:
                    ax.set_ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$', fontsize=16)
            fig.suptitle('SMICA x unWISE Masking Effects', fontsize=22)
            plt.tight_layout()
            plt.savefig(outdir + 'SMICAxunWISE_%s_vs_%s_gcase=%s.png' % (tg, gg, gcase))







# if not os.path.exists('./maplist.p'):
#     print('Generating maps, please wait...')
#     pickle.dump(Maps(allmaps=False), open('./maplist.p', 'wb'))

# with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
#    lines = FILE.readlines()

# alex_ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float')[:-1]
# alex_clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')[:-1]
# alex_lssspec = interp1d(alex_ells,alex_clgg, bounds_error=False,fill_value='extrapolate')(np.arange(6144))

# maplist = pickle.load(open('./maplist.p', 'rb'))
# delta_to_g = maplist.maps['unWISE'].sum() / maplist.maps['unWISE'].size
# estim = Estimator()
# cosmologies = {}

# get_cosmo_tag = lambda case, gwindow_orig : '%sx%s' % (case, gwindow_orig)
# for case in ('nonhybrid', 'hybrid'):
# #for case in [('nonhybrid')]:
#     print('Running cases for Clgg = ' + case)
#     for gwindow_orig in (True, False):
#         print('  Running cases with original galaxy window: ' + str(gwindow_orig))
#         cosmo_tag = get_cosmo_tag(case, gwindow_orig)
#         cosmologies[cosmo_tag] = Cosmology()
#         csm = cosmologies[cosmo_tag]
#         csm.compute_Cls(ngbar=delta_to_g, gwindow_orig=gwindow_orig)
#         if case == 'hybrid':
#             csm.Clmm_as_gg = csm.Clmm*delta_to_g**2
#             csm.Clgg = np.where(csm.Clmm_as_gg > csm.Clgg, csm.Clmm_as_gg, csm.Clgg)
#             csm.Cltaudg = ((csm.Clgg/delta_to_g**2) - 9.2e-8) * csm.get_limber_window('taud',avg=True,gwindow_orig=gwindow_orig) * delta_to_g / csm.get_limber_window('g',avg=True,gwindow_orig=gwindow_orig)
#             csm.Cltaudtaud = ((csm.Clgg/delta_to_g**2)-9.2e-8)*csm.get_limber_window('taud',avg=True,gwindow_orig=gwindow_orig)**2/csm.get_limber_window('g',avg=True,gwindow_orig=gwindow_orig)**2
#         maplist.Cls['unWISE'] = csm.Clgg.copy()
#         maplist.gaussian_Cls['unWISE'] = csm.Clgg.copy()
#         tg_gg_offset = csm.get_limber_window('taud', avg=True, gwindow_orig=gwindow_orig) / csm.get_limber_window('g', avg=True, gwindow_orig=gwindow_orig) / delta_to_g
#         for Tgauss in (True, False):
#         #for Tgauss in [(True)]:
#             print('    Running cases for Tmap = gauss = ' + str(Tgauss))
#             if Tgauss:
#                 Tmap = maplist.gaussian_maps['SMICA']
#                 Tcl  = maplist.gaussian_Cls['SMICA']
#             else:
#                 Tmap = maplist.maps['SMICA']
#                 Tcl  = maplist.Cls['SMICA']
#             for ggauss in (True, False):     
#                 print('      Running cases for lssmap = gauss = ' + str(ggauss))       
#                 if ggauss:
#                     lssmap = maplist.gaussian_maps['unWISE']
#                     gcl    = maplist.gaussian_Cls['unWISE']
#                 else:
#                     lssmap = maplist.maps['unWISE']
#                     gcl    = maplist.Cls['unWISE']
#                 for tg_is_gg in (True, False):
#                     if tg_is_gg:
#                         cltaudg = csm.Clgg * tg_gg_offset
#                         cltaudtaud = csm.Clgg * tg_gg_offset**2
#                     else:
#                         cltaudg = csm.Cltaudg.copy()
#                         cltaudtaud = csm.Cltaudtaud.copy()
#                     recon_tag = estim.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, tg_is_gg, gwindow_orig, case)
#                     estim.reconstruct(Tmap, lssmap, Tcl, gcl, maplist.mask_map, cltaudg, recon_tag)
#                     #plt.figure()
#                     #plt.loglog(maplist.Cls['unWISE'], label='Map Cls')
#                     #plt.loglog(alex_lssspec*delta_to_g**2, label='Alex\'s Clgg')
#                     #plt.loglog(csm.Clgg, label='Clgg')
#                     #plt.loglog(cltaudg**2/cltaudtaud, label=r'${C_\ell^{\dot{\tau}\mathrm{g}^2}} / C_\ell^{\dot{\tau}\dot{\tau}}$', ls='--')
#                     #plt.loglog(csm.Clmm*delta_to_g**2, label='Clmm')
#                     #plt.xlabel(r'$\ell$')
#                     #plt.ylabel(r'$C_\ell^{\mathrm{Xg}}$')
#                     #plt.ylim([plt.ylim()[0], 0.02])
#                     #plt.legend()
#                     #plt.savefig(outdir + 'galspec_' + recon_tag + '.png')
#                     #twopt_bandpowers(estim.Cls[recon_tag],
#                     #                 estim.noises[recon_tag],
#                     #                 maplist.fsky,
#                     #                 'SMICA x unWISE   [%s x %s]'%('real' if not Tgauss else 'gauss', 'real' if not ggauss else 'gauss'),
#                     #                 'recon_SMICAxunWISE_'+recon_tag,
#                     #                 lmaxplot=4000)
#                 rtag_tg_is_gg = estim.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, True, gwindow_orig, case)
#                 rtag_tg_isnt_gg = estim.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, False, gwindow_orig, case)
#                 plot_gg_tg_comparison(estim.Cls[rtag_tg_is_gg],
#                                       estim.Cls[rtag_tg_isnt_gg],
#                                       estim.noises[rtag_tg_is_gg],
#                                       estim.noises[rtag_tg_isnt_gg],
#                                       maplist.fsky,
#                                       'SMICA x unWISE   [%s x %s]'%('real' if not Tgauss else 'gauss', 'real' if not ggauss else 'gauss'),
#                                       'recon_SMICAxunWISE_'+rtag_tg_is_gg.replace('__Cltaudg~Clgg=True',''),
#                                       lmaxplot=4000)


# ## ax1 and ax3 are the left column and will show constant zdep
# ## ax2 and ax4 are the right column and will show constant clgg case
# N_unit = maplist.fsky / 2.725**2  # Multiply by theory noise to plot consistent units against reconstruction Cls (v^2/c^2)
# rtag_zorig_nonhybrid = estim.get_recon_tag('SMICA', 'unWISE', True, True, True, True, 'nonhybrid')
# rtag_zorig_hybrid    = estim.get_recon_tag('SMICA', 'unWISE', True, True, True, True, 'hybrid')
# rtag_znew_nonhybrid  = estim.get_recon_tag('SMICA', 'unWISE', True, True, True, False, 'nonhybrid')
# rtag_znew_hybrid     = estim.get_recon_tag('SMICA', 'unWISE', True, True, True, False, 'hybrid')

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,12))

# ax1.set_title('Original galaxy window', fontsize=18)
# ax1.set_ylabel('Unmodified Clgg', fontsize=18)
# ax3.set_ylabel('Modified Clgg', fontsize=18)
# ax1.loglog(estim.Cls[rtag_zorig_nonhybrid])
# ax1.loglog(estim.noises[rtag_zorig_nonhybrid] * N_unit)
# ax3.loglog(estim.Cls[rtag_zorig_hybrid])
# ax3.loglog(estim.noises[rtag_zorig_hybrid] * N_unit)

# ax2.set_title('Modified galaxy window', fontsize=18)
# ax2.loglog(estim.Cls[rtag_znew_nonhybrid])
# ax2.loglog(estim.noises[rtag_znew_nonhybrid] * N_unit)
# ax4.loglog(estim.Cls[rtag_znew_hybrid])
# ax4.loglog(estim.noises[rtag_znew_hybrid] * N_unit)

# for ax in (ax1, ax2, ax3, ax4):
#     ax.set_xlabel(r'$\ell$', fontsize=16)

# fig.suptitle(r'SMICA(gauss) x unWISE(gauss)    (where Clgg$\propto$Cltaudg)', fontsize=22)
# for ax in (ax2, ax3, ax4):
#     ax.set_ylim(ax1.get_ylim())

# plt.savefig('f')










# rtag_zorig_nonhybrid_greal = estim.get_recon_tag('SMICA', 'unWISE', True, False, True, True, 'nonhybrid')
# rtag_zorig_hybrid_greal    = estim.get_recon_tag('SMICA', 'unWISE', True, False, True, True, 'hybrid')
# rtag_znew_nonhybrid_greal  = estim.get_recon_tag('SMICA', 'unWISE', True, False, True, False, 'nonhybrid')
# rtag_znew_hybrid_greal     = estim.get_recon_tag('SMICA', 'unWISE', True, False, True, False, 'hybrid')

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,12))

# ax1.set_title('Original galaxy window', fontsize=18)
# ax1.set_ylabel('Unmodified Clgg', fontsize=18)
# ax3.set_ylabel('Modified Clgg', fontsize=18)
# ax1.loglog(estim.Cls[rtag_zorig_nonhybrid_greal])
# ax1.loglog(estim.noises[rtag_zorig_nonhybrid_greal] * N_unit)
# ax3.loglog(estim.Cls[rtag_zorig_hybrid_greal])
# ax3.loglog(estim.noises[rtag_zorig_hybrid_greal] * N_unit)

# ax2.set_title('Modified galaxy window', fontsize=18)
# ax2.loglog(estim.Cls[rtag_znew_nonhybrid_greal])
# ax2.loglog(estim.noises[rtag_znew_nonhybrid_greal] * N_unit)
# ax4.loglog(estim.Cls[rtag_znew_hybrid_greal])
# ax4.loglog(estim.noises[rtag_znew_hybrid_greal] * N_unit)

# for ax in (ax1, ax2, ax3, ax4):
#     ax.set_xlabel(r'$\ell$', fontsize=16)

# fig.suptitle(r'SMICA(gauss) x unWISE(real)    (where Clgg$\propto$Cltaudg)', fontsize=22)
# for ax in (ax2, ax3, ax4):
#     ax.set_ylim(ax1.get_ylim())

# plt.savefig('t')







# rtag_zorig_nonhybrid_greal_tg = estim.get_recon_tag('SMICA', 'unWISE', True, False, False, True, 'nonhybrid')
# rtag_zorig_hybrid_greal_tg    = estim.get_recon_tag('SMICA', 'unWISE', True, False, False, True, 'hybrid')
# rtag_znew_nonhybrid_greal_tg  = estim.get_recon_tag('SMICA', 'unWISE', True, False, False, False, 'nonhybrid')
# rtag_znew_hybrid_greal_tg     = estim.get_recon_tag('SMICA', 'unWISE', True, False, False, False, 'hybrid')

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,12))

# ax1.set_title('Original galaxy window', fontsize=18)
# ax1.set_ylabel('Unmodified Clgg', fontsize=18)
# ax3.set_ylabel('Modified Clgg', fontsize=18)
# ax1.loglog(estim.Cls[rtag_zorig_nonhybrid_greal_tg])
# ax1.loglog(estim.noises[rtag_zorig_nonhybrid_greal_tg] * N_unit)
# ax3.loglog(estim.Cls[rtag_zorig_hybrid_greal_tg])
# ax3.loglog(estim.noises[rtag_zorig_hybrid_greal_tg] * N_unit)

# ax2.set_title('Modified galaxy window', fontsize=18)
# ax2.loglog(estim.Cls[rtag_znew_nonhybrid_greal_tg])
# ax2.loglog(estim.noises[rtag_znew_nonhybrid_greal_tg] * N_unit)
# ax4.loglog(estim.Cls[rtag_znew_hybrid_greal_tg])
# ax4.loglog(estim.noises[rtag_znew_hybrid_greal_tg] * N_unit)

# for ax in (ax1, ax2, ax3, ax4):
#     ax.set_xlabel(r'$\ell$', fontsize=16)

# fig.suptitle(r'SMICA(gauss) x unWISE(real)    (where Cltaudg is modelled)', fontsize=22)
# for ax in (ax2, ax3, ax4):
#     ax.set_ylim(ax1.get_ylim())

# plt.savefig('tg')
