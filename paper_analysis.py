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

Clvv = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

outdir = 'plots/analysis/paper_analysis_latest/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

class Maps(object):
    def get_alms(self, tag, gauss=False):
        if 'unWISE' in tag:  # Use with care, harmonic transform doesn't preserve integer pixel values
            return hp.map2alm(self.input_maps[tag])
        if gauss:
            # Important order of operations. For Gaussian T maps, the power spectrum of the full sky and cut sky are significantly different.
            # Therefore to best match the input T map spectrum to the theory T spectrum the following order is required:
            # 1) Spectrum used must be (Tmap * mask) / fsky   to represent the full sky power of T
            # 2) Generate Gaussian realization
            # 3) Remask. The power spectrum matches slightly better if we don't remask, but we are using masked T inputs and the mask doesn't
            #            survive the Gaussification of the power spectrum. For this reason we also want a separate cache of Gaussian Tmap Cls
            # 4) Debeam
            return hp.almxfl(hp.map2alm(hp.synfast(hp.anafast(self.input_maps[tag] * self.mask_map) / self.fsky, 2048) * self.mask_map), 1/self.beams[tag.split('-')[0]])
        else:
            return hp.almxfl(hp.map2alm(self.input_maps[tag] * self.mask_map), 1/self.beams[tag.split('-')[0]])  
    def alm2map(self, alms, tag):
        return hp.alm2map(alms[tag], 2048)
    def get_Cls(self, alms, tag):
        return hp.alm2cl(alms[tag.split('-')[0]]) / self.fsky  # For CMB subtracted maps we only want to reconstruct the subtracted map, we want the full frequency sky for the theory Cls
    def __init__(self):
        print('Loading input maps')
        self.mask_map = np.load('data/mask_unWISE_thres_v10.npy')
        self.fsky = np.where(self.mask_map!=0)[0].size / self.mask_map.size
        #self.map_tags = ('SMICA', '100GHz', '143GHz', '217GHz', '100GHz-SMICA', '143GHz-SMICA', '217GHz-SMICA', 'unWISE')
        self.map_tags = ('SMICA', 'unWISE')
        self.input_maps = {'SMICA_input' : hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True),
                           '100GHz' : hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_100_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True),
                           '143GHz' : hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True),
                           '217GHz' : hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_217_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True),
                           '100GHz-SMICA' : fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits')[1].data['INTENSITY'].flatten(),
                           '143GHz-SMICA' : fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits')[1].data['INTENSITY'].flatten(),
                           '217GHz-SMICA' : fits.open('data/planck_data_testing/maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits')[1].data['INTENSITY'].flatten(),
                           'unWISE_input' : fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten() }
        self.beams = { 'SMICA' : hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143),
                       '100GHz' : hp.gauss_beam(fwhm=np.radians(9.66/60), lmax=6143),
                       '143GHz' : hp.gauss_beam(fwhm=np.radians(7.27/60), lmax=6143),
                       '217GHz' : hp.gauss_beam(fwhm=np.radians(5.01/60), lmax=6143) }
        self.beams['100GHz'][4001:] = self.beams['100GHz'][4000]  # Numerical trouble for healpix for beams that blow up at large ell. Sufficient to flatten out at high ell above where the 1/TT filter peaks
        print('Adding Gaussian noise to CMB maps')
        self.input_maps['SMICA'] = self.input_maps['SMICA_input'] + hp.synfast(np.repeat([hp.anafast(self.input_maps['SMICA_input'], lmax=2500)[-1]], 6144), 2048)
        print('Inpainting unWISE map in masked regions')
        ngal_per_pix = self.input_maps['unWISE_input'][np.where(self.mask_map!=0)].sum() / np.where(self.mask_map!=0)[0].size
        ngal_fill = np.random.poisson(lam=ngal_per_pix, size=np.where(self.mask_map==0)[0].size)
        self.input_maps['unWISE'] = self.input_maps['unWISE_input'].copy()
        self.input_maps['unWISE'][np.where(self.mask_map==0)] = ngal_fill
        print('Masking and debeaming temperature maps')
        self.map_alms = { tag : self.get_alms(tag) for tag in self.map_tags}
        self.maps = { tag : self.alm2map(self.map_alms, tag) for tag in self.map_tags}
        self.Cls = { tag : self.get_Cls(self.map_alms, tag) for tag in self.map_tags}
        print('Generating Gaussian realizations of maps')
        self.gaussian_alms = { tag : self.get_alms(tag, gauss=True) for tag in self.map_tags}
        self.gaussian_maps = { tag : self.alm2map(self.gaussian_alms, tag) for tag in self.map_tags}
        self.gaussian_Cls = { tag : self.get_Cls(self.gaussian_alms, tag) for tag in self.map_tags}  # Because these aren't the same. Somehow masking the gaussian realization changes the underlying power spectrum??
        print('Handling unWISE map and generating Poisson realization')
        self.maps['unWISE'] = self.input_maps['unWISE'].copy()  # Harmonic transform not friendly to integer pixel values
        self.Cls['unWISE'] = hp.anafast(self.maps['unWISE'])
        self.gaussian_maps['unWISE'] = np.random.poisson(lam=ngal_per_pix, size=self.maps['unWISE'].size)
        self.gaussian_Cls['unWISE'] = hp.anafast(self.gaussian_maps['unWISE'])



class Estimator(object):
    def a(self, z):
        return 1/(1+z)
    def ne0z(self, z):  # units 1/m^3, no actual z dependence?????
        H100_SI = 3.241e-18  # 1/s
        G_SI = 6.674e-11
        mProton_SI = 1.673e-27  # kg
        chi = 0.86
        me = 1.14
        gasfrac = 0.9
        omgh2 = gasfrac* 0.049*0.68**2
        ne0_SI = chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me                   
        return ne0_SI
    def __init__(self):
        self.reconstructions = {}
        self.Cls = {}
        self.noises = {}
        self.Tmaps_filtered = {}
        self.lssmaps_filtered = {}
        self.sigma_T = 6.6524587e-29  # m^2
        self.mperMpc = 3.086e22  # metres per megaparsec
        self.cambpars = camb.CAMBparams()
        self.cambpars.set_cosmology(H0 = conf.H0, ombh2=conf.ombh2, \
                                                  omch2=conf.omch2, mnu=conf.mnu , \
                                                  omk=conf.Omega_K, tau=conf.tau,  \
                                                  TCMB =2.725 )
        self.cambpars.InitPower.set_params(As =conf.As*1e-9 ,ns=conf.ns, r=0)
        self.cambpars.NonLinear = model.NonLinear_both
        self.cambpars.max_eta_k = 14000.0*conf.ks_hm[-1]
        self.cosmology_data = camb.get_background(self.cambpars)
        self.zmin = conf.z_min
        self.zmax = conf.z_max
        self.bin_width = (self.cosmology_data.comoving_radial_distance(self.zmax)-self.cosmology_data.comoving_radial_distance(self.zmin)) * self.mperMpc
        zrange = np.where(np.logical_and(conf.zs_hm>=self.zmin, conf.zs_hm<=self.zmax))[0]
        zmin_ind = zrange.min()
        zmax_ind = zrange.max()
        a_integrated = np.abs(simps(conf.zs_hm[zmin_ind:zmax_ind], self.a(conf.zs_hm[zmin_ind:zmax_ind])))
        self.K = -self.sigma_T * self.bin_width * a_integrated * self.ne0z(self.zmax)  # conversion from delta to tau dot, unitless conversion
    def get_recon_tag(self, Ttag, gtag, Tgauss, ggauss):
        return Ttag + '_gauss=' + str(Tgauss) + '__' + gtag + '_gauss=' + str(ggauss)
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
    def Noise_vr_diag(self, lmax, alpha, gamma, ell, cltt, clgg_binned):
        terms = 0
        cltaudg_binned = self.K * clgg_binned
        for l2 in np.arange(lmax):
            for l1 in np.arange(np.abs(l2-ell),l2+ell+1):
                if l1 > lmax-1 or l1 <2:   #triangle rule
                    continue
                gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell+1)/(4*np.pi))*self.wigner_symbol(ell, l1, l2)*cltaudg_binned[l2]
                term_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg_binned[l2]))
                if np.isfinite(term_entry):
                    terms += term_entry
        return (2*ell+1) / terms
    def combine(self, Tmap, lssmap, mask, ClTT, Clgg, Noise, convert_K=True):
        dTlm = hp.map2alm(Tmap)
        dlm  = hp.map2alm(lssmap)
        cltaudg = self.K * Clgg.copy()
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
        return Tmap_filtered * const_noise, lssmap_filtered, outmap
    def reconstruct(self, maps, Ttag, gtag, Tgauss, ggauss):  # Meant to work with Map class
        recon_tag = self.get_recon_tag(Ttag, gtag, Tgauss, ggauss)
        if Tgauss:
            Tmap = maps.gaussian_maps[Ttag]
            Tcl  = maps.gaussian_Cls[Ttag]
        else:
            Tmap = maps.maps[Ttag]
            Tcl  = maps.Cls[Ttag]
        if ggauss:
            lssmap = maps.gaussian_maps[gtag]
            gcl = maps.gaussian_Cls[gtag]
        else:
            lssmap = maps.maps[gtag]
            gcl = maps.Cls[gtag]
        if recon_tag not in self.noises.keys():
            self.noises[recon_tag] = np.repeat(self.Noise_vr_diag(6143, 0, 0, 5, Tcl, gcl), 6144)
        if recon_tag not in self.reconstructions.keys():
            self.Tmaps_filtered[recon_tag], self.lssmaps_filtered[recon_tag], self.reconstructions[recon_tag] = self.combine(Tmap, lssmap, maps.mask_map, Tcl, gcl, self.noises[recon_tag])
        if recon_tag not in self.Cls.keys():
            self.Cls[recon_tag] = hp.anafast(self.reconstructions[recon_tag])

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

maplist = Maps()
estim = Estimator()

for Ttag in maplist.map_tags[:-1]:
    gtag = maplist.map_tags[-1]
    for Tgauss in (True, False):
        for ggauss in (True, False):
            print('Reconstructing %s x %s  [Tgauss=%s, lssgauss=%s]' % (Ttag, gtag, Tgauss, ggauss))
            estim.reconstruct(maplist, Ttag, gtag, Tgauss, ggauss)
            recon_tag = estim.get_recon_tag(Ttag, gtag, Tgauss, ggauss)
            twopt(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, recon_tag, recon_tag)


### Would be nice to do multiple realizations to see the cosmic variance error bars
### Might be too big a project, but we could consider what happens if we inpaint without destroying correlations
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 1) Consider SMICA x unWISE. (If it's about as good as what we see below, we can consider what might improve things. Such as an actual tau filter)
### 2) Clgg = Pmm * bias(z) * bias(z) + shotnoise
### 3) Cltaug = Pmm * K * bias(z)
### Instead of the green input Clgg from the case plots, use these spectra instead. This should be better like Alex's.

##
### As masking grows what is effect on reconstruction

### In the prefactor just use the green curve: inpainting of all masked regions, take power spectrum, that is Clgg
### Don't mask it


### Does inpainting change the shape in a favourable way? Inpainting all masked regions.
### Prefactor ONLY has the non-inpainted spectrum (alex's line)
### Mask convolves in filter. Tau should be modeled cuz delta != delta_e