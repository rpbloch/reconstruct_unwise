'''
Figure out power spec of gaussian unWISE with this new spectrum

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
from scipy.interpolate import interp1d, interp2d
from astropy.io import fits
import camb
from camb import model
from scipy.integrate import simps
from math import lgamma
import pickle
from spectra import fftlog_integral, save_fft_weights, save_fft, limber, beyond_limber

sigma_T = 6.65245871e-29  # m^2
metres_per_megaparsec = 3.086e22
G_SI = 6.674e-11
mProton_SI = 1.673e-27
H100_SI = 3.241e-18
thompson_SI = 6.6524e-29

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
    def __init__(self, allmaps=True):
        print('Loading input maps')
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
        ngal_per_pix = self.input_maps['unWISE_input'][np.where(self.mask_map!=0)].sum() / np.where(self.mask_map!=0)[0].size
        ngal_fill = np.random.poisson(lam=ngal_per_pix, size=np.where(self.mask_map==0)[0].size)
        self.input_maps['unWISE'] = self.input_maps['unWISE_input'].copy()
        self.input_maps['unWISE'][np.where(self.mask_map==0)] = ngal_fill
        print('Masking and debeaming temperature maps')
        self.map_alms = { tag : self.get_alms(tag) for tag in self.map_tags[:-1]}
        self.maps = { tag : self.alm2map(self.map_alms, tag) for tag in self.map_tags[:-1]}
        self.Cls = { tag : self.get_Cls(self.map_alms, tag) for tag in self.map_tags[:-1]}
        print('Generating Gaussian realizations of maps')
        self.gaussian_alms = { tag : self.get_alms(tag, gauss=True) for tag in self.map_tags[:-1]}
        self.gaussian_maps = { tag : self.alm2map(self.gaussian_alms, tag) for tag in self.map_tags[:-1]}
        self.gaussian_Cls = { tag : self.get_Cls(self.gaussian_alms, tag) for tag in self.map_tags[:-1]}  # Because these aren't the same. Somehow masking the gaussian realization changes the underlying power spectrum??
        print('Handling unWISE map and generating Poisson realization')
        #### Gaussian realiation of unWISE map
        #### For each pixel, draw from Poisson distribution characterized by mean value = that pixel's value
        #### hopefully produces integer map with similar power spectrum to actual unWISE map
        self.maps['unWISE'] = self.input_maps['unWISE'].copy()  # Harmonic transform not friendly to integer pixel values
        # maplist.Cls['unWISE'] = estim.Clgg * (maplist.maps['unWISE'].sum()/maplist.maps['unWISE'].size)**2
        self.gaussian_maps['unWISE'] = np.random.poisson(lam=ngal_per_pix, size=self.maps['unWISE'].size)
        # maplist.gaussian_Cls['unWISE'] = estim.Clgg * (maplist.maps['unWISE'].sum()/maplist.maps['unWISE'].size)**2

class Cosmology(object):
    def chi_to_z(self, chi):
        return self.cosmology_data.redshift_at_comoving_radial_distance(chi)
    def z_to_chi(self, z):
        return self.cosmology_data.comoving_radial_distance(z)
    def ne0(self):
        G_SI = 6.674e-11
        mProton_SI = 1.673e-27
        H100_SI = 3.241e-18
        thompson_SI = 6.6524e-29
        m_per_Mpc = 3.086e22
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
        galaxy_bias = (0.8+1.2*conf.zs_hm)[:,np.newaxis]
        tau_bias = -thompson_SI * ne0() * simps(conf.zs_hm,(1+conf.zs_hm)**2) * m_per_Mpc * self.bin_width
        ee_mm_bias  = self.bias_e2(conf.zs_hm, conf.ks_hm)
        Pmm_full_sampled = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True,  hubble_units=False, k_hunit=False, kmax=conf.ks_hm[-1], zmax=conf.zs_hm[-1]).P(conf.zs_hm, conf.ks_hm, grid=True)
        Pmm_lin_sampled  = camb.get_matter_power_interpolator(self.cambpars, nonlinear=False, hubble_units=False, k_hunit=False, kmax=conf.ks_hm[-1], zmax=conf.zs_hm[-1]).P(conf.zs_hm, conf.ks_hm, grid=True)
        ##### Get Pmm. Project with spherical bessel, integrate over yadda yadda simple limber. Clmm. Multiply by biases scaled to mean_chi and check. NO RECCO!!!!!!!!
        Pgg_full = interp2d(conf.ks_hm, conf.zs_hm, Pmm_full_sampled * galaxy_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)
        Pgg_lin  = interp2d(conf.ks_hm, conf.zs_hm, Pmm_lin_sampled  * galaxy_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)
        Peg_full = interp2d(conf.ks_hm, conf.zs_hm, Pmm_full_sampled * galaxy_bias * ee_mm_bias, kind = 'linear', bounds_error=False, fill_value=0.0)
        Peg_lin  = interp2d(conf.ks_hm, conf.zs_hm, Pmm_lin_sampled  * galaxy_bias * ee_mm_bias, kind = 'linear', bounds_error=False, fill_value=0.0)
        Pee_full = interp2d(conf.ks_hm, conf.zs_hm, Pmm_full_sampled * ee_mm_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)
        Pee_lin  = interp2d(conf.ks_hm, conf.zs_hm, Pmm_lin_sampled  * ee_mm_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)
        c.dump(c.get_basic_conf(conf), Pee_full, 'p_ee_f1=None_f2 =None', dir_base='pks')
        c.dump(c.get_basic_conf(conf), Pee_lin,  'p_linear_ee_f1=None_f2 =None', dir_base='pks')
        c.dump(c.get_basic_conf(conf), Peg_full, 'p_eg_f1=None_f2 =None', dir_base='pks')
        c.dump(c.get_basic_conf(conf), Peg_lin,  'p_linear_eg_f1=None_f2 =None', dir_base='pks')
        c.dump(c.get_basic_conf(conf), Pgg_full, 'p_gg_f1=None_f2 =None', dir_base='pks')
        c.dump(c.get_basic_conf(conf), Pgg_lin,  'p_linear_gg_f1=None_f2 =None', dir_base='pks')
        ells = np.unique(np.append(np.geomspace(1,6143,120).astype(int), 6143))
        self.Clgg = np.zeros(ells.size)
        self.Cltaudg = np.zeros(ells.size)
        self.Cltaudtaud = np.zeros(ells.size)
        save_fft_weights('g',None)
        save_fft_weights('taud',None)
        for l, ell in enumerate(ells):
            save_fft('g',None,0,ell)
            save_fft('taud',None,0,ell)
            if ell < 30:  # UNWISE: Possible for UNWISE this may have to be flagged to a high or lmax lswitch
                pee_limb = peg_limb = pgg_limb = None
            else:   
                chis_interp = np.linspace(self.cosmology_data.comoving_radial_distance(1e-2), self.cosmology_data.comoving_radial_distance(conf.z_max+1.1), 1000)            
                pee_limb_sample = limber(Pee_full, chis_interp, ell) - limber(Pee_lin, chis_interp, ell)
                peg_limb_sample = limber(Peg_full, chis_interp, ell) - limber(Peg_lin, chis_interp, ell)
                pgg_limb_sample = limber(Pgg_full, chis_interp, ell) - limber(Pgg_lin, chis_interp, ell)
                pee_limb = interp1d(chis_interp, pee_limb_sample, kind='linear', bounds_error=False, fill_value=0.0)
                peg_limb = interp1d(chis_interp, peg_limb_sample, kind='linear', bounds_error=False, fill_value=0.0)
                pgg_limb = interp1d(chis_interp, pgg_limb_sample, kind='linear', bounds_error=False, fill_value=0.0)
            self.Cltaudg[l] = beyond_limber('taud','g',None,None,0,0,fftlog_integral('taud',None,0,ell)[0],ell, peg_limb)
            self.Clgg[l]    = beyond_limber('g','g',   None,None,0,0,fftlog_integral('g',None,0,ell)[0], ell, pgg_limb,) + 9.2e-8
            self.Cltaudtaud[l]    = beyond_limber('taud','taud',   None,None,0,0,fftlog_integral('taud',None,0,ell)[0], ell, pee_limb)
        self.Clgg = interp1d(ells, self.Clgg, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
        self.Cltaudg = interp1d(ells, self.Cltaudg, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
        self.Cltaudtaud = interp1d(ells, self.Cltaudtaud, bounds_error=False, fill_value='extrapolate')(np.arange(6144))


class Estimator(object):
    def __init__(self):
        self.reconstructions = {}
        self.Cls = {}
        self.noises = {}
        self.Tmaps_filtered = {}
        self.lssmaps_filtered = {}
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
    def reconstruct(self, maps, Ttag, gtag, Tgauss, ggauss, taudgcl):  # Meant to work with Map class
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
            self.noises[recon_tag] = np.repeat(self.Noise_vr_diag(6143, 0, 0, 5, Tcl, gcl, taudgcl), 6144)
        if recon_tag not in self.reconstructions.keys():
            self.Tmaps_filtered[recon_tag], self.lssmaps_filtered[recon_tag], self.reconstructions[recon_tag] = self.combine(Tmap, lssmap, maps.mask_map, Tcl, gcl, taudgcl, self.noises[recon_tag])
        if recon_tag not in self.Cls.keys():
            self.Cls[recon_tag] = hp.anafast(self.reconstructions[recon_tag])

def twopt_bandpowers(recon_Cls, theory_noise, FSKY, plottitle, filename, lmaxplot=700, convert_K=True):
    ell = np.arange(2, lmaxplot)
    n_bands = (len(ell) - 1) // 5
    ell_bands = np.array([np.mean(ell[i*5:(i+1)*5]) for i in range(n_bands)])
    Cls_bands = np.zeros((3, n_bands))
    for i in range(n_bands):
        Cls_bands[0, i] = np.mean(recon_Cls[i*5:(i+1)*5])
        Cls_bands[1, i] = np.mean(theory_noise[i*5:(i+1)*5]) * FSKY / 2.725**2 if convert_K else np.mean(theory_noise[i*5:(i+1)*5]) * FSKY
        Cls_bands[2, i] = np.mean(Clvv[i*5:(i+1)*5])
    plt.figure()
    plt.loglog(ell_bands, Cls_bands[0], label='Reconstruction')
    if convert_K:
        plt.loglog(ell_bands, Cls_bands[1], label='Theory * fsky')
    else:
        plt.loglog(ell_bands, Cls_bands[1], label='Theory * fsky')
    plt.loglog(ell_bands[:2], Cls_bands[2, :2], label='Theory Signal') 
    plt.title(plottitle)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$')
    plt.legend()
    plt.savefig(outdir + filename)
    plt.close('all')
  

Clvv = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]
#Cltaudg = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_taud_g_lmax=6144', dir_base = 'Cls/'+c.direc('taud','g',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]
#Cltaudtaud = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_taud_taud_lmax=6144', dir_base = 'Cls/'+c.direc('taud','taud',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

#if not os.path.exists('./maplist.p'):
#    maplist = Maps()
#    pickle.dump(maplist, open('./maplist.p', 'wb'))
#else:
#    maplist = pickle.load(open('./maplist.p', 'rb'))

maplist = Maps(allmaps=False)
estim = Estimator()



# import camb

# def a(z):
#     return 1/(1+z)

# def ne0():  # 1/m^3
#     chi = 0.86
#     me = 1.14
#     gasfrac = 0.9
#     omgh2 = gasfrac* conf.ombh2
#     ne0_SI = chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me
#     return ne0_SI

# def fe(z):
#     a = 0.475
#     b = 0.703
#     c = 3.19
#     z0 = 5.42
#     return a*(z+b)**0.02 * (1-np.tanh(c*(z-z0)))

# def ne0z(z):  # adapted from eq. 5, 17 in http://arxiv.org/abs/2010.01560
#     return fe(z) * ne0()

# cambpars = camb.CAMBparams()
# cambpars.set_cosmology(H0 = conf.H0, ombh2=conf.ombh2, \
#                                      omch2=conf.omch2, mnu=conf.mnu , \
#                                      omk=conf.Omega_K, tau=conf.tau,  \
#                                      TCMB =2.725 )
# cambpars.InitPower.set_params(As =conf.As*1e-9 ,ns=conf.ns, r=0)
# cambpars.NonLinear = model.NonLinear_both
# cambpars.max_eta_k = 14000.0*conf.ks_hm[-1]
# cambpars.set_matter_power(redshifts=conf.zs_hm.tolist(), kmax=conf.ks_hm[-1],k_per_logint=20)
# cosmology_data = camb.get_background(cambpars)
# bin_width = (cosmology_data.comoving_radial_distance(conf.z_max)-cosmology_data.comoving_radial_distance(conf.z_min))

# def bias_e2(z, k):  # Supposed b^2 = P_ee/P_mm(dmo) bias given in eq. 20, 21. # Verified that b^2 / fe(z)**.5 matches ReCCO em exactly (and squared matches ee exactly).
#     bstar2 = lambda z : 0.971 - 0.013*z
#     gamma = lambda z : 1.91 - 0.59*z + 0.10*z**2
#     kstar = lambda z : 4.36 - 3.24*z + 3.10*z**2 - 0.42*z**3
#     bias_squared = np.zeros((z.size, k.size))
#     for zid, redshift in enumerate(z):
#         bias_squared[zid, :] = bstar2(redshift) / ( 1 + (k/kstar(redshift))**gamma(redshift) ) / fe(redshift)**.5
#     return bias_squared

# galaxy_bias = (0.8+1.2*conf.zs_hm)[:,np.newaxis]
# tau_bias    = (-sigma_T * metres_per_megaparsec * ne0() * a(conf.zs_hm)**-2)
# ee_mm_bias  = bias_e2(conf.zs_hm, conf.ks_hm)

# Pmm_lin_sampled  = camb.get_matter_power_interpolator(cambpars, nonlinear=False, hubble_units=False, k_hunit=False,kmax=conf.ks_hm[-1], zmax=conf.zs_hm[-1]).P(conf.zs_hm, conf.ks_hm, grid=True)
# Pmm_full_sampled  = camb.get_matter_power_interpolator(cambpars, nonlinear=True, hubble_units=False, k_hunit=False,kmax=conf.ks_hm[-1], zmax=conf.zs_hm[-1]).P(conf.zs_hm, conf.ks_hm, grid=True)
# pmm = c.load(c.get_basic_conf(conf), 'p_linear_mm_f1=None_f2 =None', dir_base = 'pks')(conf.ks_hm, conf.zs_hm)
# pme = c.load(c.get_basic_conf(conf), 'p_linear_me_f1=None_f2 =None', dir_base = 'pks')(conf.ks_hm, conf.zs_hm)
# pee = c.load(c.get_basic_conf(conf), 'p_linear_ee_f1=None_f2 =None', dir_base = 'pks')(conf.ks_hm, conf.zs_hm)

# plt.figure()
# plt.loglog(pee[:,0],label='ReCCO')
# plt.loglog(Pmm_lin_sampled[:,0]*bias_e2(conf.zs_hm, conf.ks_hm)[:,0]**2,label='Manual', ls='--')
# plt.legend()
# plt.savefig('P_ee')

# plt.figure()
# plt.loglog(pmm[:,0],label='ReCCO')
# plt.loglog(Pmm_lin_sampled[:,0],label='Manual', ls='--')
# plt.legend()
# plt.savefig('P_mm')

# plt.figure()
# plt.loglog(c.load(c.get_basic_conf(conf), 'p_linear_me_f1=None_f2 =None', dir_base = 'pks')(conf.ks_hm, conf.zs_hm)[:,0],label='ReCCO')
# plt.loglog(np.abs(Pmm_lin_sampled[:,0]*bias_e2(conf.zs_hm, conf.ks_hm)[:,0]),label='Manual', ls='--')
# plt.legend()
# plt.savefig('P_em')

# ells = np.unique(np.append(np.geomspace(1,6143,120).astype(int), 6143))

# Clgg       = np.zeros(ells.size)
# Cltaudg    = np.zeros(ells.size)
# Cltaudtaud = np.zeros(ells.size)

# Pgg_full = interp2d(conf.ks_hm, conf.zs_hm, Pmm_full_sampled * galaxy_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)
# Pgg_lin  = interp2d(conf.ks_hm, conf.zs_hm, Pmm_lin_sampled  * galaxy_bias**2 , kind = 'linear', bounds_error=False, fill_value=0.0)
# Peg_full = interp2d(conf.ks_hm, conf.zs_hm, Pmm_full_sampled * galaxy_bias * ee_mm_bias, kind = 'linear', bounds_error=False, fill_value=0.0)
# Peg_lin  = interp2d(conf.ks_hm, conf.zs_hm, Pmm_lin_sampled  * galaxy_bias * ee_mm_bias, kind = 'linear', bounds_error=False, fill_value=0.0)
# Pee_full = interp2d(conf.ks_hm, conf.zs_hm, Pmm_full_sampled * ee_mm_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)
# Pee_lin  = interp2d(conf.ks_hm, conf.zs_hm, Pmm_lin_sampled  * ee_mm_bias**2, kind = 'linear', bounds_error=False, fill_value=0.0)

# save_fft_weights('g',None)
# save_fft_weights('taud',None)

# for l, ell in enumerate(ells):
#     save_fft('g',None,0,ell)
#     save_fft('taud',None,0,ell)
#     if ell < 30:  # UNWISE: Possible for UNWISE this may have to be flagged to a high or lmax lswitch
#         pee_limb = peg_limb = pgg_limb = None
#     else:   
#         chis_interp = np.linspace(cosmology_data.comoving_radial_distance(1e-2), cosmology_data.comoving_radial_distance(conf.z_max+1.1), 1000)            
#         pee_limb_sample = limber(Pee_full, chis_interp, ell) - limber(Pee_lin, chis_interp, ell)
#         peg_limb_sample = limber(Peg_full, chis_interp, ell) - limber(Peg_lin, chis_interp, ell)
#         pgg_limb_sample = limber(Pgg_full, chis_interp, ell) - limber(Pgg_lin, chis_interp, ell)
#         pee_limb = interp1d(chis_interp, pee_limb_sample, kind='linear', bounds_error=False, fill_value=0.0)
#         peg_limb = interp1d(chis_interp, peg_limb_sample, kind='linear', bounds_error=False, fill_value=0.0)
#         pgg_limb = interp1d(chis_interp, pgg_limb_sample, kind='linear', bounds_error=False, fill_value=0.0)
#     Cltaudg[l] = beyond_limber('taud','g',None,None,0,0,fftlog_integral('taud',None,0,ell)[0],ell, peg_limb)
#     Clgg[l]    = beyond_limber('g','g',   None,None,0,0,fftlog_integral('g',None,0,ell)[0], ell, pgg_limb,) + 9.2e-8
#     Cltaudtaud[l]    = beyond_limber('taud','taud',   None,None,0,0,fftlog_integral('taud',None,0,ell)[0], ell, pee_limb)

# Clgg = interp1d(ells, Clgg, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
# Cltaudg = interp1d(ells, Cltaudg, bounds_error=False, fill_value='extrapolate')(np.arange(6144))
# Cltaudtaud = interp1d(ells, Cltaudtaud, bounds_error=False, fill_value='extrapolate')(np.arange(6144))

# Clgg_recco = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_lmax=6144', dir_base = 'Cls/'+c.direc('g','g',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]
# Cltaudg_recco = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_taud_g_lmax=6144', dir_base = 'Cls/'+c.direc('taud','g',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]
# Cltaudtaud_recco = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_taud_taud_lmax=6144', dir_base = 'Cls/'+c.direc('taud','taud',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

# plt.figure()
# plt.loglog(Clgg_recco, label='ReCCO')
# plt.loglog(Clgg,label='Manual', ls='--')
# plt.legend()
# plt.savefig('gg')
# plt.figure()
# plt.loglog(np.abs(Cltaudg_recco),label='ReCCO')
# plt.loglog(np.abs(Cltaudg), label='Manual', ls='--')
# plt.legend()
# plt.savefig('tg')
# plt.figure()
# plt.loglog(Cltaudtaud_recco,label='ReCCO')
# plt.loglog(Cltaudtaud, label='Manual', ls='--')
# plt.legend()
# plt.savefig('tt')

delta_to_g = maplist.maps['unWISE'].sum() / maplist.maps['unWISE'].size
estim.Clgg *= delta_to_g**2
estim.Cltaudg *= delta_to_g

estim.Cltaudg *= estim.bin_width
estim.Cltaudtaud *= estim.bin_width**2

#plt.figure()
#plt.loglog(Cltaudg**2/Cltaudtaud,label='ReCCO')
#plt.loglog(estim.Cltaudg**2/estim.Cltaudtaud, label='Manual')
#plt.legend()
#plt.title(r"$C_\ell^{\dot{\tau}\mathrm{g}^2}\,/\,C_\ell^{\dot{\tau}\dot{\tau}}$")
#plt.xlabel(r"$\ell$")
#plt.savefig('taudg')


#plt.figure()
#plt.loglog((estim.Cltaudg**2/estim.Cltaudtaud)/(Cltaudg**2/Cltaudtaud))
#plt.xlabel(r"$\ell$")
#plt.savefig('taudg-ratio')


maplist.Cls['unWISE']           = estim.Clgg.copy()
maplist.gaussian_Cls['unWISE']  = estim.Clgg.copy()

test=hp.anafast(maplist.maps['unWISE'])

with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
   lines = FILE.readlines()

alex_ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float')[:-1]
alex_clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')[:-1]
alex_lssspec = interp1d(alex_ells,alex_clgg, bounds_error=False,fill_value='extrapolate')(np.arange(6144))

plt.figure()
plt.loglog(test, label='Mask-region inpainted unWISE map')
plt.loglog(alex_lssspec * delta_to_g**2, label='Alex\'s Clgg')
plt.loglog(maplist.Cls['unWISE'], label='ReCCO Clgg')
plt.loglog(estim.Cltaudg**2/estim.Cltaudtaud, label=r'${C_\ell^{\dot{\tau}\mathrm{g}^2}} / C_\ell^{\dot{\tau}\dot{\tau}}$')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{Xg}}$')
plt.legend()
plt.savefig('cl_1bin')

plt.figure()
plt.loglog(estim.Clgg, label='Clgg')
plt.loglog(estim.Cltaudg**2*delta_to_g**.5/estim.Cltaudtaud, label=r'${C_\ell^{\dot{\tau}\mathrm{g}^2}} / C_\ell^{\dot{\tau}\dot{\tau}}$')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{Xg}}$')
plt.legend()
plt.savefig('cltaudg_vs_clgg')


estim.reconstruct(maplist, 'SMICA', 'unWISE', False, False)
recon_tag = estim.get_recon_tag('SMICA', 'unWISE', False, False)
#twopt(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, 'real data, ReCCO spectra', 'full_recco_lss_taud')
twopt_bandpowers(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, 'real data, ReCCO spectra', 'full_recco_lss_taud_bandpowers')

# estim_matching = Estimator(cltaudg=maplist.Cls['unWISE'])

# estim_matching.reconstruct(maplist, 'SMICA', 'unWISE', False, False)
# twopt(estim_matching.Cls[recon_tag], estim_matching.noises[recon_tag], maplist.fsky, r'real data, ReCCO spectra $C_\ell^{\mathrm{gg}}=C_\ell^{\dot{\tau}\mathrm{g}}$', 'matching_cltaudg_recco_lss_taud')
# twopt_bandpowers(estim_matching.Cls[recon_tag], estim_matching.noises[recon_tag], maplist.fsky, r'real data, ReCCO spectra $C_\ell^{\mathrm{gg}}=C_\ell^{\dot{\tau}\mathrm{g}}$', 'matching_cltaudg_recco_lss_taud_bandpowers')

# from copy import deepcopy
# maplist_mapspec = deepcopy(maplist)
# maplist_mapspec.Cls['unWISE'] = hp.anafast(maplist_mapspec.maps['unWISE'])
# maplist_mapspec.gaussian_Cls['unWISE'] = hp.anafast(maplist_mapspec.gaussian_maps['unWISE'])
# estim_maps = Estimator(cltaudg=maplist_mapspec.Cls['unWISE'])

# estim_maps.reconstruct(maplist_mapspec, 'SMICA', 'unWISE', False, False)
# twopt(estim_maps.Cls[recon_tag], estim_maps.noises[recon_tag], maplist_mapspec.fsky, r'real data, map spectra $C_\ell^{\mathrm{gg}}=C_\ell^{\dot{\tau}\mathrm{g}}$', 'matching_cltaudg_maps_lss_taud')
# twopt_bandpowers(estim_maps.Cls[recon_tag], estim_maps.noises[recon_tag], maplist_mapspec.fsky, r'real data, map spectra $C_\ell^{\mathrm{gg}}=C_\ell^{\dot{\tau}\mathrm{g}}$', 'matching_cltaudg_maps_lss_taud_bandpowers')



# # for Ttag in maplist.map_tags[:-1]:
# #     gtag = maplist.map_tags[-1]
# #     for Tgauss in (True, False):
# #         for ggauss in (True, False):
# #             print('Reconstructing %s x %s  [Tgauss=%s, lssgauss=%s]' % (Ttag, gtag, Tgauss, ggauss))
# #             estim.reconstruct(maplist, Ttag, gtag, Tgauss, ggauss)
# #             recon_tag = estim.get_recon_tag(Ttag, gtag, Tgauss, ggauss)
# #             twopt(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, recon_tag, recon_tag)


# ### Would be nice to do multiple realizations to see the cosmic variance error bars
# ### Might be too big a project, but we could consider what happens if we inpaint without destroying correlations
# #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ### 1) Consider SMICA x unWISE. (If it's about as good as what we see below, we can consider what might improve things. Such as an actual tau filter)
# ### 2) Clgg = Pmm * bias(z) * bias(z) + shotnoise
# ### 3) Cltaug = Pmm * K * bias(z)
# ### Instead of the green input Clgg from the case plots, use these spectra instead. This should be better like Alex's.

# ##
# ### As masking grows what is effect on reconstruction


# ###??????whats this: Prefactor ONLY has the non-inpainted spectrum (alex's line)
