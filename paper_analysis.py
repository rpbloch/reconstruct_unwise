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
        print('Masking and debeaming maps')
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
    def get_limber_window(self, tag, avg=False):
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
            galaxy_bias = (0.8+0.65*self.chi_to_z(chis))  # Changed from 0.8 + 1.2z to better fit inpainted unWISE map spectrum
            window = galaxy_bias * interp1d(z ,dndz, kind= 'linear', bounds_error=False, fill_value=0)(self.chi_to_z(chis)) * self.cosmology_data.h_of_z(self.chi_to_z(chis))
        elif tag == 'taud':
            window = (-thompson_SI * self.ne0() * (1+self.chi_to_z(chis))**2 * m_per_Mpc)
        if avg:
            return simps(window, chis)
        else:
            return window
    def compute_Cls(self, ngbar):
        chis = self.sample_chis(1000)
        Pmm_full = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True,  hubble_units=False, k_hunit=False, kmax=conf.ks_hm[-1], zmax=conf.zs_hm[-1])
        ells = np.unique(np.append(np.geomspace(1,6143,120).astype(int), 6143))
        matter_window = self.get_limber_window('m')
        galaxy_window = self.get_limber_window('g')
        taud_window   = self.get_limber_window('taud')
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
    def get_recon_tag(self, Ttag, gtag, Tgauss, ggauss, Cltaudgtag):
        return Ttag + '_gauss=' + str(Tgauss) + '__' + gtag + '_gauss=' + str(ggauss) + '__Cltaudg~Clgg=' + str(Cltaudgtag)
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
  
# Clvv = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

## We want 16 cases: 8 with the hybrid Clgg and 8 with the unmodified Clgg. Each set of 8 contains 4 plots of two panels.
## 4 plots corresponding to the (True, False) Ttag and gtag combinations, and each panel showing the same combination
## but one has Cltaudg~Clgg and the other has the unmodified Cltaudg.
## This way we can see if there is any advantage or issue using the modified Clgg, and we can also see what effect
## a modeled Cltaudg has on our predictions.
## Currently for the hybrid case we have an issue when real T is used that the theory noise is too high, when Cltaudg~Clgg.
## This shouldn't be the case since the hybrid Clgg (if it affected the noise floor) would show this in every combination.
if not os.path.exists('./maplist.p'):
    maplist = Maps(allmaps=False)
    pickle.dump(maplist, open('./maplist.p', 'wb'))
else:
    maplist = pickle.load(open('./maplist.p', 'rb'))

estim = Estimator()
csm = Cosmology()

delta_to_g = maplist.maps['unWISE'].sum() / maplist.maps['unWISE'].size

csm.compute_Cls(ngbar=delta_to_g)

with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
   lines = FILE.readlines()

alex_ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float')[:-1]
alex_clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')[:-1]
alex_lssspec = interp1d(alex_ells,alex_clgg, bounds_error=False,fill_value='extrapolate')(np.arange(6144)) * delta_to_g**2

plt.figure()
plt.loglog(maplist.Cls['unWISE'], label='Map Cls')
plt.loglog(alex_lssspec, label='Alex\'s Clgg')
plt.loglog(csm.Clgg, label='Clgg')
plt.loglog(csm.Cltaudg**2/csm.Cltaudtaud, label=r'${C_\ell^{\dot{\tau}\mathrm{g}^2}} / C_\ell^{\dot{\tau}\dot{\tau}}$', ls='--')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{Xg}}$')
plt.ylim([plt.ylim()[0], 0.02])
plt.legend()
plt.savefig('computed_spectra')

plt.figure()
plt.loglog(maplist.Cls['unWISE'], label='Map Cls')
plt.loglog(csm.Clgg, label='Clgg')
plt.loglog(csm.Clmm*delta_to_g**2, label='Clmm (scaled to gg)')
#plt.loglog((csm.Clmm*delta_to_g/csm.get_limber_window('g', avg=True))**2 / csm.Clmm,label='Clmm (scaled to tg')
#plt.loglog(csm.Cltaudg**2/csm.Cltaudtaud, label=r'${C_\ell^{\dot{\tau}\mathrm{g}^2}} / C_\ell^{\dot{\tau}\dot{\tau}}$', ls='--')
plt.ylim([plt.ylim()[0], 0.02])
plt.legend()
plt.savefig('mm_gg')

for Tgauss in (True, False):
    for ggauss in (True, False):
        Cltaudgtag = False
        recon_tag = estim.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, Cltaudgtag)
        estim.reconstruct(maplist, 'SMICA', 'unWISE', Tgauss, ggauss, taudgcl=csm.Cltaudg)
        twopt_bandpowers(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, 'SMICA x unWISE   [%s x %s]'%('real' if not Tgauss else 'gauss', 'real' if not ggauss else 'gauss'), 'recon_SMICAxunWISE_%sx%s'%('real' if not Tgauss else 'gauss', 'real' if not ggauss else 'gauss'),lmaxplot=4000)

estim_gg = Estimator()
csm.Clmm_as_gg = csm.Clmm*delta_to_g**2
csm.Clgg_hybrid = np.where(csm.Clmm_as_gg > csm.Clgg, csm.Clmm_as_gg, csm.Clgg)
csm.Cltaudg_hybrid = ((csm.Clgg_hybrid/delta_to_g**2) - 9.2e-8) * csm.get_limber_window('taud',avg=True) * delta_to_g / csm.get_limber_window('g',avg=True)
csm.Cltaudtaud_hybrid = ((csm.Clgg_hybrid/delta_to_g**2)-9.2e-8)*csm.get_limber_window('taud',avg=True)**2/csm.get_limber_window('g',avg=True)**2

plt.figure()
plt.loglog(maplist.Cls['unWISE'], label='Map')
plt.loglog(csm.Clgg_hybrid, label='hybrid Clgg')
plt.legend()
plt.ylim([plt.ylim()[0], 0.02])
plt.savefig('mm_gg_hybrid')

plt.figure()
plt.loglog(maplist.Cls['unWISE'], label='Map Cls')
plt.loglog(csm.Clgg_hybrid, label='Clgg')
plt.loglog(csm.Cltaudg_hybrid**2/csm.Cltaudtaud_hybrid, label=r'${C_\ell^{\dot{\tau}\mathrm{g}^2}} / C_\ell^{\dot{\tau}\dot{\tau}}$', ls='--')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{Xg}}$')
plt.ylim([plt.ylim()[0], 0.02])
plt.legend()
plt.savefig('computed_spectra_hybrid')


maplist2 = pickle.load(open('./maplist.p', 'rb'))
Tgauss = ggauss = True
recon_tag = estim_gg.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss)

estim_gg.reconstruct(maplist2, 'SMICA', 'unWISE', Tgauss, ggauss, taudgcl=csm.Cltaudg_hybrid)
twopt_bandpowers(estim_gg.Cls[recon_tag], estim_gg.noises[recon_tag], maplist2.fsky, 'SMICA x unWISE   [gauss x gauss]', 'recon_SMICAxunWISE_gaussxgauss_tg=gg',lmaxplot=4000)


for Tgauss in (True, False):
    for ggauss in (True, False):
        Cltaudgtag = True
        recon_tag = estim_gg.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, Cltaudgtag)
        estim_gg.reconstruct(maplist2, 'SMICA', 'unWISE', Tgauss, ggauss, taudgcl=maplist2.Cls['unWISE']*csm.get_limber_window('taud',avg=True)*csm.bin_width/delta_to_g)
        twopt_bandpowers(estim_gg.Cls[recon_tag], estim_gg.noises[recon_tag], maplist2.fsky, 'SMICA x unWISE   [%s x %s]'%('real' if not Tgauss else 'gauss', 'real' if not ggauss else 'gauss'), 'recon_SMICAxunWISE_%sx%s_tg=gg'%('real' if not Tgauss else 'gauss', 'real' if not ggauss else 'gauss'),lmaxplot=4000)

# for Ttag in maplist.map_tags[:-1]:
#     gtag = maplist.map_tags[-1]
#     for Tgauss in (True, False):
#         for ggauss in (True, False):
#             print('Reconstructing %s x %s  [Tgauss=%s, lssgauss=%s]' % (Ttag, gtag, Tgauss, ggauss))
#             estim.reconstruct(maplist, Ttag, gtag, Tgauss, ggauss)
#             recon_tag = estim.get_recon_tag(Ttag, gtag, Tgauss, ggauss)
#             twopt(estim.Cls[recon_tag], estim.noises[recon_tag], maplist.fsky, recon_tag, recon_tag)


