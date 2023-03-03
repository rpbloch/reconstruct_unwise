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

outdir = 'plots/analysis/paper_analysis_latest/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

Clvv = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_vr_vr_lmax=6144', dir_base = 'Cls/'+c.direc('vr','vr',conf)), c.load(c.get_basic_conf(conf),'L_sample_lmax=6144', dir_base = 'Cls'))[:6144,0,0]

unWISEmap = fits.open('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits')[1].data['T'].flatten()
ngbar = unWISEmap.sum() / unWISEmap.size

gcl_full = hp.anafast(unWISEmap)
gmap_full_gauss = hp.synfast(gcl_full, 2048)
gmap_full_real = unWISEmap.copy()

mask_map = np.load('data/mask_unWISE_thres_v10.npy')
fsky = np.where(mask_map!=0)[0].size / mask_map.size

SMICAinp = hp.reorder(fits.open('data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True)
noisegen = hp.synfast(np.repeat([hp.anafast(SMICAinp, lmax=2500)[-1]], 6144), 2048)
SMICAmap_gauss_unmodified = hp.synfast(hp.anafast(SMICAinp + noisegen),2048)
SMICAmap_real_unmodified = SMICAinp + noisegen

SMICAbeam = hp.gauss_beam(fwhm=np.radians(5/60), lmax=6143)
SMICAmap_gauss = hp.alm2map(hp.almxfl(hp.map2alm(SMICAmap_gauss_unmodified), 1/SMICAbeam), 2048)
SMICAmap_real = hp.alm2map(hp.almxfl(hp.map2alm(SMICAmap_real_unmodified), 1/SMICAbeam), 2048)
ClTT = hp.anafast(SMICAmap_real)

estim = Estimator()
csm = Cosmology()
csm.compute_Cls(ngbar=ngbar)

Tmaps = { True : SMICAmap_gauss, False : SMICAmap_real}
gmaps = { True : gmap_full_gauss, False : gmap_full_real}
for Tgauss in (True, False):
    for ggauss in (True, False):
        recon_tag = estim.get_recon_tag('SMICA', 'unWISE', Tgauss, ggauss, notes='postmask_unWISE')
        estim.reconstruct(Tmaps[Tgauss], gmaps[ggauss], ClTT, csm.Clgg, mask_map, csm.Cltaudg, recon_tag)

recon_tags = [estim.get_recon_tag('SMICA', 'unWISE', tg, gg, notes='postmask_unWISE') for tg in (True, False) for gg in (True, False)]
gauss_tags = ['T=%s x lss=%s' % (tg,gg) for tg in ('gauss','real') for gg in ('gauss','real')]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(12,12))
for i, ax in zip(np.arange(4), (ax1, ax2, ax3, ax4)):
    ax.loglog(estim.Cls[recon_tags[i]], lw=2)
    ax.loglog(estim.noises[recon_tags[i]] * fsky / 2.725**2, lw=2)
    ax.loglog(Clvv[:10])
    ax.set_title('SMICA x unWISE  [%s]' % (gauss_tags[i]), fontsize=16)
    if i > 1:
        ax.set_xlabel(r'$\ell$', fontsize=16)
    if i % 2 == 0:
        ax.set_ylabel(r'$N_\ell^{\mathrm{vv}}\ \left[v^2/c^2\right]$', fontsize=16)
fig.suptitle('SMICA x unWISE Gaussian vs Real Inputs', fontsize=22)
plt.tight_layout()
plt.savefig(outdir + 'SMICAxunWISE_real_vs_gauss.png')
