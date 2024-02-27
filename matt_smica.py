import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import scipy
import healpy as hp
import config as conf

plt.switch_backend('agg')

# Define some useful functions

def be(z,chi,ell):
    bstar = np.sqrt(0.971 - 0.013*z)
    gamma = 0.10*z**2 - 0.59*z + 1.91
    kstar = -0.42*z**3 + 3.10*z**2 - 3.24*z + 4.36
    return bstar/np.sqrt(1 + (ell/chi/kstar)**gamma)

def wg(z,chi):
    dndz_in = np.loadtxt('data/unWISE/blue.txt')
    z_dndz = dndz_in[:,0]
    dndz_all = dndz_in[:,1]
    dndz = np.interp(z,z_dndz,dndz_all)
    hs = results.h_of_z(z)
    b_g = 0.8 + 1.2*z
    return dndz*hs*b_g

def wtau(z,chi):
    ne0thom = 5.18e-7 # mean electron number density times Thomson in units of mpc^-1
    return ne0thom*(1+z)**2

def pkratio(z,chi,ellbar_in,zbar_in,chibar_in):
    ratio = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        ratio[i] = PK.P(z[i],ellbar_in/chi[i])
        i=i+1
    return ratio/PK.P(zbar_in,ellbar_in/chibar_in)

def cltaudotg(z_in,chi_in,ell):
    wg_in = wg(z_in,chi_in)
    wtau_in = wtau(z_in,chi_in)
    be_in = be(z_in,chi_in,ell)
    pmm = PK.P(z_in,(ell+.5)/chi_in)
    return wg_in * wtau_in * be_in * pmm / (chi_in**2)

def clggmodel_bin(z_in,chi_in,ell):
    wg_in = wg(z_in,chi_in)
    pmm = PK.P(z_in,(ell+.5)/chi_in)
    return wg_in**2 * pmm / (chi_in**2) 


def reconstruct(cmb_map,cmb_mask,galaxy_map,recon_mask,Ctaug,fwhm,filterlow,filterhigh,premask=False):
    
    nside = hp.get_nside(cmb_map)
    lmax = 3*nside-1
    
    fsky_cmb = np.sum(cmb_mask)/hp.nside2npix(nside)
    fsky_recon = np.sum(recon_mask)/hp.nside2npix(nside)
    
    N_tot = np.sum(recon_mask*galaxy_map)
    n_av = N_tot/np.sum(recon_mask)
        
    gdensity_map = (galaxy_map-n_av)/n_av # input map assumed to be number counts. Convert to overdensity.
        
    Bl = hp.sphtfunc.gauss_beam(fwhm, lmax=lmax, pol=False)
    
    g_pspec = hp.anafast(recon_mask*gdensity_map)/fsky_recon
    t_pspec = hp.anafast(cmb_mask*cmb_map)/fsky_cmb/2.725**2/Bl/Bl  # Note Planck maps in muK, so 2.725 used to convert to dT/T
    
    tfilterspec = 2.725 * Bl * t_pspec
    
    lowhighfilter = np.zeros(lmax+1)
    lowhighfilter[filterlow:filterhigh] = 1.
    
    if premask==False:
        talms = hp.sphtfunc.map2alm(cmb_map)
    else:
        talms = hp.sphtfunc.map2alm(cmb_mask*cmb_map)
    
    cmbfiltered = hp.sphtfunc.almxfl(talms, lowhighfilter/tfilterspec)
    cmbfiltered_map = hp.sphtfunc.alm2map(cmbfiltered,nside)
    
    clfilter = lowhighfilter*Ctaug/g_pspec
    
    galms = hp.sphtfunc.map2alm(gdensity_map)
    gfiltered = hp.sphtfunc.almxfl(galms, clfilter)
    gfiltered_map = hp.sphtfunc.alm2map(gfiltered,nside)
    
    Nrec = 1./np.sum((2.*ells+1)*lowhighfilter*Ctaug**2/t_pspec/g_pspec/(4.*np.pi))
    
    recon = Nrec*cmbfiltered_map*gfiltered_map
    
    return recon, Nrec


#Import maps

smica = hp.read_map("data/planck_data_testing/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")
unwise = hp.read_map("data/unWISE/numcounts_map1_2048-r1-v2_flag.fits")

maskunwise = np.load("data/cache/recon_mask.npy")
maskplnk = hp.read_map("data/planck_data_testing/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")
maskgal = hp.read_map("data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",field=4)
onesmask=np.ones(len(smica))

# Set up the galaxy-tau cross power

nside=2048
kmax=10  #kmax to use


pars = camb.CAMBparams(Evolve_baryon_cs=True)
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_accuracy(AccuracyBoost=2)

results = camb.get_background(pars)

PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
    hubble_units=False, k_hunit=False, kmax=kmax,
    var1=model.Transfer_tot,var2=model.Transfer_tot, zmax=10,k_per_logint=200)

nz = 500
chis = np.linspace(0.001,5000,nz)
chis = np.linspace(0.001,5000,nz)
zs=results.redshift_at_comoving_radial_distance(chis)
hs = results.h_of_z(zs)
ells = np.arange(3*nside)

ibar=250
zbar=zs[ibar]
chibar=chis[ibar]

ellbar=2000

# Velocity window function

wv = (chibar**2 / chis**2) * (wg(zs,chis)/wg(zbar,chibar)) * ( (1+zs)**2 / (1+zbar)**2 )* (be(zs,chis,ellbar) / be(zbar,chibar,ellbar)) * pkratio(zs,chis,ellbar,zbar,chibar)

dchi=np.trapz(wv,chis)
wv=wv/dchi

Ctaug = cltaudotg(zbar,chibar,ells)*dchi

fsky_recon = np.sum(maskunwise)/hp.nside2npix(nside)
fsky_cmb = np.sum(maskplnk)/hp.nside2npix(nside)

# SMICA/unwise mocks

N_tot = np.sum(maskunwise*unwise)
n_av = N_tot/np.sum(maskunwise)

gdensity_map = (unwise-n_av)/n_av # input map assumed to be number counts. Convert to overdensity.

gdensity_masked = hp.ma(gdensity_map)
gdensity_masked.mask = np.logical_not(maskunwise)

unwise_pspec = hp.anafast(hp.pixelfunc.remove_dipole(gdensity_masked))/fsky_recon

unwisemock = n_av*hp.synfast(unwise_pspec,2048)+n_av

deltamock = (unwisemock-n_av)/n_av

smica_pspec = hp.anafast(maskplnk*smica)/fsky_cmb

smica_mock = hp.synfast(smica_pspec,2048)

deltamock_alm = hp.map2alm(deltamock,use_pixel_weights=True,iter=5)

vspec = 6e-9/np.arange(len(smica_pspec))**2.1

vspec[0]=0

v_mock = hp.synfast(vspec,2048)

highfilter = np.ones(len(vspec))
highfilter[0:20] = 0.

tksz_mock = v_mock*hp.alm2map(hp.sphtfunc.almxfl(deltamock_alm,highfilter*Ctaug/unwise_pspec),nside)


# Do reconstruction without pre-masking on full sky

reconmap, Nrec = reconstruct(smica,maskgal*maskplnk,unwise,maskunwise,Ctaug,.00145,500,4000,premask=False)

reconmap_mock, Nrec_mock = reconstruct(smica_mock+tksz_mock,maskgal*maskplnk,unwisemock,maskunwise,Ctaug,.00145,500,4000,premask=False)

reconmap_pre, Nrec_pre = reconstruct(smica,maskgal*maskplnk,unwise,maskunwise,Ctaug,.00145,500,4000,premask=True)

reconmap_mock_pre, Nrec_mock_pre = reconstruct(smica_mock+tksz_mock,maskgal*maskplnk,unwisemock,maskunwise,Ctaug,.00145,500,4000,premask=True)


#reconmap_mock_noise, Nrec_mock_noise = reconstruct(smica_mock,maskplnk,unwisemock,maskunwise,Ctaug,.00145,500,4000,premask=False)

recon_masked = hp.ma(reconmap)
recon_masked.mask = np.logical_not(maskunwise)

mock_masked = hp.ma(reconmap_mock)
mock_masked.mask = np.logical_not(maskunwise)

recon_masked_pre = hp.ma(reconmap_pre)
recon_masked_pre.mask = np.logical_not(maskunwise)

mock_masked_pre = hp.ma(reconmap_mock_pre)
mock_masked_pre.mask = np.logical_not(maskunwise)


total = hp.anafast(hp.remove_dipole(mock_masked))
total_real = hp.anafast(hp.remove_dipole(recon_masked))
total_pre = hp.anafast(hp.remove_dipole(mock_masked_pre))
total_real_pre = hp.anafast(hp.remove_dipole(recon_masked_pre))

fsky_recon = np.sum(maskunwise)/hp.nside2npix(nside)



plt.figure()
plt.semilogy(total/fsky_recon,'k')
plt.semilogy(Nrec*np.ones(len(total)),'b--')
#plt.semilogy(Nrec_mock*np.ones(len(total)),'r--')
#plt.semilogy(total/fsky_recon,'b')
#plt.semilogy(total_pre/fsky_recon,'r')

plt.xlim(2,150)
plt.ylim(1e-9,2e-8)
plt.savefig('recon_Cls_SMICA')



bandpowers = lambda spectrum : np.array([spectrum[2:][1+(5*i):1+(5*(i+1))].mean() for i in np.arange(spectrum.size//5)])
x_ells = bandpowers(np.arange(total.size))

plt.figure()
plt.semilogy(x_ells, bandpowers(total)/fsky_recon,'k')
plt.semilogy(x_ells, Nrec_mock*np.ones(x_ells.size),'b--')
#plt.semilogy(Nrec_mock*np.ones(len(total)),'r--')
#plt.semilogy(total/fsky_recon,'b')
plt.semilogy(x_ells, bandpowers(total_pre)/fsky_recon,'r')

plt.xlim(2,150)
plt.ylim(1e-9,2e-8)
plt.savefig('recon_Cls_bandpower_smica')


mono_pre, dipo_pre = hp.pixelfunc.fit_dipole(-recon_masked_pre)
print(mono_pre)
print(np.sqrt(dipo_pre[0]**2+dipo_pre[1]**2+dipo_pre[2]**2))
print(hp.vec2ang(dipo_pre,lonlat=True))


mono, dipo = hp.pixelfunc.fit_dipole(-recon_masked)
print(mono)
print(np.sqrt(dipo[0]**2+dipo[1]**2+dipo[2]**2))
print(hp.vec2ang(dipo,lonlat=True))

reconmapsmooth=hp.sphtfunc.smoothing(-reconmap,fwhm=.1,use_pixel_weights=True,iter=5)

reconmapsmooth_masked=hp.ma(reconmapsmooth)
reconmapsmooth_masked.mask=np.logical_not(maskunwise)

plt.figure()
hp.mollview(reconmapsmooth,max=0.0018,min=-.0011)
plt.savefig('reconmapsmooth_smica')

plt.figure()
hp.mollview(reconmapsmooth_masked)
plt.savefig('reconmapsmooth_masked_smica')

plt.figure()
hp.mollview(hp.sphtfunc.smoothing(-recon_masked,fwhm=.1,use_pixel_weights=True,iter=5))
plt.savefig('sphtfunc_recon_masked_smica')
