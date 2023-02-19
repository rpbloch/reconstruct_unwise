#bessel = lambda sg, sT : sp.kn(0,np.abs(np.linspace(synfast64.min(),synfast64.max(),100))/(sg*sT))/(sg*sT*np.pi)
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

basic_conf_dir = c.get_basic_conf(conf)
estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(conf, exclude = False))+'/T_freq='+str(143)+'/'  
bin_width = 525.7008226572364
binning = 10000  # Number of bins to use
print('Setup....')
### Loading Cls
load = lambda tag1, tag2 : loginterp.log_interpolate_matrix(c.load(basic_conf_dir,'Cl_'+tag1+'_'+tag2+'_lmax=6144', dir_base = 'Cls/'+c.direc(tag1,tag2,conf)), c.load(basic_conf_dir,'L_sample_lmax=6144', dir_base = 'Cls'))

with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
    lines = FILE.readlines()

ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float')
clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')

Tmap11 = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/143ghz/ffp10_noise_143_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
Tcl = hp.anafast(Tmap11)


ClTT = np.append(Tcl, Tcl[-1])
Cldd = interp1d(ells,clgg, bounds_error=False,fill_value='extrapolate')(np.arange(6145))
Cltaudd = load('taud','g')[:,0,0]*bin_width
vrmap = c.load(basic_conf_dir, 'vr_full_fine_2048_64_real=0_bin=0', dir_base=estim_dir+'sims')
vrn, vrbins, vrpatches = plt.hist(vrmap, bins=binning)

ClTT[:100] = 1e15
Cldd[:100] = 1e15

outdir = 'plots/analysis/plots_%dx%d/Tmap_prog/binning_%d/'% (8,32,binning)
if not os.path.exists(outdir):
    os.makedirs(outdir)
### Load and filter maps
lssmap = c.load(basic_conf_dir, 'lss_vr_2048_real=0_bin=0', dir_base=estim_dir+'sims')[0,:]
hp.mollview(lssmap,title='Gaussian Realization of unWISE gg Spectrum')
plt.savefig(outdir+'lssmap')

dlm = hp.map2alm(lssmap)
dlm_zeta = hp.almxfl(dlm, np.divide(Cltaudd, Cldd, out=np.zeros_like(Cltaudd), where=Cldd!=0))
gfilt = hp.alm2map(dlm_zeta, 2048)

ng, binsg, patchesg = plt.hist(lssmap, bins=binning)
ngfilt, binsgfilt, patchesgfilt = plt.hist(gfilt, bins=binning)

def do(Tmap):
    outmap = Tmap*lssmap
    dTlm = hp.map2alm(Tmap)
    dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
    Tfilt = hp.alm2map(dTlm_xi, 2048)
    filtered_outmap = hp.alm2map(dTlm_xi, 2048)*hp.alm2map(dlm_zeta,2048)
    return outmap, filtered_outmap, Tfilt



print("Running synfast...")
##### Plot maps and histograms
synfast_Tmap = hp.synfast(Tcl,2048)
hp.mollview(synfast_Tmap,title='Gaussian Realization of Planck CMB Noise')
plt.savefig(outdir+'synfast_Tmap')

synfast_outmap, synfast_filtered_outmap, synfast_Tfilt = do(synfast_Tmap)
synfast_nT, synfast_binsT, synfast_patchesT = plt.hist(synfast_Tmap, bins=binning)

synfast_nTfilt, synfast_binsTfilt, synfast_patchesTfilt = plt.hist(synfast_Tfilt, bins=binning)

plt.figure()

l1, = plt.plot(synfast_nT/simps(synfast_nT),label='Temperature',drawstyle='steps')
plt.plot(synfast_nTfilt/simps(synfast_nTfilt), label='Temperature (filtered)', drawstyle='steps', ls='--',c=l1.get_c())
l1, = plt.plot(ng/simps(ng),label='LSS',drawstyle='steps')
plt.plot(ngfilt/simps(ngfilt), label='LSS (filtered)', drawstyle='steps', ls='--',c=l1.get_c())

plt.title('Pixel Distribution of Input Maps')
plt.xlabel('Bin Number')
plt.ylabel('Normalized # of Pixels')

plt.legend()
plt.savefig(outdir+'synfast_hists')


synfast_nout, synfast_binsout, synfast_patchesout = plt.hist(synfast_outmap, bins=binning)
synfast_noutfilt, synfast_binsoutfilt, synfast_patchesoutfilt = plt.hist(synfast_filtered_outmap, bins=binning)

synfast_norm_prod_out = kn(0, np.abs(np.linspace((synfast_binsout[1]+synfast_binsout[0])/2,(synfast_binsout[-1]+synfast_binsout[-2])/2,binning))/(np.std(synfast_Tmap)*np.std(lssmap))) \
    / (np.pi * np.std(synfast_Tmap) * np.std(lssmap))

mismatch = synfast_nout.max() / synfast_norm_prod_out.max()
plt.figure()
plt.plot((synfast_binsout[1:]+synfast_binsout[:-1])/2, synfast_nout/mismatch, label='Map Pixels')
plt.plot((synfast_binsout[1:]+synfast_binsout[:-1])/2, synfast_norm_prod_out, ls='--', label='Normal Product Distribution')

plt.title('Pixel Distribution of Output Map')
plt.xlabel('v/c')
plt.ylabel('# of Pixels')
plt.xlim([((synfast_binsout[1:]+synfast_binsout[:-1])/2)[np.where((synfast_norm_prod_out/mismatch)>0.01*(synfast_norm_prod_out/mismatch).max())][0],((synfast_binsout[1:]+synfast_binsout[:-1])/2)[np.where((synfast_norm_prod_out/mismatch)>0.01*(synfast_norm_prod_out/mismatch).max())][-1]])
plt.legend()
plt.savefig(outdir+'synfast_normprod')





print("Running noise...")
noise_Tmap = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/143ghz/ffp10_noise_143_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten()
hp.mollview(noise_Tmap,title='Simulated Planck CMB Noise')
plt.savefig(outdir+'noise_Tmap')

noise_outmap, noise_filtered_outmap, noise_Tfilt = do(noise_Tmap)
noise_nT, noise_binsT, noise_patchesT = plt.hist(noise_Tmap, bins=binning)

noise_nTfilt, noise_binsTfilt, noise_patchesTfilt = plt.hist(noise_Tfilt, bins=binning)

plt.figure()

l1, = plt.plot(noise_nT/simps(noise_nT),label='Temperature',drawstyle='steps')
plt.plot(noise_nTfilt/simps(noise_nTfilt), label='Temperature (filtered)', drawstyle='steps', ls='--',c=l1.get_c())
l1, = plt.plot(ng/simps(ng),label='LSS',drawstyle='steps')
plt.plot(ngfilt/simps(ngfilt), label='LSS (filtered)', drawstyle='steps', ls='--',c=l1.get_c())

plt.title('Pixel Distribution of Input Maps')
plt.xlabel('Bin Number')
plt.ylabel('Normalized # of Pixels')

plt.legend()
plt.savefig(outdir+'noise_hists')


noise_nout, noise_binsout, noise_patchesout = plt.hist(noise_outmap, bins=binning)
noise_noutfilt, noise_binsoutfilt, noise_patchesoutfilt = plt.hist(noise_filtered_outmap, bins=binning)

noise_norm_prod_out = kn(0, np.abs(np.linspace((noise_binsout[1]+noise_binsout[0])/2,(noise_binsout[-1]+noise_binsout[-2])/2,binning))/(np.std(noise_Tmap)*np.std(lssmap))) \
    / (np.pi * np.std(noise_Tmap) * np.std(lssmap))

mismatch = noise_nout.max() / noise_norm_prod_out.max()
plt.figure()
plt.plot((noise_binsout[1:]+noise_binsout[:-1])/2, noise_nout/mismatch, label='Map Pixels')
plt.plot((noise_binsout[1:]+noise_binsout[:-1])/2, noise_norm_prod_out, ls='--', label='Normal Product Distribution')

plt.title('Pixel Distribution of Output Map')
plt.xlabel('v/c')
plt.ylabel('# of Pixels')
plt.xlim([((noise_binsout[1:]+noise_binsout[:-1])/2)[np.where((noise_norm_prod_out/mismatch)>0.01*(noise_norm_prod_out/mismatch).max())][0],((noise_binsout[1:]+noise_binsout[:-1])/2)[np.where((noise_norm_prod_out/mismatch)>0.01*(noise_norm_prod_out/mismatch).max())][-1]])
plt.legend()
plt.savefig(outdir+'noise_normprod')








print("Running noise w/ mask....")
mask_map = np.load('data/mask_unWISE_thres_v10.npy')
hp.mollview(mask_map)
plt.savefig(outdir+'mask')
### Load and filter maps
ng_mask, binsg_mask, patchesg_mask = plt.hist((lssmap*mask_map)[np.where(mask_map!=0)], bins=binning)
dlm_mask = hp.map2alm(lssmap*mask_map)
dlm_zeta_mask = hp.almxfl(dlm_mask, np.divide(Cltaudd, Cldd, out=np.zeros_like(Cltaudd), where=Cldd!=0))
gfilt_mask = hp.alm2map(dlm_zeta_mask, 2048)
ngfilt_mask, binsgfilt_mask, patchesgfilt_mask = plt.hist((gfilt_mask)[np.where(mask_map!=0)], bins=binning)



noise_Tmap_mask = fits.open('/home/richard/Desktop/ReCCO/code/data/planck_data_testing/sims/noise/143ghz/ffp10_noise_143_full_map_mc_00000.fits')[1].data['I_STOKES'].flatten() * mask_map
hp.mollview(noise_Tmap_mask,title='Simulated Planck CMB Noise')
plt.savefig(outdir+'noise_Tmap_mask')

noise_outmap_mask = noise_Tmap_mask*lssmap*mask_map
dTlm = hp.map2alm(noise_Tmap_mask)
dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
noise_Tfilt_mask = hp.alm2map(dTlm_xi, 2048)
noise_filtered_outmap_mask = hp.alm2map(dTlm_xi, 2048)*hp.alm2map(dlm_zeta_mask,2048)

#noise_outmap_mask, noise_filtered_outmap_mask, noise_Tfilt_mask = do(noise_Tmap_mask)
noise_nT_mask, noise_binsT_mask, noise_patchesT_mask = plt.hist(noise_Tmap_mask[np.where(mask_map!=0)], bins=binning)

noise_nTfilt_mask, noise_binsTfilt_mask, noise_patchesTfilt_mask = plt.hist(noise_Tfilt_mask[np.where(mask_map!=0)], bins=binning)

plt.figure()

l1, = plt.plot(noise_nT_mask/simps(noise_nT_mask),label='Temperature',drawstyle='steps')
plt.plot(noise_nTfilt_mask/simps(noise_nTfilt_mask), label='Temperature (filtered)', drawstyle='steps', ls='--',c=l1.get_c())
l1, = plt.plot(ng_mask/simps(ng_mask),label='LSS',drawstyle='steps')
plt.plot(ngfilt_mask/simps(ngfilt_mask), label='LSS (filtered)', drawstyle='steps', ls='--',c=l1.get_c())

plt.title('Pixel Distribution of (Masked) Input Maps')
plt.xlabel('Bin Number')
plt.ylabel('Normalized # of Pixels')

plt.legend()
plt.savefig(outdir+'noise_hists_mask')


noise_nout_mask, noise_binsout_mask, noise_patchesout_mask = plt.hist(noise_outmap_mask[np.where(mask_map!=0)], bins=binning)
noise_noutfilt_mask, noise_binsoutfilt_mask, noise_patchesoutfilt_mask = plt.hist(noise_filtered_outmap_mask[np.where(mask_map!=0)], bins=binning)

noise_norm_prod_out_mask = kn(0, np.abs(np.linspace((noise_binsout_mask[1]+noise_binsout_mask[0])/2,(noise_binsout_mask[-1]+noise_binsout_mask[-2])/2,binning))/(np.std(noise_Tmap_mask[np.where(mask_map!=0)])*np.std(lssmap[np.where(mask_map!=0)]))) \
    / (np.pi * np.std(noise_Tmap_mask[np.where(mask_map!=0)]) * np.std(lssmap[np.where(mask_map!=0)]))
noise_norm_prod_out_mask_filt = kn(0, np.abs(np.linspace((noise_binsoutfilt_mask[1]+noise_binsoutfilt_mask[0])/2,(noise_binsoutfilt_mask[-1]+noise_binsoutfilt_mask[-2])/2,binning))/(np.std(noise_Tmap_mask[np.where(mask_map!=0)])*np.std(lssmap[np.where(mask_map!=0)]))) \
    / (np.pi * np.std(noise_Tmap_mask[np.where(mask_map!=0)]) * np.std(lssmap[np.where(mask_map!=0)]))

mismatch1 = noise_norm_prod_out_mask.max() / noise_nout_mask.max()
mismatch2 = noise_norm_prod_out_mask_filt.max() / noise_nout_mask.max()
plt.figure()
plt.plot((noise_binsout_mask[1:]+noise_binsout_mask[:-1])/2, noise_nout_mask, label='Map Pixels')
l1,=plt.plot((noise_binsout_mask[1:]+noise_binsout_mask[:-1])/2, noise_norm_prod_out_mask/mismatch1, label='Normal Product Distribution',ls='--')
#plt.plot((noise_binsoutfilt_mask[1:]+noise_binsoutfilt_mask[:-1])/2, noise_norm_prod_out_mask_filt/mismatch2, label='Normal Product Distribution (Filtered)', c=l1.get_c(), ls='--')

plt.title('Pixel Distribution of Output Map from Masked Inputs')
plt.xlabel('v/c')
plt.ylabel('# of Pixels')
plt.xlim([((noise_binsout_mask[1:]+noise_binsout_mask[:-1])/2)[np.where((noise_norm_prod_out_mask/mismatch1)>0.01*(noise_norm_prod_out_mask/mismatch1).max())][0],((noise_binsout_mask[1:]+noise_binsout_mask[:-1])/2)[np.where((noise_norm_prod_out_mask/mismatch1)>0.01*(noise_norm_prod_out_mask/mismatch1).max())][-1]])
plt.legend()
plt.savefig(outdir+'noise_normprod_mask')













Tmap_143 = hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'], n2r=True)
hp.mollview(Tmap_143,title='Planck 143GHz Sky',norm='hist')
plt.savefig(outdir+'Tmap_143')

ClTT143 = hp.anafast(Tmap_143)
ClTT143 = np.append(ClTT143, ClTT143[-1])
ClTT143[:100] = 1e15

output_map_143 = lssmap*Tmap_143
dTlm = hp.map2alm(output_map_143)
dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT143.size), ClTT143, out=np.zeros_like(np.ones(ClTT143.size)), where=ClTT143!=0))
Tfilt_143 = hp.alm2map(dTlm_xi, 2048)
filtered_outmap_143 = hp.alm2map(dTlm_xi, 2048)*hp.alm2map(dlm_zeta,2048)

nT_143, binsT_143, patchesT_143 = plt.hist(Tmap_143, bins=binning)

nTfilt_143, binsTfilt_143, patchesTfilt_143 = plt.hist(Tfilt_143, bins=binning)

plt.figure()

l1, = plt.plot(nT_143/simps(nT_143),label='Temperature',drawstyle='steps')
plt.plot(nTfilt_143/simps(nTfilt_143), label='Temperature (filtered)', drawstyle='steps', ls='--',c=l1.get_c())
l1, = plt.plot(ng/simps(ng),label='LSS',drawstyle='steps')
plt.plot(ngfilt/simps(ngfilt), label='LSS (filtered)', drawstyle='steps', ls='--',c=l1.get_c())

plt.title('Pixel Distribution of Input Maps')
plt.xlabel('Bin Number')
plt.ylabel('Normalized # of Pixels')

plt.legend()
plt.savefig(outdir+'143_hists')


nout_143, binsout_143, patchesout_143 = plt.hist(output_map_143, bins=binning)
noutfilt_143, binsoutfilt_143, patchesoutfilt_143 = plt.hist(filtered_outmap_143, bins=binning)

norm_prod_out_143 = kn(0, np.abs(np.linspace((binsout_143[1]+binsout_143[0])/2,(binsout_143[-1]+binsout_143[-2])/2,binning))/(np.std(Tmap_143)*np.std(lssmap))) \
    / (np.pi * np.std(Tmap_143) * np.std(lssmap))

mismatch = nout_143.max() / norm_prod_out_143.max()
plt.figure()
plt.plot((binsout_143[1:]+binsout_143[:-1])/2, nout_143/mismatch, label='Map Pixels')
plt.plot((binsout_143[1:]+binsout_143[:-1])/2, norm_prod_out_143, ls='--', label='Normal Product Distribution')
plt.plot((vrbins[1:]+vrbins[:-1])/2, vrn / (vrn.max()/norm_prod_out_143.max()), label='Theory Velocity Field')

plt.title('Pixel Distribution of Output Map')
plt.xlabel('v/c')
plt.ylabel('# of Pixels')
plt.xlim([((binsout_143[1:]+binsout_143[:-1])/2)[np.where((norm_prod_out_143/mismatch)>0.01*(norm_prod_out_143/mismatch).max())][0],((binsout_143[1:]+binsout_143[:-1])/2)[np.where((norm_prod_out_143/mismatch)>0.01*(norm_prod_out_143/mismatch).max())][-1]])
plt.legend()
plt.savefig(outdir+'normprod_143')
















Tmap_143_mask = Tmap_143 * mask_map
hp.mollview(Tmap_143_mask,title='Masked Planck 143GHz Sky')
plt.savefig(outdir+'Tmap_143_mask')

output_map_143_mask = Tmap_143_mask*lssmap*mask_map
dTlm = hp.map2alm(Tmap_143_mask)
dTlm_xi = hp.almxfl(dTlm, np.divide(np.ones(ClTT.size), ClTT, out=np.zeros_like(np.ones(ClTT.size)), where=ClTT!=0))
Tfilt_143_mask = hp.alm2map(dTlm_xi, 2048)
filtered_outmap_143_mask = hp.alm2map(dTlm_xi, 2048)*hp.alm2map(dlm_zeta_mask,2048)

nT_143_mask, binsT_143_mask, patchesT_143_mask = plt.hist(Tmap_143_mask[np.where(mask_map!=0)], bins=binning)

nTfilt_143_mask, binsTfilt_143_mask, patchesTfilt_143_mask = plt.hist(filtered_outmap_143_mask[np.where(mask_map!=0)], bins=binning)

plt.figure()

l1, = plt.plot(nT_143_mask/simps(nT_143_mask),label='Temperature',drawstyle='steps')
plt.plot(nTfilt_143_mask/simps(nTfilt_143_mask), label='Temperature (filtered)', drawstyle='steps', ls='--',c=l1.get_c())
l1, = plt.plot(ng_mask/simps(ng_mask),label='LSS',drawstyle='steps')
plt.plot(ngfilt_mask/simps(ngfilt_mask), label='LSS (filtered)', drawstyle='steps', ls='--',c=l1.get_c())

plt.title('Pixel Distribution of (Masked) Input Maps')
plt.xlabel('Bin Number')
plt.ylabel('Normalized # of Pixels')

plt.legend()
plt.savefig(outdir+'143_hists_mask')



nout_143_mask, binsout_143_mask, patchesout_143_mask = plt.hist(output_map_143_mask[np.where(mask_map!=0)], bins=binning)
noutfilt_143_mask, binsoutfilt_143_mask, patchesoutfilt_143_mask = plt.hist(filtered_outmap_143_mask[np.where(mask_map!=0)], bins=binning)

norm_prod_out_143_mask = kn(0, np.abs(np.linspace((binsout_143_mask[1]+binsout_143_mask[0])/2,(binsout_143_mask[-1]+binsout_143_mask[-2])/2,binning))/(np.std(Tmap_143_mask[np.where(mask_map!=0)])*np.std(lssmap[np.where(mask_map!=0)]))) \
    / (np.pi * np.std(noise_Tmap_mask[np.where(mask_map!=0)]) * np.std(lssmap[np.where(mask_map!=0)]))
norm_prod_out_143_mask_filt = kn(0, np.abs(np.linspace((binsoutfilt_143_mask[1]+binsoutfilt_143_mask[0])/2,(binsoutfilt_143_mask[-1]+binsoutfilt_143_mask[-2])/2,binning))/(np.std(Tmap_143_mask[np.where(mask_map!=0)])*np.std(lssmap[np.where(mask_map!=0)]))) \
    / (np.pi * np.std(Tmap_143_mask[np.where(mask_map!=0)]) * np.std(lssmap[np.where(mask_map!=0)]))

mismatch1 = norm_prod_out_143_mask.max() / nout_143_mask.max()
mismatch2 = norm_prod_out_143_mask_filt.max() / nout_143_mask.max()
plt.figure()
plt.plot((binsout_143_mask[1:]+binsout_143_mask[:-1])/2, nout_143_mask, label='Map Pixels')
l1,=plt.plot((binsout_143_mask[1:]+binsout_143_mask[:-1])/2, norm_prod_out_143_mask/mismatch1, label='Normal Product Distribution',ls='--')
#plt.plot((noise_binsoutfilt_mask[1:]+noise_binsoutfilt_mask[:-1])/2, norm_prod_out_143_mask_filt/mismatch2, label='Normal Product Distribution (Filtered)', c=l1.get_c(), ls='--')

plt.title('Pixel Distribution of Output Map from Masked Inputs')
plt.xlabel('v/c')
plt.ylabel('# of Pixels')
plt.xlim([((binsout_143_mask[1:]+binsout_143_mask[:-1])/2)[np.where((norm_prod_out_143_mask/mismatch1)>0.01*(norm_prod_out_143_mask/mismatch1).max())][0],((binsout_143_mask[1:]+binsout_143_mask[:-1])/2)[np.where((norm_prod_out_143_mask/mismatch1)>0.01*(norm_prod_out_143_mask/mismatch1).max())][-1]])
plt.legend()
plt.savefig(outdir+'normprod_143_mask')





'''
Investigate the non-gaussianity of estimator outputs
1-pt and 2-pt statistics


import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})

import numpy as np
import healpy as hp
import os
import config as conf
import common as c
from scipy.stats import kurtosis, skew
#import estim
#from astropy.io import fits


frequency = 143
basic_conf_dir = c.get_basic_conf(conf)
estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(conf, exclude = False))+'/T_freq='+str(frequency)+'/'  

class estim_holder(object):  # Currently (but not necessarily exclusively) constant values
    def __init__(self):
        self.nbin = 8
        self.N_fine_modes = 32

EST = estim_holder()
outdir = 'plots/analysis/plots_%dx%d/lss-gauss_TT-143sky/'% (EST.nbin, EST.N_fine_modes)
if not os.path.exists(outdir):
    os.makedirs(outdir)


def load_noise(nside, nsideout, extra_tag=''):
    Noise = c.load(basic_conf_dir,'Nvrvr_'+str(nside)+'_'+str(nsideout)+extra_tag+'_lcut'+str(3*nside), dir_base = estim_dir+'sims')
    return Noise

def load_PCA_weights(nside, nsideout):
    PCA = c.load(basic_conf_dir,'PCA_N'+str(nside)+'Nout'+str(nsideout), dir_base = estim_dir+'sims')
    R_P = PCA['rot']
    weights = np.abs(np.median(R_P[:,-1,:],axis=0)/np.linalg.norm((np.median(R_P[:,-1,:],axis=0))))
    return weights
        
def load_sim(nside, nsideout, mask, extra_tag=''):
    qe_full = c.load(basic_conf_dir,'qe_vr_'+str(nside)+'_'+str(nsideout)+'_full'+'_real=0'+extra_tag+'_mask='+str(mask)+'_nlevel=0_lcut'+str(3*nside), dir_base = estim_dir+'sims')
    qe_gauss = c.load(basic_conf_dir,'qe_vr_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real=0'+extra_tag+'_mask='+str(mask)+'_nlevel=0_lcut'+str(3*nside), dir_base = estim_dir+'sims')
    weights = load_PCA_weights(nside, nsideout)
    vrecon_full  = np.sum( qe_full*weights[:,np.newaxis], axis=0)
    vrecon_gauss = np.sum(qe_gauss*weights[:,np.newaxis], axis=0)
    if mask[:4] == 'True':
        mask_map = hp.ud_grade(np.load(mask.split('[@')[1].split(']')[0].replace('>>','/')), nsideout)
        if 1 not in mask_map:
            mask_map = np.load(mask.split('[@')[1].split(']')[0].replace('>>','/'))
            mask_map = hp.ud_grade(hp.ud_grade(mask_map,int(hp.npix2nside(mask_map.size)/((hp.npix2nside(mask_map.size)/nsideout)/4))),nsideout)
        vrecon_full *= mask_map
        vrecon_gauss *= mask_map
    else:
        mask_map = np.ones(vrecon_full.shape)
    return vrecon_full, vrecon_gauss, mask_map

mask_tag_false = 'False[@data>>mask_unWISE_thres_v10.npy]'
mask_tag_unwise = 'True[@data>>mask_unWISE_thres_v10.npy]'

TTsky_unmasked, _, _ = load_sim(2048, 2048, mask=mask_tag_false)
TTsky_masked, _, mask_unwise_N2048 = load_sim(2048, 2048, mask=mask_tag_unwise)
TTnoise_unmasked, _, _ = load_sim(2048, 2048, mask=mask_tag_false, extra_tag='_sims_noisemap')
TTnoise_masked, _, _, load_sim(2048, 2048, mask=mask_tag_unwise, extra_tag='_sims_noisemap')

TTsky_unmasked_pixels = TTsky_masked[np.where(mask_unwise_N2048!=0)]
TTnoise_unmasked_pixels = TTnoise_masked[np.where(mask_unwise_N2048!=0)]

plt.figure()
plt.hist(TTnoise_unmasked_pixels, bins=100, label='Unmasked Noise', density=True)
plt.hist(TTsky_unmasked, bins=100, label='Unmasked 143GHz', density=True)
plt.hist(TTsky_unmasked_pixels, bins=100, label='Masked 143GHz', alpha=0.75, density=True)
plt.legend(title='Masking')
plt.title('Masked/unmasked Reconstruction')
plt.xlabel('v/c ?')
#plt.ylabel(r'$N_\mathrm{pix}$')
plt.savefig(outdir + 'hist_masked_unmasked_comparison')

k_noisemap_unmasked = kurtosis(TTnoise_unmasked[np.where(mask_unwise_N2048!=0)])
k_noisemap_masked = kurtosis(TTnoise_unmasked_pixels)
k_unmasked = kurtosis(TTsky_unmasked[np.where(mask_unwise_N2048!=0)])
k_masked = kurtosis(TTsky_unmasked_pixels)

s_noisemap_unmasked = skew(TTnoise_unmasked[np.where(mask_unwise_N2048!=0)])
s_noisemap_masked = skew(TTnoise_unmasked_pixels)
s_unmasked = skew(TTsky_unmasked[np.where(mask_unwise_N2048!=0)])
s_masked = skew(TTsky_unmasked_pixels)

plt.figure()
plt.plot([0,1,2,3],[k_noisemap_unmasked, k_noisemap_masked, k_unmasked, k_masked], label='Kurtosis')
plt.plot([0,1,2,3], [s_noisemap_unmasked, s_noisemap_masked, s_unmasked, s_masked], label='Skewness')
plt.legend(title='Measure')
plt.xticks(ticks=[0,1,2,3],labels=['TT noise (no mask)','TT noise (mask)','143GHz (no mask)', '143GHz (mask)'])
plt.title('Kurtosis and Skewness as a Function of Mask')
plt.xlabel('Mask')
plt.savefig(outdir + 'kurtosis_and_skewness_N2048')




# mask_tag_gal70 = 'True[@data>>mask_planckgal70.npy]'
# mask_tag_pl60 = 'True[@data>>mask_unwisethres_planckgal60union.npy]'
# mask_tag_pl40 = 'True[@data>>mask_unwisethres_planckgal40union.npy]'


# Let's look at the 1-pt statistics for our different cases between resolutions
# N1024_unmasked, _, _ = load_sim(1024, 64, mask=mask_tag_false)
# N2048_unmasked, _, _ = load_sim(2048, 64, mask=mask_tag_false)
# N1024_masked, _, mask_unwise_N1024 = load_sim(1024, 64, mask=mask_tag_unwise)
# N2048_masked, _, mask_unwise_N2048 = load_sim(2048, 64, mask=mask_tag_unwise)
# N1024_gal70, _, mask_planck70_N1024 = load_sim(1024, 64, mask=mask_tag_gal70)
# N2048_gal70, _, mask_planck70_N2048 = load_sim(2048, 64, mask=mask_tag_gal70)
# N1024_pl60, _, mask_UWPl60_N1024 = load_sim(1024, 64, mask=mask_tag_pl60)
# N2048_pl60, _, mask_UWPl60_N2048 = load_sim(2048, 64, mask=mask_tag_pl60)
# N1024_pl40, _, mask_UWPl40_N1024 = load_sim(1024, 64, mask=mask_tag_pl40)
# N2048_pl40, _, mask_UWPl40_N2048 = load_sim(2048, 64, mask=mask_tag_pl40)


# N2048_unwise_unmasked_pixels = N2048_masked[np.where(mask_unwise_N2048!=0)]
# N1024_unwise_unmasked_pixels = N1024_masked[np.where(mask_unwise_N1024!=0)]
# N2048_gal70_unmasked_pixels = N2048_gal70[np.where(mask_planck70_N2048!=0)]
# N1024_gal70_unmasked_pixels = N1024_gal70[np.where(mask_planck70_N1024!=0)]
# N2048_pl60_unmasked_pixels = N2048_pl60[np.where(mask_UWPl60_N2048!=0)]
# N1024_pl60_unmasked_pixels = N1024_pl60[np.where(mask_UWPl60_N1024!=0)]
# N2048_pl40_unmasked_pixels = N2048_pl40[np.where(mask_UWPl40_N2048!=0)]
# N1024_pl40_unmasked_pixels = N1024_pl40[np.where(mask_UWPl40_N1024!=0)]


# plt.figure()
# plt.hist(N2048_unmasked, bins=100, label='N2048', density=True)
# plt.hist(N1024_unmasked, bins=100, label='N1024', alpha=0.75, density=True)
# plt.legend(title='Resolution')
# plt.title('Full-sky reconstruction of Gaussian inputs')
# plt.xlabel('v/c ?')
# #plt.ylabel(r'$N_\mathrm{pix}$')
# plt.savefig(outdir + 'hist_unmasked_resolution_comparison')

# plt.figure()
# plt.hist(N2048_gal70_unmasked_pixels, bins=100, label='N2048', density=True)
# plt.hist(N1024_gal70_unmasked_pixels, bins=100, label='N1024', alpha=0.75, density=True)
# plt.legend(title='Resolution')
# plt.title('Masked reconstruction of Gaussian inputs with Planck 70% galaxy cut')
# plt.xlabel('v/c ?')
# #plt.ylabel(r'$N_\mathrm{pix}$')
# plt.savefig(outdir + 'hist_mask-gal70_resolution_comparison')

# plt.figure()
# plt.hist(N2048_unwise_unmasked_pixels, bins=100, label='N2048', density=True)
# plt.hist(N1024_unwise_unmasked_pixels, bins=100, label='N1024', alpha=0.75, density=True)
# plt.legend(title='Resolution')
# plt.title('Masked reconstruction of Gaussian inputs with unWISE mask$')
# plt.xlabel('v/c ?')
# #plt.ylabel(r'$N_\mathrm{pix}$')
# plt.savefig(outdir + 'hist_mask-unwise_resolution_comparison')

# plt.figure()
# plt.hist(N2048_pl60_unmasked_pixels, bins=100, label='N2048', density=True)
# plt.hist(N1024_pl60_unmasked_pixels, bins=100, label='N1024', alpha=0.75, density=True)
# plt.legend(title='Resolution')
# plt.title('Masked reconstruction of Gaussian inputs with\nextended galactic plane cut to $f_\mathrm{sky}=0.6$')
# plt.xlabel('v/c ?')
# #plt.ylabel(r'$N_\mathrm{pix}$')
# plt.savefig(outdir + 'hist_mask-pl60_resolution_comparison')

# plt.figure()
# plt.hist(N2048_pl40_unmasked_pixels, bins=100, label='N2048', density=True)
# plt.hist(N1024_pl40_unmasked_pixels, bins=100, label='N1024', alpha=0.75, density=True)
# plt.legend(title='Resolution')
# plt.title('Masked reconstruction of Gaussian inputs with\nextended galactic plane cut to $f_\mathrm{sky}=0.4$')
# plt.xlabel('v/c ?')
# #plt.ylabel(r'$N_\mathrm{pix}$')
# plt.savefig(outdir + 'hist_mask-pl40_resolution_comparison')


# # We don't seem to make any improvements by doing this at N2048 over N1024

# plt.figure()
# plt.hist(N2048_unmasked, bins=100, label='Full-sky', density=True, histtype='step')
# plt.hist(N2048_gal70_unmasked_pixels, bins=100, label='Planck 0.7', density=True, histtype='step')
# plt.hist(N2048_unwise_unmasked_pixels, bins=100, label='unWISE binary mask', density=True, histtype='step')
# plt.hist(N2048_2048_unwise_unmasked_pixels, bins=100, label='unWISE N$_\mathrm{out}$2048', density=True, histtype='step')
# plt.hist(N2048_pl60_unmasked_pixels, bins=100, label='unWISE and Planck 0.6', density=True, histtype='step')
# plt.hist(N2048_pl40_unmasked_pixels, bins=100, label='unWISE and Planck 0.4', density=True, histtype='step')
# plt.legend(title='Masking Case')
# plt.title('Reconstruction of Gaussian inputs with various masks')
# plt.xlabel('v/c ?')
# plt.xlim((-0.07412132410350684, 0.07206240023194098))
# #plt.ylabel(r'$N_\mathrm{pix}$')
# plt.savefig(outdir + 'hist_mask+plane-extension_comparison')

# # Let's plot the kurtosis and skew as a function of mask shape/size for
# # the N1024 case. From the results above we expect the skew to remain
# # around zero, while the kurtosis should increase as the mask shape
# # begins to dominate the sky.

# k_unmasked = kurtosis(N1024_unmasked)
# k_gal70 = kurtosis(N1024_gal70_unmasked_pixels)
# k_masked = kurtosis(N1024_unwise_unmasked_pixels)
# k_pl60 = kurtosis(N1024_pl60_unmasked_pixels)
# k_pl40 = kurtosis(N1024_pl40_unmasked_pixels)

# s_unmasked = skew(N1024_unmasked)
# s_gal70 = skew(N1024_gal70_unmasked_pixels)
# s_masked = skew(N1024_unwise_unmasked_pixels)
# s_pl60 = skew(N1024_pl60_unmasked_pixels)
# s_pl40 = skew(N1024_pl40_unmasked_pixels)

# plt.figure()
# plt.plot([0,1,2,3,4],[k_unmasked, k_gal70, k_masked, k_pl60, k_pl40], label='Kurtosis')
# plt.plot([0,1,2,3,4], [s_unmasked, s_gal70, s_masked, s_pl60, s_pl40], label='Skewness')
# plt.legend(title='Measure')
# plt.xticks(ticks=[0,1,2,3,4],labels=['unmasked','Planck (Pl) 70', 'unWISE (UW)', 'UW+Pl60', 'UW+Pl40'])
# plt.title('Kurtosis and Skewness as a Function of Mask')
# plt.xlabel('Mask')
# plt.savefig(outdir + 'kurtosis_and_skewness_N1024')

# # Let's just check the story is unchanged for N2048

# k_unmasked = kurtosis(N2048_unmasked)
# k_gal70 = kurtosis(N2048_gal70_unmasked_pixels)
# k_masked = kurtosis(N2048_unwise_unmasked_pixels)
# k_pl60 = kurtosis(N2048_pl60_unmasked_pixels)
# k_pl40 = kurtosis(N2048_pl40_unmasked_pixels)

# s_unmasked = skew(N2048_unmasked)
# s_gal70 = skew(N2048_gal70_unmasked_pixels)
# s_masked = skew(N2048_unwise_unmasked_pixels)
# s_pl60 = skew(N2048_pl60_unmasked_pixels)
# s_pl40 = skew(N2048_pl40_unmasked_pixels)

# plt.figure()
# plt.plot([0,1,2,3,4],[k_unmasked, k_gal70, k_masked, k_pl60, k_pl40], label='Kurtosis')
# plt.plot([0,1,2,3,4], [s_unmasked, s_gal70, s_masked, s_pl60, s_pl40], label='Skewness')
# plt.legend(title='Measure')
# plt.xticks(ticks=[0,1,2,3,4],labels=['unmasked','Planck (Pl) 70', 'unWISE (UW)', 'UW+Pl60', 'UW+Pl40'])
# plt.title('Kurtosis and Skewness as a Function of Mask')
# plt.xlabel('Mask')
# plt.savefig(outdir + 'kurtosis_and_skewness_N2048')



# plt.figure();plt.loglog(hp.anafast(N2048_2048_masked)[10:]);plt.savefig(outdir+'2pt20482048')
'''




'''
Investigating non-gaussianity of sims w/ real data (using snipped files of weighted recon from different cases)



import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

nbins = 1000

def gaussian(x, a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d

plt.switch_backend('agg')

TTsynfastmap = np.load('synfast_recon.npy')
TTdatamap = np.load('data_recon.npy')

TTsynfasthist, TTsynfastbins, _ = plt.hist(TTsynfastmap[np.where(TTsynfastmap!=0.)], bins=nbins, log=True)
TTdatahist, TTdatabins, _ = plt.hist(TTdatamap[np.where(TTdatamap!=0.)], bins=nbins, log=True)

TTsynfastxpoints = np.array([(TTsynfastbins[i]+TTsynfastbins[i+1])/2 for i in np.arange(TTsynfastbins.size-1)])
TTdataxpoints = np.array([(TTdatabins[i]+TTdatabins[i+1])/2 for i in np.arange(TTdatabins.size-1)])

TTsynfastpopt, TTsynfastpcov = curve_fit(gaussian, TTsynfastxpoints, TTsynfasthist)
TTdatapopt, TTdatapcov = curve_fit(gaussian, TTdataxpoints, TTdatahist)

plt.figure()
plt.semilogy(TTsynfastxpoints, TTsynfasthist); plt.fill_between(TTsynfastxpoints,TTsynfasthist,label='TT synfast')
plt.semilogy(TTdataxpoints, TTdatahist); plt.fill_between(TTdataxpoints, TTdatahist,label='TT data')
plt.semilogy(TTsynfastxpoints, gaussian(TTsynfastxpoints, *TTsynfastpopt), c='k')
plt.semilogy(TTdataxpoints, gaussian(TTdataxpoints, *TTdatapopt), c='k')
plt.legend()
plt.savefig('histograms')

'''


























'''
# Shortcuts for noise fixing, code block after this one.
import numpy as np
import config as conf
import estim
import common as c
import matplotlib.pyplot as plt ; plt.switch_backend('agg')
import healpy

hp = healpy
lmax = 3072
lfilt = 3
Ls = [1,2,3,4,5,6,10]
fine = False
cal = False
nside = 1024
nsideout = 64
r = 0
n_level = 0
use_cleaned = False
mask = False
frequency = None

EST = estim.estimator(conf_module = conf, data_lmax = 3072)
EST.set_Nfine(32)
self = EST
highpass = lambda MAP, ellcut : hp.alm2map(hp.almxfl(hp.map2alm(MAP), [0 if ell > ellcut else 1 for ell in np.arange(nsideout*3)]),nsideout)
import os
outdir = 'plots/plots_%dx%d/manual_ClTT_matching/'% (EST.nbin, EST.N_fine_modes)
if not os.path.exists(outdir):
    os.mkdir(outdir)

EST.set_theory_Cls(add_ksz=True, add_ml=False, add_lensing=False, use_cleaned=False, frequency=None, get_haar=False)

lcut=3072
Lsamp = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
Noise_int = np.zeros((len(Lsamp),self.nbin)) 

for lid, l  in enumerate(Lsamp):
    for i in np.arange(self.nbin):
        Noise_int[lid,i] = self.Noise_vr_diag(lmax, i, i, ell)

'''








'''
### Shortcuts for spectra test/fix on bad reconstruction
#### TESTING RECCO THEORY NOISE (WRONG FOR UNWISE) VS MANUAL NOISE (RIGHT)
import numpy as np
import config as conf
import common as c
import matplotlib.pyplot as plt ; plt.switch_backend('agg')
import healpy
from scipy.interpolate import interp1d
import loginterp
from math import lgamma
import os

outdir = 'plots/plots_8x32/manual_ClTT_matching/manual/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
hp = healpy
lmax = 3072
nside = 1024
nsideout = 64
simsdir = 'estim/0914867466fac2b64333db77493b96c9/T_freq=None/'
bin_width = 525.7008226572364

########## manual estimator
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


def manual_noise(cltt, clgg, cltaudg):
    cltt[:20] = clgg[:20,0,0] = 1e15
    cltaudg_binned = cltaudg[:,0,0]*bin_width
    clgg_binned = clgg[:,0,0]
    Noise_ells = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
    Noise = np.zeros((Noise_ells.size))
    wigs = 0
    for lid, ell in enumerate(Noise_ells):
        terms = 0
        print('Computing noise at ell = %d' % ell)
        for l2 in np.arange(3072):
            for l1 in np.arange(np.abs(l2-ell),l2+ell+1):
                if l1 > lmax-1 or l1 <2:   #triangle rule
                    if wigner_symbol(ell, l1, l2) != 0.:
                        wigs += wigner_symbol(ell, l1, l2)
                    continue
                gamma_ksz = np.sqrt((2*l1+1)*(2*l2+1)*(2*ell+1)/(4*np.pi))*wigner_symbol(ell, l1, l2)*cltaudg_binned[l2]
                term_entry = (gamma_ksz*gamma_ksz/(cltt[l1]*clgg_binned[l2]))
                if not np.isnan(term_entry):
                    terms += term_entry
        Noise[lid] = (2*ell+1)/terms
        print(wigs)
    print('             ',wigs)
    return loginterp.log_interpolate(Noise, Noise_ells)
                
def get_noise(cltt, clgg, cltaudg):
    Noise_ells = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
    Noise = np.zeros((Noise_ells.size, 8))
    C = cltaudg[:,0,0]
    L = np.unique(np.append(np.geomspace(2,lmax-2,300).astype(int),lmax-1))
    Lnz = L
    L_int = L[np.in1d(L, Lnz)]   
    for lid, ell in enumerate(Noise_ells):
        print(ell)
        a = []
        for l1_id, l2 in enumerate(L_int):    
            terms = 0
            for l1 in np.arange(np.abs(l2-ell),l2+ell+1):
                if l1 > lmax-1 or l1 <2:   #triangle rule
                    continue
                factor = np.sqrt((2*ell+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*bin_width*C[l2]*wigner_symbol(ell, l1, l2)
                G = (cltt[l2]*clgg[l1]*factor)/(cltt[l1]*cltt[l2]*clgg[l1]*clgg[l2])
                terms += factor*G
                if np.isnan(terms):
                    break
            if np.isnan(terms):
                break
            a.append(terms)
        #Ignore last couple ell cause they can be problematic, regardless of lmax
        I = interp1d(L_int[:-2] ,np.asarray(a)[:-2], kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
        c =   np.sum(I)
        N_alphaalpha = (2*ell+1)/c
        
        a = []
        for l1_id, l1 in enumerate(L):   
            terms = 0    
            for l2 in np.arange(np.abs(l1-ell),l1+ell+1):
                if l2 > lmax-1 or l2 <2:   #triangle rule
                    continue
                factor = np.sqrt((2*ell+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*bin_width*C[l2]*wigner_symbol(ell, l1, l2)
                g_alpha = g_gamma_1 = (cltt[l2]*clgg[l1]*factor)/(cltt[l1]*cltt[l2]*clgg[l1]*clgg[l2])        
                #g_gamma_2 = (cltt[l1]*clgg[l2]*factor)/(cltt[l2]*cltt[l1]*clgg[l2]*clgg[l1])        
                terms  += g_alpha*(g_gamma_1*cltt[l1]*clgg[l2] )
            a.append(terms)        
        I = interp1d(L[:-2] ,np.asarray(a)[:-2], kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
        
        Noise[lid] = np.sum(I) * N_alphaalpha * N_alphaalpha / (2*ell+1)
    
    return np.swapaxes(loginterp.log_interpolate_vector(Noise, Noise_ells),0,1)
    
    


def reconst_bin0(temp_map, gal_map, cltt, clgg, cltaudg, noise_factor):
    Tlm = hp.map2alm(temp_map)
    glm = hp.map2alm(gal_map)
    cltaudg_bin = cltaudg * bin_width
    cltt[:2] = 0
    Tfilter = [1/cltt[i] if cltt[i] !=0 else 0 for i in np.arange(cltt.size)]
    gfilter = [cltaudg_bin[i,0,0]/clgg[i] if min((abs(cltaudg_bin[i,0,0]),clgg[i])) != 0 else 0 for i in np.arange(clgg.shape[0])]
    Tlm_filtered = hp.almxfl(Tlm, Tfilter)
    glm_filtered = hp.almxfl(glm, gfilter)
    T_filtered = hp.alm2map(Tlm_filtered, 1024)
    g_filtered = hp.alm2map(glm_filtered, 1024)
    vmap = T_filtered*g_filtered
    veff_reconst = hp.alm2map(hp.almxfl(hp.map2alm(vmap, lmax=(3*64-1)), noise_factor), 64)
    return veff_reconst
    

#### Compare the thing in eq 9 to what we get out of noise. Even can compare f and g. Code eq 9 from scratch and test piecemeal

lssmap = c.load(c.get_basic_conf(conf), 'lss_vr_1024_real=0_bin=0', dir_base=simsdir+'sims')[0]  # unWISE has only 1 lss bin
kszmap = c.load(c.get_basic_conf(conf), 'ksz_1024_real=0', dir_base=simsdir+'sims')
pCMBmap = c.load(c.get_basic_conf(conf),'T0_1024_real=0_bin=0', dir_base = simsdir+'sims')
taudmaps = np.zeros((8,12*1024**2))
for b in np.arange(8):
    taudmaps[b,:] = c.load(c.get_basic_conf(conf),'taudmaps_1024_64_real=0_bin=%d'%b,simsdir+'sims')

#if not os.path.exists(outdir+'override_noise.npy'):
    #print('getting noise')
    #noise         = get_noise(hp.anafast(pCMBmap+kszmap), hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'))
    #noise_cmbonly = get_noise(hp.anafast(pCMBmap),        hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'))
    #choice = input('Save noise? [Y/N]: ')
    #if choice.upper() == 'Y':
        #np.savez(outdir+'override_noise', noise=noise, noise_cmbonly=noise_cmbonly)
#else:
    #print('|| |    W A R N I N G    | ||   noise loaded in and not computed')
    #noise         = np.load(outdir+'override_noise.npy')['noise']
    #noise_cmbonly = np.load(outdir+'override_noise.npy')['noise_cmbonly']

#qe_full     = reconst_bin0(pCMBmap+kszmap,                              lssmap, hp.anafast(pCMBmap+kszmap), hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'), noise[0]        )
#qe_gauss    = reconst_bin0(pCMBmap+hp.synfast(hp.anafast(kszmap),1024), lssmap, hp.anafast(pCMBmap+kszmap), hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'), noise[0]        )
##qe_constKSZ = reconst_bin0(pCMBmap,                                     lssmap, hp.anafast(pCMBmap),        hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'), noise_cmbonly[0])

#plt.figure()
#plt.loglog(np.arange(192), hp.anafast(qe_full), label='pCMB + correlated kSZ')
#plt.loglog(np.arange(192), hp.anafast(qe_gauss), label='pCMB + uncorrelated kSZ')
##plt.loglog(np.arange(192), hp.anafast(qe_constKSZ), label='pCMB only')
#plt.loglog(np.arange(192), noise[0], label='Theory noise')
#plt.legend()
#plt.savefig(outdir + 'noise_kSZ=True_pCMB=%s' % pCMB_on)


#noise_manual = manual_noise(hp.anafast(pCMBmap+kszmap), hp.anafast(lssmap)[:,np.newaxis,np.newaxis], np.load(outdir+'Cltaudg_healpix.npy'))
#noise_cmbonly = manual_noise(hp.anafast(pCMBmap), hp.anafast(lssmap)[:,np.newaxis,np.newaxis], np.load(outdir+'Cltaudg_healpix.npy'))

#qe_full_manual     = reconst_bin0(pCMBmap+kszmap,                              lssmap, hp.anafast(pCMBmap+kszmap), hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'), noise_manual        )
#qe_gauss_manual     = reconst_bin0(pCMBmap+hp.synfast(hp.anafast(kszmap),1024),                              lssmap, hp.anafast(pCMBmap+kszmap), hp.anafast(lssmap), np.load(outdir+'Cltaudg_healpix.npy'), noise_manual        )




import estim
EST = estim.estimator(conf_module = conf, data_lmax = 3072)
self = EST
EST.set_Nfine(32)  # Needed for right kSZ power
EST.estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(self.conf, exclude = False))+'/T_freq='+str(None)+'/'
theory_gg = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,self.lss),self.load_L())
theory_taudg = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud',self.lss),self.load_L())
theory_pCMB = loginterp.log_interpolate_matrix(self.load_theory_Cl('pCMB','pCMB'), c.load(self.basic_conf_dir,'L_pCMB_lmax='+str(self.data_lmax), dir_base = 'Cls'))
theory_kSZ = loginterp.log_interpolate_matrix(self.load_theory_Cl('kSZ','Nfine_'+str(self.N_fine_modes)),self.load_L())


pCMB_on = True
if not pCMB_on:
    Tmap = np.zeros(pCMBmap.size)
    theory_TT = theory_kSZ.copy()[:,0,0]
else:
    Tmap = pCMBmap.copy()
    theory_TT = (theory_kSZ + theory_pCMB)[:,0,0]

use_theory_spectra = True
if use_theory_spectra:
    cltt = theory_TT[:3072]
    cltaudg = theory_taudg[:3072,:,:]
    clgg = theory_gg[:3072,:,:]    
else:
    cltt = hp.anafast(Tmap+kszmap)[:,np.newaxis,np.newaxis]
    clgg = hp.anafast(lssmap)[:,np.newaxis,np.newaxis]
    cltaudg = np.load(outdir+'Cltaudg_healpix.npy')    

noise_manual = manual_noise(cltt, clgg, cltaudg)
qe_full_manual  = reconst_bin0(Tmap+kszmap,                              lssmap, cltt, clgg[:,0,0], cltaudg, noise_manual        )
qe_gauss_manual = reconst_bin0(Tmap+hp.synfast(hp.anafast(kszmap),1024), lssmap, cltt, clgg[:,0,0], cltaudg, noise_manual        )

plt.figure()
plt.loglog(np.arange(192), hp.anafast(qe_full_manual), label='pCMB + correlated kSZ')
plt.loglog(np.arange(192), hp.anafast(qe_gauss_manual), label='pCMB + uncorrelated kSZ')
plt.loglog(np.arange(192), noise_manual, label='Theory noise')
plt.legend()
plt.savefig(outdir + 'noise_kSZ=True_pCMB=%s.png' % pCMB_on)


'''
























'''
Broken-down estimator but gives weird CTT results with pCMB???
### Shortcuts for spectra test/fix on bad reconstruction
import numpy as np
import config as conf
import estim
import common as c
import matplotlib.pyplot as plt ; plt.switch_backend('agg')
import healpy
from scipy.interpolate import interp1d

hp = healpy
lmax = 3072
lfilt = 3
nside = 1024
nsideout = 64
r = 0
n_level = 0
i = binn = lssid = 0

EST = estim.estimator(conf_module = conf, data_lmax = 3072)
EST.set_Nfine(32)
self = EST
highpass = lambda MAP, ellcut : hp.alm2map(hp.almxfl(hp.map2alm(MAP), [0 if ell > ellcut else 1 for ell in np.arange(nsideout*3)]),nsideout)
import os
outdir = 'plots/plots_%dx%d/manual_ClTT_matching/manual/'% (EST.nbin, EST.N_fine_modes)
if not os.path.exists(outdir):
    os.mkdir(outdir)

#self.set_theory_Cls(add_ksz = True, add_ml = False, use_cleaned = False, frequency = None)
EST.estim_dir = 'estim/'+c.get_hash(c.get_basic_conf(self.conf, exclude = False))+'/T_freq='+str(None)+'/'
#EST.Ttag = 'T0'
#EST.beam = self.conf.beamArcmin_T*np.pi/180./60.
#EST.dT   = self.conf.noiseTuKArcmin_T*np.pi/180./60./self.conf.T_CMB
#lcut = 3*nside-self.lcut_num
#ls = np.arange(3*nside)
#beam_window = np.exp(-ls*(ls+1)*(self.beam**2)/(16.*np.log(2)))
#ones =  np.ones(3*nside)
#cut = np.where(np.arange(3*nside)<lcut, 1, 1e-30)
#bin_width = EST.deltachi
bin_width = 525.7008226572364

lssmap = c.load(c.get_basic_conf(conf), 'lss_vr_1024_real=0_bin=0', dir_base=EST.estim_dir+'sims')[0]  # unWISE has only 1 lss bin
kszmap = c.load(c.get_basic_conf(conf), 'ksz_1024_real=0', dir_base=EST.estim_dir+'sims')
pCMBmap = c.load(self.basic_conf_dir,'T0_'+str(nside)+'_real='+str(r)+'_bin='+str(0), dir_base = self.estim_dir+'sims')
taudmaps = np.zeros((8,12*1024**2))
for b in np.arange(8):
    taudmaps[b,:] = c.load(c.get_basic_conf(conf),'taudmaps_1024_64_real=0_bin=%d'%b,EST.estim_dir+'sims')

import loginterp
inc_cmb = True
theory = {'lss-lss' : False,
          'taud-lss' : False,
          'T-lss' : True,
          'T-T' : False}

casetag = 'theory_lss-lss=%s_taud-lss=%s_T-lss=%s_T-T=%s_pCMB=%s' % (theory['lss-lss'], theory['taud-lss'], theory['T-lss'], theory['T-T'], inc_cmb)

theory_gg = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,self.lss),self.load_L())
theory_taudg = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud',self.lss),self.load_L())
theory_pCMB = loginterp.log_interpolate_matrix(self.load_theory_Cl('pCMB','pCMB'), c.load(self.basic_conf_dir,'L_pCMB_lmax='+str(self.data_lmax), dir_base = 'Cls'))
theory_kSZ = loginterp.log_interpolate_matrix(self.load_theory_Cl('kSZ','Nfine_'+str(self.N_fine_modes)),self.load_L())
theory_TT = theory_kSZ.copy()

#if not inc_cmb:
    #Tmap = np.zeros(shape=kszmap.shape)
#else:
    #Tmap = pCMBmap.copy()
    #theory_TT += theory_pCMB
#if theory['lss-lss']:
    #EST.Cls['lss-lss'] = theory_gg
#else:
    #EST.Cls['lss-lss'] = hp.anafast(lssmap)[:,np.newaxis,np.newaxis]
#if theory['taud-lss']:
    #EST.Cls['taud-lss'] = theory_taudg
#else:
    #EST.Cls['taud-lss'] = np.zeros((3072,8,1))
    #if os.path.exists(outdir+'Cltaudg_healpix.npy'):
        #EST.Cls['taud-lss'] = np.load(outdir+'Cltaudg_healpix.npy')
    #else:
        #for b in np.arange(8):
            #EST.Cls['taud-lss'][:,b,0] = hp.anafast(taudmaps[b,:], lssmap)
        #np.save(outdir+'Cltaudg_healpix',EST.Cls['taud-lss'])           
#if theory['T-lss']:
    #EST.Cls['T-lss'] = np.zeros((3072,1,1))
#else:
    #if not inc_cmb:
        #EST.Cls['T-lss'] = hp.anafast(kszmap,lssmap)[:,np.newaxis,np.newaxis]
    #else:
        #EST.Cls['T-lss'] = hp.anafast(Tmap+kszmap,lssmap)[:,np.newaxis,np.newaxis]   
#if theory['T-T']:
    #EST.Cls['T-T'] = theory_TT
#else:
    #if not inc_cmb:
        #EST.Cls['T-T'] = hp.anafast(kszmap)[:,np.newaxis,np.newaxis]
    #else:
        #EST.Cls['T-T'] = hp.anafast(Tmap+kszmap)[:,np.newaxis,np.newaxis]


# Compute fresh noise
if os.path.exists(outdir+'Noise_%s.npy' % casetag):
    Noise = np.load(outdir+'Noise_%s.npy' % casetag)
else:
    EST.cs = {}  # Wipe memory of previously computed coupling sums
    Lsamp = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
    Noise_int = np.zeros((len(Lsamp),self.nbin)) 
    for lid, l  in enumerate(Lsamp):
        for i in np.arange(self.nbin):
            Noise_int[lid,i] = self.Noise_iso(lmax-1, 'vr', i, i, l)
    Noise = np.swapaxes(loginterp.log_interpolate_vector(Noise_int, Lsamp),0,1)
    np.save(outdir+'Noise_%s.npy' % casetag, Noise)

cllsslssbinned = EST.Cls['lss-lss'][:3072,:,:]
cltaudlssbinned = EST.Cls['taud-lss'][:3072,:,:]
clTlssbinned = EST.Cls['T-lss'][:3072,:,:]
Cltaudd = cltaudlssbinned[:,i,lssid]*bin_width  # UNWISE
Cldd = cllsslssbinned[:,lssid,lssid]
ClTd = clTlssbinned[:,0,lssid]
ClTT = EST.Cls['T-T'][:3072,0,0]

Tfieldsignal = Tmap + kszmap
Tnoisefield = Tmap + hp.synfast(hp.anafast(kszmap),1024)


TconstantkSZfield = Tmap + np.mean(kszmap)

lssfield = lssmap[np.newaxis,:]



plt.figure()
plt.loglog(np.arange(1,3072), hp.anafast(Tmap+kszmap)[1:])
plt.loglog(np.arange(1,3072), EST.Cls['T-T'][1:3072,0,0])
plt.savefig(outdir+'TTspec')
plt.figure()
plt.loglog(np.arange(1,3072), hp.anafast(lssmap)[1:])
plt.loglog(np.arange(1,3072), EST.Cls['lss-lss'][1:3072,0,0])
plt.savefig(outdir+'ggspec')
if os.path.exists(outdir+'Cltaudg_healpix.npy'):
    plt.figure()
    for b in [0,self.nbin//2,self.nbin-1]:
        l, = plt.semilogx(np.arange(1,3072), np.load(outdir+'Cltaudg_healpix.npy')[1:3072,b,0], label='bin %d'%(b+1))
        plt.semilogx(np.arange(1,3072), EST.Cls['taud-lss'][1:3072,b,0], c=l.get_c())
    plt.legend()
    plt.savefig(outdir+'taudgspec')



def reconst_bin(Tfield, binn):
    dTlm = healpy.map2alm(Tfield)
    #dTlm_beamed = healpy.almxfl(dTlm,(1./beam_window)*cut)
    dlm_in = healpy.almxfl(healpy.map2alm(lssfield[lssid]),cut)  # UNWISE
    dTlm_xi = healpy.almxfl(dTlm,np.divide(ones, ClTT, out=np.zeros_like(ones), where=ClTT!=0))
    dlm_zeta = healpy.almxfl(dlm_in, np.divide(Cltaudd, Cldd, out=np.zeros_like(Cltaudd), where=Cldd!=0))
    dTlm_xi_f = dTlm_xi
    dlm_zeta_f = dlm_zeta
    xizeta = healpy.alm2map(dTlm_xi_f, nside,verbose=False)*healpy.alm2map(dlm_zeta_f,nside,verbose=False)
    dTlm_xibar = healpy.almxfl(dTlm, np.divide(Cltaudd, ClTd, out=np.zeros_like(Cltaudd), where=ClTd!=0) )  
    dlm_zetabar = healpy.almxfl(dlm_in, np.divide(ones, ClTd, out=np.zeros_like(ones), where=ClTd!=0) )
    ffactor1 = ClTd**2
    ffactor2 = ClTT * Cldd
    filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)
    dTlm_xibar_f = healpy.almxfl(dTlm_xibar, filterf)
    dlm_zetabar_f = healpy.almxfl(dlm_zetabar, filterf)
    xizetabar = healpy.alm2map(dTlm_xibar_f, nside,verbose=False)*healpy.alm2map(dlm_zetabar_f,nside,verbose=False)
    veff_reconstlm = healpy.almxfl(healpy.map2alm(xizeta-xizetabar,lmax=(3*nsideout-1)),Noise[binn])
    finalmap = healpy.alm2map(veff_reconstlm, nsideout)
    return finalmap






qe_full = reconst_bin(Tfieldsignal, 0)
qe_gauss = reconst_bin(Tnoisefield, 0)
#qe_constKSZ = reconst_bin(TconstantkSZfield, 0)

plt.figure()
plt.loglog(np.arange(192), hp.anafast(qe_full), label='signalmap')
plt.loglog(np.arange(192), hp.anafast(qe_gauss), label='noisemap')
#plt.loglog(np.arange(192), hp.anafast(qe_constKSZ), label='no kSZ')
plt.loglog(np.arange(192), Noise[0])
#plt.loglog(np.arange(192), Noise[0]*np.mean(hp.anafast(qe_gauss)[1:]/Noise[0,1:]),label='Noise x %.1f'%np.mean(hp.anafast(qe_gauss)[1:]/Noise[0,1:]))
plt.legend()
plt.savefig(outdir + 'noise_%s' % casetag)



'''



'''
#### TEST COARSE GRAINING MAP SPECTRA
self.conf.N_bins = 8
Cltaudcoarse = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud','taud'),self.load_L())
plt.figure()
for b in [0,conf.N_bins//2,conf.N_bins-1]:
    l, = plt.loglog(np.arange(1,3072), hp.anafast(taudmaps[b,:])[1:], label='bin %d'%(b+1)) 
    plt.loglog(np.arange(1,3073), Cltaudcoarse[1:,b,b],c=l.get_c())
plt.legend()
plt.title('tau spec')
plt.savefig(outdir+'taudspec')

plt.figure()
for b in [0,self.N_fine_modes//2,self.N_fine_modes-1]:
    l, = plt.loglog(np.arange(1,3072), hp.anafast(taudmaps_fine[b,:])[1:], label='bin %d'%(b+1)) 
    plt.loglog(np.arange(1,3073), EST.Cls['taud-taud'][1:,b,b], c=l.get_c())
plt.legend()
plt.title('tau spec fine')
plt.savefig(outdir+'taudspecfine')


Clvrvrcoarse = loginterp.log_interpolate_matrix(self.load_theory_Cl('vr','vr'), self.load_L())
plt.figure()
for b in [0,conf.N_bins//2,conf.N_bins-1]:
    l, = plt.loglog(np.arange(1,3072), hp.anafast(vmaps[b,:])[1:], label='bin %d'%(b+1)) 
    plt.loglog(np.arange(1,3073), Clvrvrcoarse[1:,b,b],c=l.get_c())
plt.legend()
plt.title('vr spec')
plt.savefig(outdir+'vrspec')

plt.figure()
for b in [0,self.N_fine_modes//2,self.N_fine_modes-1]:
    l, = plt.loglog(np.arange(1,3072), hp.anafast(vmaps_fine[b,:])[1:], label='bin %d'%(b+1)) 
    plt.loglog(np.arange(1,3073), EST.Cls['vr-vr'][1:,b,b], c=l.get_c())
plt.legend()
plt.title('vr spec fine')
plt.savefig(outdir+'vrspecfine')
'''








'''
#####
## TEST CLEANING ON PLANCK x unWISE 
#####


USE_CIB = True
USE_TSZ = True
USE_INOISE = True

import os
outdir = 'plots/debug_cleaning/cib=%s_tsz=%s_noise=%s/' % (USE_CIB,USE_TSZ,USE_INOISE)
if not os.path.exists(outdir):
    os.mkdir(outdir)

from spectra import *
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16})
plt.switch_backend('agg')
lmax = 3072
X = 'g'

ell_sparse = np.unique(np.append(np.geomspace(1,lmax,120).astype(int), lmax)) 
CTT_clean = np.zeros((len(ell_sparse),1,1))
CTX_clean = np.zeros((len(ell_sparse),1,conf.N_bins))
F = freqs()

NMAPS = len(F)
C = np.zeros((NMAPS, NMAPS, len(ell_sparse)))
for i in np.arange(NMAPS):
    for j in np.arange(NMAPS):
        C[i,j] = get_CTT(F[i], F[j], lmax, primary=True, inoise=USE_INOISE, cib=USE_CIB, tsz=USE_TSZ)

print('Computing ILC weights')
weights_l = np.zeros((NMAPS, len(ell_sparse)))
es = np.ones(NMAPS)
for lid, l  in enumerate(ell_sparse):
    try:
        Cl_inv = np.linalg.inv(C[:,:,lid])
    except np.linalg.LinAlgError as e:
        print('%s at l = %d' % (e, l))
        weights_l[:,lid] = np.ones(NMAPS) / np.linalg.norm(np.ones(NMAPS))
    else:
        weights_l[:,lid] = (np.dot(Cl_inv, es) / np.dot(np.dot(es,Cl_inv),es))

ell_facs = np.zeros(len(ell_sparse))
cross_facs = np.zeros((len(ell_sparse),conf.N_bins))
extragal_cross = np.zeros((NMAPS, len(ell_sparse), conf.N_bins))
extragal_specs = np.zeros((NMAPS, NMAPS, len(ell_sparse)))
noise_specs = np.zeros((NMAPS, NMAPS, len(ell_sparse)))  

print('Adding dirty components')
for i in range(NMAPS):
    extragal_cross[i] += get_CTX( X,F[i], lmax, isw = False, cib=USE_CIB, tsz=USE_TSZ)
    noise_specs[i,i] += get_CTT(F[i], F[i], lmax,primary=False, inoise=USE_INOISE,cib=False,tsz=False)
    for j in range(NMAPS):          
        extragal_specs[i,j] += get_CTT(F[i], F[j], lmax, primary=False,inoise=False,cib=USE_CIB,tsz=USE_TSZ)    

for lid, l in enumerate(ell_sparse):
    if l< 2:
        continue
    else:                        
        w = weights_l[:,lid]
        c_extragal = extragal_specs[:,:,lid]
        N_l = noise_specs[:,:,lid]
        ell_facs[lid] = np.dot(np.dot(w, c_extragal+N_l),w)  # This is the cleaning residual for the auto spectrum
        for n in range(conf.N_bins):
            cross_facs[lid,n] = np.dot(w, extragal_cross[:,lid,n])  # The cleaned cross spectra is just the weighted CgT as shot/inoise are uncorrelated






## Plot cleaning TT residuals
plt.figure()
if np.where(ell_facs<0)[0].size==0:
    plt.loglog(ell_sparse[2:],ell_facs[2:])
else:
    plt.semilogx(ell_sparse[2:],ell_facs[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{T}_c\mathrm{T}_c} - C_\ell^{\mathrm{TT}}$')
plt.title('TT residuals')
plt.tight_layout(pad=1.03)
plt.savefig(outdir + 'TT residuals')


## Plot cleaned TT against theory TT
plt.figure()
pCMB = get_CTT(F[0], F[0], lmax,primary=True,inoise=False,cib=False,tsz=False)
plt.loglog(ell_sparse[2:], pCMB[2:], label='Primary CMB')
plt.loglog(ell_sparse[2:], pCMB[2:] + ell_facs[2:], ls='--', label='Cleaned CMB')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{TT}}$')
plt.title('Cleaned TT')
plt.legend()
plt.tight_layout(pad=1.03)
plt.savefig(outdir + 'TT cleaned')


## Plot CIB and g auto against CIBg cross
plt.figure()
gg = c.load(basic_conf,'Cl_g_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','g',conf))[:,0,0]
for freq_i in F:
    CIBCIB = c.load(basic_conf,'Cl_CIB'+'('+str(min(freq_i,freq_i))+')'+'_CIB'+'('+str(max(freq_i,freq_i))+')'+'_lmax=3072', dir_base = 'Cls')[:,0,0]
    CIBg = c.load(basic_conf,'Cl_CIB'+'('+str(freq_i)+')'+'_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','CIB',conf))[:,:,0]
    l, = plt.loglog(ell_sparse, gg*CIBCIB, label='%d GHz'% freq_i)
    plt.loglog(ell_sparse, CIBg**2, c=l.get_c(),ls='--')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell\quad\left[\mathrm{CIBg}^2\right]$')
plt.title(r'CIBCIB*gg vs CIBg$^2$')
plt.legend(ncol=3,loc='lower left',title=r'Freq (dashed lines are CIBg$^2$)')
plt.tight_layout(pad=1.03)
plt.savefig(outdir + 'CIBg_cross')
## Plot correlation coefficient
plt.figure()
gg = c.load(basic_conf,'Cl_g_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','g',conf))[:,0,0]
for freq_i in F:
    CIBCIB = c.load(basic_conf,'Cl_CIB'+'('+str(min(freq_i,freq_i))+')'+'_CIB'+'('+str(max(freq_i,freq_i))+')'+'_lmax=3072', dir_base = 'Cls')[:,0,0]
    CIBg = c.load(basic_conf,'Cl_CIB'+'('+str(freq_i)+')'+'_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','CIB',conf))[:,0,0]
    plt.semilogx(ell_sparse, np.sqrt(CIBg**2/(CIBCIB*gg)),label='%d GHz'%freq_i)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sqrt{\frac{\left(C_\ell^{\mathrm{CIBg}}\right)^2}{C_\ell^{\mathrm{gg}}C_\ell^{\mathrm{CIBCIB}}}}$',rotation=0,labelpad=35)
plt.title(r'Correlation coefficient between $C_\ell^{\mathrm{gg}}$ and $C_\ell^{\mathrm{CIBCIB}}$')
plt.legend(bbox_to_anchor=(1.,0.75))
plt.tight_layout(pad=1.03)
plt.savefig(outdir + 'CIBg_corrcoeff')


## Plot tSZ and g auto against tSZg cross
plt.figure()
gg = c.load(basic_conf,'Cl_g_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','g',conf))[:,0,0]
for freq_i in F:
    tSZtSZ = c.load(basic_conf,'Cl_tSZ'+'('+str(min(freq_i,freq_i))+')'+'_tSZ'+'('+str(max(freq_i,freq_i))+')'+'_lmax=3072', dir_base = 'Cls')[:,0,0]
    tSZg = c.load(basic_conf,'Cl_tSZ'+'('+str(freq_i)+')'+'_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','tSZ',conf))[:,:,0]
    l, = plt.loglog(ell_sparse, gg*tSZtSZ, label='%d GHz'% freq_i)
    plt.loglog(ell_sparse, tSZg**2, c=l.get_c(),ls='--')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell\quad\left[\mathrm{tSZg}^2\right]$')
plt.title(r'tSZtSZ*gg vs tSZg$^2$')
plt.legend(ncol=3,loc='lower left',title=r'Freq (dashed lines are tSZg$^2$)')
plt.tight_layout(pad=1.03)
plt.savefig(outdir + 'tSZg_cross')
## Plot correlation coefficient
plt.figure()
freq_i = F[0]
gg = c.load(basic_conf,'Cl_g_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','g',conf))[:,0,0]
TSZTSZ = c.load(basic_conf,'Cl_tSZ'+'('+str(min(freq_i,freq_i))+')'+'_tSZ'+'('+str(max(freq_i,freq_i))+')'+'_lmax=3072', dir_base = 'Cls')[:,0,0]
TSZg = c.load(basic_conf,'Cl_tSZ'+'('+str(freq_i)+')'+'_g_lmax=3072', dir_base = 'Cls/'+c.direc('g','tSZ',conf))[:,0,0]
plt.semilogx(ell_sparse, np.sqrt(TSZg**2/(TSZTSZ*gg)))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\sqrt{\frac{\left(C_\ell^{\mathrm{tSZg}}\right)^2}{C_\ell^{\mathrm{gg}}C_\ell^{\mathrm{tSZtSZ}}}}$',rotation=0,labelpad=35)
plt.title(r'Correlation coefficient between $C_\ell^{\mathrm{gg}}$ and $C_\ell^{\mathrm{tSZtSZ}}$')
plt.tight_layout(pad=1.03,rect=[0,0,.9,1])
plt.savefig(outdir + 'tSZg_corrcoeff')


## We returned to using the halomodel to calculate all the spectra to stay consistent.
## This fixed our bad cleaning errors when CIB+noise is combined, but keep in mind
## this distances our unWISE Clgg from Alex's unWISE Clgg. We plot the difference here.
import loginterp
hmod_clgg = loginterp.log_interpolate(gg,ell_sparse)
with open('data/unWISE/Bandpowers_Auto_Sample1.dat', 'r') as FILE:
    x = FILE.readlines()
ells = np.array([float(ell) for ell in x[0].split(' ')])
clgg = np.array([float(g) for g in x[1].split(' ')])

truncated_ells = np.array([ell for ell in ells if ell <= lmax]).astype(int)
truncated_clgg = clgg[np.arange(truncated_ells.size)]

plt.figure()
plt.loglog(truncated_ells, truncated_clgg, label='unWISE (Alex)')
plt.loglog(truncated_ells, hmod_clgg[truncated_ells], label='ReCCO')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{\mathrm{gg}}$')
plt.title(r'$C_\ell^{\mathrm{gg}}$ for halomodel and from ArXiV:2105.03421v2')
plt.legend()
plt.savefig(outdir + 'Clgg_Comparison')


'''












'''
##### ESTIMATOR TESTING #####

import common as c
import config as conf
import estim
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import loginterp
plt.switch_backend('agg')

realno = 0
nside_maps = 1024
nside_reconst = 64
EST = estim.estimator(conf_module = conf, data_lmax = 3072)
EST.set_Nfine(64)


lmax=3072
nside=1024
self=EST
healpy=hp



EST.get_qe_vr_sims(nside_maps , nside_reconst , 0, use_cleaned = False, frequency = None, mask = False)

qe_full, qe_gauss =  EST.load_qe_sims_maps(realno,'vr', nside_maps, nside_reconst, 0, mask = False)

EST.get_clqe_vr_sims(nside_maps, nside_reconst, 0, mask = False) 

C1,C2,C3,C4 =EST.load_clqe_sim(realno,'vr',nside_maps , nside_reconst, 0, mask = False) 

plt.figure()
plt.loglog(np.arange(3*nside_reconst ), C1[:,0], label= 'reconstructed signal')
plt.loglog(np.arange(3*nside_reconst ), C2[:,0], label= 'true signal')
plt.loglog(np.arange(3*nside_reconst ), C3[:,0], label= 'difference between true and rec map')
plt.loglog(np.arange(3*nside_reconst ), C4[:,0], label= 'reconstruction noise')
plt.legend()
plt.savefig('plots/reconstruction_quality')

Ls = np.arange(1,11)

SN,R_P = EST.pmode_vrvr(lmax=3072,L=Ls,fine=False,cal=False)
plt.figure()
plt.plot(np.arange(EST.nbin), R_P[0,-1,:])
plt.savefig('plots/PC1')

plt.figure()
plt.plot(Ls,np.sum(SN,axis=1))
plt.savefig('plots/SNR')
########
### MAPS (including definitions for hp.anafast later)
########

###### KSZ ################################
kszmap = c.load(c.get_basic_conf(conf), EST.estim_dir+'sims/ksz_1024_real=0')
ksz_cls = hp.anafast(kszmap)

hp.mollview(kszmap,title='kSZ')
plt.savefig('plots/kszmap.png')
plt.close()




plt.figure()
plt.loglog(EST.Cls['kSZ-kSZ'][:,0,0], label='Theory')
plt.loglog(ksz_cls, label='Map')
plt.legend()
plt.savefig('plots/kszspec.png')



#########################################





#### UNWISE LSS AND TAUdot MAPS ###################################
hp.mollview(c.load(c.get_basic_conf(conf),EST.estim_dir+'sims/lss_vr_1024_real=%d_bin=0' % realno)[0])
plt.savefig('plots/unwiseLSS')

#gmap = c.load(c.get_basic_conf(conf), EST.estim_dir+'sims/lss_vr_1024_real=0_bin=0')
#Tmap = c.load(c.get_basic_conf(conf), EST.estim_dir+'sims/T0_1024_real=0_bin=0')


###### RECONSTRUCTIONS################################
vactual = c.load(c.get_basic_conf(conf),EST.estim_dir+'sims/vr_actual_1024_64_real=%d' % realno)
vrecon = c.load(c.get_basic_conf(conf),EST.estim_dir+'sims/qe_vr_1024_64_full_real=%d_mask=False_nlevel=0_lcut3072' % realno)

# Use rotation matrix to bring v actual map to corrected v actual map
# ??? Would also use pmode weights to balance the recon map, but negative eigenvalues for this selection
# !!! Can add white noise at low level to R1 to avoid negative eigenvalues
ellcut = 10
weights = np.zeros((3*64,EST.nbin))
vact_alms = np.array([hp.map2alm(vactual[binno]) for binno in np.arange(EST.nbin)])
vact_rot_alms = np.zeros(shape=vact_alms.shape,dtype='complex128')
for ell in np.arange(ellcut+1):
    R_at_ell = EST.R_ell(64*3,'vr','vr',ell)
    filtered_alms = np.zeros(shape=vact_alms.shape,dtype='complex128')
    for binno in np.arange(EST.nbin):
        filtered_alms[binno] = hp.almxfl(vact_alms[binno], [0 if l != ell else 1 for l in np.arange(64*3)])
    vact_rot_alms += np.dot(np.complex128(R_at_ell), filtered_alms)

actual_plot = np.zeros((EST.nbin, 12*64**2))
for binno in np.arange(EST.nbin):
    actual_plot[binno] = hp.alm2map(vact_rot_alms[binno], 64)


for binno in np.arange(8):        
    reconstruct_plot = hp.alm2map(hp.almxfl(hp.map2alm(vrecon[binno]), [0 if ell > ellcut else 1 for ell in np.arange(64*3)]), 64)
    actual_rotated_plot = actual_plot[binno]
    hp.mollview(reconstruct_plot,title='recon')
    plt.savefig('plots/recon_%d'%binno)
    hp.mollview(actual_rotated_plot,title='actual')
    plt.savefig('plots/actual_%d'%binno)
########################################################


weights = np.abs(np.sum(R_P[:,-1,:],axis=0) / np.linalg.norm(np.sum(R_P[:,-1,:],axis=0)))

plt.figure()
hp.mollview(hp.alm2map(hp.almxfl(hp.map2alm(np.dot(weights[np.newaxis,:],vrecon)[0]), [0 if ell > ellcut else 1 for ell in np.arange(64*3)]), 64))
plt.savefig('plots/combo_recon')
actual_rotated_plot = np.mean(actual_plot,axis=0)
hp.mollview(actual_rotated_plot)
plt.savefig('plots/combo_actual')
'''
###########
#############
############# TEST AREA 
############
###########
'''
#### Make sure coarse graining of vmaps, taudmaps is correct
import common as c
import config as conf
import estim
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import loginterp
plt.switch_backend('agg')

realno = 0
nside_maps = 1024
nside_reconst = 64
EST = estim.estimator(conf_module = conf, data_lmax = 3072)
EST.set_Nfine(64)

lmax=3072
nside=1024
self=EST
healpy=hp

#EST.get_qe_vr_sims(nside_maps , nside_reconst , 0, use_cleaned = False, frequency = None, mask = False)
#sims, alms = self.get_maps_and_alms(['taud','g','vr'],nside,3*nside)  # Run new maps
#np.savez('data/testing/coarse_grain_testing_vr_taud_maps', taudmaps_fine = sims[0], vmaps_fine=sims[2])  # Save maps to save time
cached_data = np.load('data/testing/coarse_grain_testing_vr_taud_maps.npz')  # Load saved maps
taudmaps_fine = cached_data['taudmaps_fine']  # UNWISE
vmaps_fine    = cached_data['vmaps_fine']  #UNWISE: Make SURE list number agrees with label number

vmaps = np.zeros((self.nbin, healpy.nside2npix(nside)))  # UNWISE
taudmaps = np.zeros((self.nbin, healpy.nside2npix(nside)))  # UNWISE
for i in np.arange(self.nbin):  # UNWISE
    vmaps[i,:] = np.mean(vmaps_fine[self.nbin*i:self.nbin*(i+1),:], axis=0)  # UNWISE
    taudmaps[i,:] = np.sum(taudmaps_fine[self.nbin*i:self.nbin*(i+1),:], axis=0) / (self.N_fine_modes/self.nbin)

theory = loginterp.log_interpolate_matrix(EST.load_theory_Cl('taud','taud'),EST.load_L())
vtheory = loginterp.log_interpolate_matrix(EST.load_theory_Cl('vr','vr'),EST.load_L())

self.conf.N_bins = self.N_fine_modes  # UNWISE
theory_fine = loginterp.log_interpolate_matrix(EST.load_theory_Cl('taud','taud'),EST.load_L())
vtheory_fine = loginterp.log_interpolate_matrix(EST.load_theory_Cl('vr','vr'),EST.load_L())
self.conf.N_bins = self.nbin

examine_bins = np.array([0,self.nbin-1,self.N_fine_modes-1])
taud_selection = np.zeros((examine_bins[np.where(examine_bins<self.nbin)].size, 3072))
vmap_selection = np.zeros((examine_bins[np.where(examine_bins<self.nbin)].size, 3072))
taudfine_selection = np.zeros((examine_bins.size, 3072))
vmapfine_selection = np.zeros((examine_bins.size, 3072))
for i, b in enumerate(examine_bins):
    if b < self.nbin:
        taud_selection[i] = hp.anafast(taudmaps[b])
        vmap_selection[i] = hp.anafast(vmaps[b])
    taudfine_selection[i] = hp.anafast(taudmaps_fine[b])
    vmapfine_selection[i] = hp.anafast(vmaps_fine[b])     

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10), sharex=True)
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default line colour list
for i, b in enumerate(examine_bins):
    taud_fine = taudfine_selection[i]
    vmap_fine = vmapfine_selection[i]
    if b < self.nbin:
        taud = taud_selection[i]
        vmap = vmap_selection[i]
        ax1.loglog(np.arange(1,3072), taud[1:], c=colours[i], label='Bin %d'%(b+1))
        ax2.loglog(np.arange(1,3072), vmap[1:], c=colours[i], label='Bin %d'%(b+1))
        ax1.loglog(np.arange(1,3073), theory[1:,b,b], c=colours[i], ls='--')
        ax2.loglog(np.arange(1,3073), vtheory[1:,b,b], c=colours[i], ls='--')
    finelabel = 'Bin %d' % (b+1)
    ax3.loglog(np.arange(1,3072), taud_fine[1:], c=colours[i], label=finelabel)
    ax4.loglog(np.arange(1,3072), vmap_fine[1:], c=colours[i], label=finelabel)
    ax3.loglog(np.arange(1,3073), theory_fine[1:,b,b], c=colours[i], ls='--')
    ax4.loglog(np.arange(1,3073), vtheory_fine[1:,b,b], c=colours[i], ls='--')

ax3.legend()
ax1.set_title('tau maps')
ax2.set_title('vr maps')
ax3.set_title('tau maps (fine)')
ax4.set_title('vr maps (fine)')
fig.text(0.04, 0.5, r'$C_\ell^{\mathrm{XX}}$', va='center', rotation='vertical')
ax3.set_xlabel(r'$\ell$')
ax4.set_xlabel(r'$\ell$')
plt.savefig('plots/coarse_grain_testing')
plt.close('all')
        




























### Make sure we produce properly correlated maps.
## Does order of labels matter? v,t,g indices produce garbage beyond v but so far t,g,v indices produce fine maps as long as vbin <=10
labels=['taud','g','vr']
lswitch=309


dim_list = []
labels = ['vr','taud','g']
lswitch = 3073
for lab in labels:
    dim_list.append(EST.load_theory_Cl(lab,lab).shape[1])

print("formatting covmat")

cltot = EST.covmat_healpy(labels, lswitch)

clmatrix = np.zeros((lswitch,129,129))
pos = 0
for i in np.arange(129):
    for j in np.arange(i, 129):
        clmatrix[:,i,j] = cltot[pos]
        pos += 1

vbin = 30
testmatrix = np.zeros((lswitch,65+vbin,65+vbin))
testmatrix[:,:65,:65] = clmatrix[:,64:,64:]
for vb in np.arange(vbin):
    testmatrix[:,:65,65+vb] = clmatrix[:,vb,64:]

testmatrix[:,65:,65:] = clmatrix[:,:vbin,:vbin]

clnew = []
for i in np.arange(65+vbin):
    for j in np.arange(i, 65+vbin):
        clnew.append(testmatrix[:,i,j])

testmaps = hp.synalm(clnew,lswitch,new=False)

for i in np.arange(testmaps.shape[0]):
    if 0 <= i <= 63:
        title = 'taud bin %d' % (i+1)
    elif i == 64:
        title = 'g (one bin)'
    else:
        title = 'vr bin %d' % (i-64)
    plt.figure()
    hp.mollview(hp.alm2map(testmaps[i], 1024),title=title)
    plt.savefig('plots/mapdump/synthmaps_%d'%i)
    plt.close('all')

#####
#############










'''
