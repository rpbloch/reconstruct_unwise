'''
mask_map = hp.ud_grade(np.load('data/mask_%d_growth.npy'%1024), 2048)
plancksimTT = hp.anafast(mask_map*fits.open('data/planck_data_testing/sims/total/ffp10_newdust_total_143_full_map.fits')[1].data['TEMPERATURE'].flatten())
planckmapTT = hp.anafast(mask_map*hp.reorder(fits.open('data/planck_data_testing/maps/HFI_SkyMap_143_2048_R3.01_full.fits')[1].data['I_STOKES'].flatten(),n2r=True))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,5))
ax1.loglog(np.arange(101,3072), fsky*EST.Cls['T-T'][101:3072,0,0], label='ReCCO')
ax1.loglog(np.arange(101,6140), plancksimTT[101:6140],label='Planck SIM')
ax1.loglog(np.arange(101,6140), planckmapTT[101:6140],label='Planck MAP')
ax1.legend()
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$C_\ell^{\mathrm{TT}}\quad \left[\mathrm{{K_{CMB}}^2}\right]$',labelpad=0.5)

ax2.loglog(np.arange(1500,3072), fsky*EST.Cls['T-T'][1500:3072,0,0], label='ReCCO')
ax2.loglog(np.arange(1500,6140), plancksimTT[1500:6140],label='Planck SIM')
freezeylo, freezeyhi = ax2.get_ylim()
ax2.loglog(np.arange(1500,6140), planckmapTT[1500:6140],label='Planck MAP')
ax2.set_ylim([freezeylo, freezeyhi])
ax2.legend()
ax2.set_xlabel(r'$\ell$')
fig.suptitle('TT Spectra at 143 GHz')
plt.savefig('cmbs')


'''


import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import config as conf
import common as c
import estim
from astropy.io import fits

plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})

nside = 2048
nsideout = 64
lmax = 3*nside
lfilt = 3
Ls = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
fine = False
cal = False
n_growth = 5  # mask growth iteration
use_cleaned = False
mask = False
frequency = 143

EST = estim.estimator(conf_module = conf, data_lmax = lmax)
EST.set_Nfine(32)
#EST.realnum = 30
#EST.maskfile = 'data/mask_unwisethres_planckgal60union.npy'
#EST.maskfile = 'data/mask_planckgal70.npy'
mask = str(mask) + '[@' + EST.maskfile.replace('/','>>') + ']'
outdir = 'plots/N%d/plots_%dx%d/lss-gauss_TT-143sky/Nout%d/mask=%s/'% (nside,EST.nbin, EST.N_fine_modes, nsideout, mask)

datadir = 'data/planck_data_testing/sims/'
mapdir = 'data/planck_data_testing/maps/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

mask_map = np.load(EST.maskfile)
#mask_map = fits.open('data/unWISE/mask_unWISE_thres_v10.fits')[1].data['T'].flatten()
if mask[:4] == 'True':
    fsky = 1-(np.where(mask_map==0)[0].size/mask_map.size)
else:
    fsky = 1.
print('Running setup...')
############ PREP #############
## MASK
#if mask:
    #mask_filename = 'mask_unWISE-binary_N1024_N64thresh0.5.npy'
    ##mask_filename = 'mask_unWISE-binary_N1024.npy'
    ##mask_filename = 'mask_1024.npy'
    #mask_filepath = 'data/' + mask_filename
    ##grown_mask_filepath = 'data/' + mask_filename.split('.npy')[0] + '_grown.npy'
    #grown_mask_filepath = 'placeholder12341234'
    #mask_map = np.load(mask_filepath)
    ##mask_map = np.load('data/mask_%d_growth.npy'%nside)
    ##mask_map = np.load('data/mask_'+str(nside)+'.npy')
    #fsky = 1-(np.where(mask_map==0)[0].size/mask_map.size)
    ##=========================
    ## for growing our own mask
    ##=========================
    #if os.path.exists(grown_mask_filepath):
        #mask_map = np.load(grown_mask_filepath)
    #else:
        #mask_map_orig = np.load(mask_filepath)
        ##mask_map = mask_map_orig.copy()
        ##print('growing mask...')
        ##print(' 1.0000',end=' -> ')
        ##for j in np.arange(n_growth):
            #mask_map_iter = mask_map.copy()
            #for pix in np.arange(mask_map_iter.size):
                #if not mask_map_iter[pix]:
                    #pixel_ids = hp.get_all_neighbours(nside,pix)
                    #mask_map[pixel_ids] = 0
            #print('%.4f' % (np.where(mask_map==0)[0].size/np.where(mask_map_orig==0)[0].size),end=' -> ')
        ##print('done!')
        ##np.save(grown_mask_filepath, mask_map)
        #hp.mollview(mask_map_orig,title='mask (n_growth = 0)')
        #plt.savefig(outdir+'mask_original')
    ##hp.mollview(mask_map,title='mask (n_growth = %d)'%n_growth)
    ##plt.savefig(outdir+'mask_grown')    
    ##fsky = 1-(np.where(mask_map==0)[0].size/mask_map.size)
    ##============================
    ##end section
    ##============================
#else:
    #fsky = 1.

#=============================
# Commented out below: stuff for adding planck elements bit-by-bit.
# commented out since using Planck data is changed at the estimator level
#=================================
# NOISE (add to CMB for high \ell data)
#data_nside = 2048
#noise_files = [datadir+'noise/143ghz/'+filename for filename in os.listdir(datadir+'noise/143ghz/') if '.fit' in filename]
#noise_files.sort()
#if not os.path.exists(datadir + 'noise_143ghz_Nl_100sims.npy'):
    #noisemaps = np.zeros((len(noise_files), hp.nside2npix(data_nside)))
    #noise_cls = np.zeros((len(noise_files), 3*data_nside))
    #for i, filename in enumerate(noise_files):
        #print('taking spectrum for noisemap %d' % (i+1))
        #noisemaps[i,:] = fits.open(filename)[1].data['I_STOKES'].flatten()
        #noise_cls[i,:] = hp.anafast(noisemaps[i,:])
    #np.save(datadir + 'noise_143ghz_Nl_100sims', np.mean(noise_cls,axis=0))
    #noisemap = noisemaps[0].copy()
#else:
    #noise_cls = np.load(datadir + 'noise_143ghz_Nl_100sims.npy')[np.newaxis,:]
    #noisemap = fits.open(noise_files[0])[1].data['I_STOKES'].flatten()
    
## pCMB: converges waaaay faster than noise, 40 sims more than enough to get a smooth line.
##pcmb_files = [datadir+'pCMB/143ghz/'+filename for filename in os.listdir(datadir+'pCMB/143ghz/') if '.fit' in filename]
##pcmb_files.sort()
##if not os.path.exists(datadir + 'pCMB_143ghz_ClTT_40sims.npy'):
    ##cmbmaps = np.zeros((len(pcmb_files), hp.nside2npix(data_nside)))
    ##cmb_cls = np.zeros((len(pcmb_files), 3*data_nside))
    ##for i, filename in enumerate(pcmb_files):
        ##print('taking spectrum for cmbmap %d' % (i+1))
        ##cmbmaps[i,:] = fits.open(filename)[1].data['TEMPERATURE'].flatten()
        ##cmb_cls[i,:] = hp.anafast(cmbmaps[i,:])
    ##np.save(datadir + 'pCMB_143ghz_ClTT_40sims', np.mean(cmb_cls,axis=0))
    ##Tmap = cmbmaps[0].copy()
    ##cltt = np.mean(cmb_cls, axis=0)
##else:
    ##cmb_cls = np.load(datadir + 'pCMB_143ghz_ClTT_40sims.npy')[np.newaxis,:]
    ##Tmap = fits.open(pcmb_files[0])[1].data['TEMPERATURE'].flatten()    
    ##cltt = np.mean(cmb_cls, axis=0)

#plt.figure()
#plt.loglog(np.arange(2,3*data_nside), hp.anafast(noisemap)[2:], label='map')
#plt.loglog(np.arange(2,3*data_nside), np.mean(noise_cls, axis=0)[2:], label='theory')
#plt.legend()
#plt.savefig(outdir + 'planck_noise_spec')

##plt.figure()
##plt.loglog(np.arange(2,3*data_nside), hp.anafast(Tmap)[2:], label='map')
##plt.loglog(np.arange(2,3*data_nside), np.mean(cmb_cls, axis=0)[2:], label='theory')
##plt.legend()
##plt.savefig(outdir + 'planck_pCMB_spec')

## COMBINING TT AND NOISE FOR CONSISTENCY
#print('Computing TT offset...')
##cltt = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_pCMB_pCMB_lmax='+str(lmax), dir_base = 'Cls/'+c.direc('pCMB','pCMB',conf)), c.load(c.get_basic_conf(conf),'L_pCMB_lmax='+str(lmax), dir_base = 'Cls'))[:,0,0]
###### PUT IN cltt from 100 CMB sims, like in noise.
#Tmap = hp.ud_grade(hp.reorder(fits.open('data/planck_data_testing/COM_CMB_IQU-smica_2048_R3.00_full.fits')[1].data['I_STOKES_INP'], n2r=True), nside)

#cltt = np.ones(lmax)*1e-30
#prefac = np.arange(lmax)*(np.arange(lmax)+1)/2/np.pi
#with open(datadir + '../COM_PowerSpect_CMB-TT-full_R3.01.txt') as FILE:
    #lines = FILE.readlines()[1:]
#for line in lines:
    #ell, ttval = [float(item) for item in line.split(' ') if item != ''][:2]
    #cltt[int(ell)] = ttval / (prefac[int(ell)] * 1e12)  # convert from uK_CMB to K_CMB, from Dl to Cl

### Change theory Cl pCMB to come up to Planck
#offset_magnitude = np.median(hp.anafast(Tmap,lmax=799)[5:]/cltt[5:800])  # ONLY FOR LEGIT CMB MAP, otherwise 1 for cmb sims
### Account for the fact that theory Cl then gets too big in the high \ell regime because it's just noise
#offset_shape = hp.gauss_beam(np.radians(8/60),lmax=lmax)  # ONLY FOR LEGIT CMB MAP, otherwise 1 for cmb sims

#np.save(datadir + 'planck_TT_offset', offset_magnitude*offset_shape)

#total_cl = offset_magnitude*offset_shape[:3*nside]*cltt[:3*nside] + np.mean(noise_cls,axis=0)[:3*nside]
#total_map = Tmap + hp.ud_grade(noisemap, nside)

#plt.figure()
#plt.loglog(hp.anafast(total_map)[2:])
#plt.loglog(total_cl[2:])
#plt.savefig(outdir + 'TT_spec')


#===============================
# End section
#==============================



##### RUN THE THING
print('Running estimator.')
lowpass = lambda MAP, ellcut : hp.alm2map(hp.almxfl(hp.map2alm(MAP), [0 if ell > ellcut else 1 for ell in np.arange(nsideout*3)]),nsideout)

EST.get_qe_vr_sims(nside , nsideout , 0, use_cleaned = use_cleaned, frequency = frequency, mask = mask)

print('===POSTCOMP===')
EST.get_clqe_vr_sims(nside, nsideout, 0, mask = mask) 

vrecon, recon_noise =  EST.load_qe_sims_maps(0,'vr', nside, nsideout, 0, mask = mask)
if mask[:4]=='True':
    mask_map_recon = hp.ud_grade(mask_map, nsideout)[np.newaxis,:]
    if 1 not in mask_map_recon:  # Happens weirdly for large to small nside/nsideout
        mask_map_recon = hp.ud_grade(hp.ud_grade(mask_map,int(hp.npix2nside(mask_map.size)/((hp.npix2nside(mask_map.size)/nsideout)/4))),nsideout)
    vrecon *= mask_map_recon
    recon_noise *= mask_map_recon
vactual = c.load(c.get_basic_conf(conf),EST.estim_dir+'sims/vr_actual_%d_%d_real=0' % (nside,nsideout))
vactual_rot = np.zeros(shape=vactual.shape)

print('PCA')
#R_at_ell = np.zeros((EST.nbin,EST.nbin))
#for i in np.arange(EST.nbin):
#    for j in np.arange(EST.nbin):
#        R_at_ell[i,j] = EST.R_vr_unwise(lmax, i, j, 2)
R_at_ell = c.load(EST.basic_conf_dir,'Rvrvr_'+str(nside)+'_'+str(nsideout)+'_lcut'+str(lmax), dir_base = EST.estim_dir+'sims')
SN,R_P = EST.pmode_vrvr(lmax=lmax,L=Ls,nside=nside, nsideout=nsideout, fine=False,cal=False)
vactual_rot_alms = np.dot(R_at_ell, np.array([hp.map2alm(vactual[b,:]) for b in np.arange(EST.nbin)]))
weights = np.abs(np.median(R_P[:,-1,:],axis=0)/np.linalg.norm((np.median(R_P[:,-1,:],axis=0))))
for b in np.arange(EST.nbin):
    vactual_rot[b,:] = hp.alm2map(vactual_rot_alms[b,:],nsideout)

real_out_map = np.sum(vrecon*weights[:,np.newaxis], axis=0)
real_actual_map = np.sum(vactual_rot*weights[:,np.newaxis], axis=0)
real_noise = np.sum(recon_noise*weights[:,np.newaxis], axis=0)

hp.mollview(lowpass(real_out_map,lfilt),title=r'reconstruction (lowpass $\ell\leq%d$'%lfilt)
plt.savefig(outdir+'qe_reconstructed_lowpass')
hp.mollview(real_out_map,title=r'reconstruction')
plt.savefig(outdir+'qe_reconstructed')
hp.mollview(lowpass(real_actual_map,lfilt),title=r'actual (rotated) (lowpass $\ell\leq%d$'%lfilt)
plt.savefig(outdir+'qe_actual_lowpass')
hp.mollview(real_actual_map,title=r'actual')
plt.savefig(outdir+'qe_actual')


Noise = c.load(EST.basic_conf_dir,'Nvrvr_'+str(nside)+'_'+str(nsideout)+'_lcut'+str(3*nside), dir_base = EST.estim_dir+'sims')
#print('Recomputing (diag) noise...')
# theorynoise = np.zeros((EST.nbin, EST.nbin))
# for alpha in np.arange(EST.nbin):
#     for gamma in np.arange(EST.nbin):
#         theorynoise[alpha,gamma] = EST.Noise_vr_matrix(lmax,alpha,gamma,2)


print('Plots...')
plt.figure()
plt.loglog(np.arange(3,3*nsideout ), hp.anafast(real_out_map)[3:], label='weighted signal')
#plt.loglog(np.arange(3,3*nsideout ), hp.anafast(real_actual_map)[3:], label= 'weighted true signal')
#plt.loglog(np.arange(3,3*nsideout ), hp.anafast(real_noise)[3:], label= 'weighted noise')  # With TT data commented out since it synfasts the sky power, which distributes the huge galaxy residual power across the sky creating a large magnitude reconstruction
#plt.loglog([0,nsideout*3], [fsky*np.dot(np.dot(weights,theorynoise),weights.T)]*2,label='theory noise')
plt.loglog(np.arange(nsideout*3), fsky*np.dot(weights[np.newaxis,:],Noise)[0], label='theory noise')
plt.legend()
#plt.ylim([1e-8,5e-7])
plt.title('Reconstruction quality of weighted sums')
plt.savefig(outdir + 'reconstruction_quality')

#==================
# Deprecated: with the PCA working we don't need to look at single bins each time
#=================
# Plot original reconstruction in bin 1
#bin0_recon = lowpass(vrecon[0,:],lfilt)
#bin0_actual_rot = lowpass(vactual[0,:],lfilt)
#hp.mollview(vrecon[0,:],title='Reconstructed bin 1/%d'%EST.nbin)
#plt.savefig(outdir + 'qe_recon_bin0')
#hp.mollview(vactual[0,:],title='Actual bin 1/%d'%EST.nbin)
#plt.savefig(outdir + 'qe_actual_bin0')

#C1,C2,C3,C4 =EST.load_clqe_sim(0,'vr',nside , nsideout, 0, mask = mask) 
#plt.figure()
#plt.loglog(np.arange(1,3*nsideout ), C1[1:,0], label= 'reconstructed signal')
##plt.loglog(np.arange(3*nsideout ), C2[:,0], label= 'true signal')
#plt.loglog([1,3*nsideout], [fsky*theorynoise[0,0]]*2,label='theory noise')
#plt.loglog(np.arange(1,3*nsideout ), C4[1:,0], label= 'reconstruction noise')
#plt.legend()
#plt.title('Bin 1/%d' % EST.nbin)
#plt.savefig(outdir + 'reconstruction_quality_bin0')
#===================
# End section
#==================



# Plot principle components
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
for i in np.arange(3):
    l, = ax1.plot(EST.zb.zbins_zcentral, R_P[0,-(i+1),:],label='PC%d'%(i+1))
    ax2.plot(EST.zb.zbins_zcentral, R_P[0,-(i+1),:]/np.linalg.norm(R_P[0,-(i+1),:]),c = l.get_c())
fig.legend(bbox_to_anchor=(1.,0.5))
fig.suptitle('Principle Components   [pCMB+143Ghz Noise+kSZ x unWISE]')
ax1.set_xlabel('z',fontsize=12)
ax2.set_xlabel('z')
ax1.set_title('unnormalized',fontsize=12)
ax2.set_title('normalized',fontsize=12)
fig.tight_layout(rect=[0,0,.9,1])
fig.savefig(outdir + 'principle_components')

## Plot input vs theory spectra
lssmap = c.load(c.get_basic_conf(conf), 'lss_vr_%d_real=0_bin=0'%nside, dir_base=EST.estim_dir+'sims')[0]  # unWISE has only 1 lss bin
#kszmap = c.load(c.get_basic_conf(conf), 'ksz_1024_real=0', dir_base=EST.estim_dir+'sims')
Tmap = c.load(EST.basic_conf_dir,EST.Ttag+'_'+str(nside)+'_real=0_bin='+str(0), dir_base = EST.estim_dir+'sims')

#plt.figure()
#plt.loglog(np.arange(100,3072), hp.anafast(kszmap)[100:])
#plt.loglog(np.arange(100,3073), EST.Cls['kSZ-kSZ'][100:,0,0])
#plt.title('kSZ spectra')
#plt.savefig(outdir+'kszspec')

plt.figure()
plt.loglog(np.arange(100,lmax), hp.anafast(Tmap)[100:])
plt.loglog(np.arange(100,EST.Cls['T-T'][:,0,0].size), EST.Cls['T-T'][100:,0,0])
plt.title('Temperature spectrum')
plt.savefig(outdir + 'Tspec')

plt.figure()
plt.loglog(np.arange(100,lmax), hp.anafast(lssmap)[100:])
plt.loglog(np.arange(100,EST.Cls['lss-lss'][:,0,0].size), EST.Cls['lss-lss'][100:,0,0])
plt.title('LSS spectra')
plt.savefig(outdir+'lssspec')

# Plot maps
hp.mollview(lssmap,title='lss')
plt.savefig(outdir+'lss')
plt.close()
#hp.mollview(kszmap,title='kSZ')
#plt.savefig(outdir+'ksz')
#plt.close()
hp.mollview(vrecon[0],title='recon')
plt.savefig(outdir+'recon')
plt.close()
hp.mollview(vactual[0],title='actual')
plt.savefig(outdir+'actual')
plt.close()
hp.mollview(recon_noise[0],title='noisemap')
plt.savefig(outdir+'noisemap')
plt.close('all')
hp.mollview(Tmap,title='T map')
plt.savefig(outdir+'Tmap')
plt.close()
if mask[:4]=='True':
    hp.mollview(mask_map, title='mask')
    plt.savefig(outdir+'mask')
    plt.close()
