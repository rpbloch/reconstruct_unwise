import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import config as conf
import common as c
import estim

plt.switch_backend('agg')
plt.rcParams.update({'axes.labelsize' : 12, 'axes.titlesize' : 16, 'figure.titlesize' : 16})

lmax = 3072
lfilt = 3
Ls = [1,2,3,4,5,6,10,30,60,150,191]
fine = False
cal = False
nside = 1024
nsideout = 64
n_growth = 10  # mask growth iteration
use_cleaned = False
mask = False
frequency = 143
CMBfac = 1.0

EST = estim.estimator(conf_module = conf, data_lmax = 3072)
EST.set_Nfine(32)

lowpass = lambda MAP, ellcut : hp.alm2map(hp.almxfl(hp.map2alm(MAP), [0 if ell > ellcut else 1 for ell in np.arange(nsideout*3)]),nsideout)

outdir = 'plots/plots_%dx%d/PCA/kSZ+%.2fpCMB_mask=%s_cleaned=%s/'% (EST.nbin, EST.N_fine_modes, CMBfac, mask, use_cleaned)

if not os.path.exists(outdir):
    os.makedirs(outdir)

if mask:
    if os.path.exists('data/mask_%d_growth.npy'%nside):
        mask_map = np.load('data/mask_%d_growth.npy'%nside)
    else:
        mask_map_orig = np.load('data/mask_'+str(nside)+'.npy')
        mask_map = mask_map_orig.copy()
        print('growing mask...')
        print(' 1.0000',end=' -> ')
        for j in np.arange(n_growth):
            mask_map_iter = mask_map.copy()
            for pix in np.arange(mask_map_iter.size):
                if not mask_map_iter[pix]:
                    pixel_ids = hp.get_all_neighbours(nside,pix)
                    mask_map[pixel_ids] = 0
            print('%.4f' % (np.where(mask_map==0)[0].size/np.where(mask_map_orig==0)[0].size),end=' -> ')
        print('done!')
        np.save('data/mask_%d_growth'%nside, mask_map)
        hp.mollview(mask_map_orig,title='mask (n_growth = 0)')
        plt.savefig(outdir+'mask_original')
    hp.mollview(mask_map,title='mask (n_growth = %d)'%n_growth)
    plt.savefig(outdir+'mask_grown')    
    fsky = 1-(np.where(mask_map==0)[0].size/mask_map.size)
else:
    fsky = 1.

EST.get_qe_vr_sims(nside , nsideout , 0, use_cleaned = use_cleaned, frequency = frequency, mask = mask)
EST.get_clqe_vr_sims(nside, nsideout, 0, mask = mask) 

vrecon, recon_noise =  EST.load_qe_sims_maps(0,'vr', nside, nsideout, 0, mask = mask)
if mask:
    mask_map_recon = hp.ud_grade(mask_map, nsideout)[np.newaxis,:]
    vrecon *= mask_map_recon
    recon_noise *= mask_map_recon
vactual = c.load(c.get_basic_conf(conf),EST.estim_dir+'sims/vr_actual_1024_64_real=0')
vactual_rot = np.zeros(shape=vactual.shape)

weighted_recon_binned = np.zeros(shape=vrecon.shape)
actual_rotated_plot_binned = np.zeros(shape=vactual.shape)
weighted_noise_binned = np.zeros(shape=recon_noise.shape)

R_at_ell = np.zeros((EST.nbin,EST.nbin))
for i in np.arange(EST.nbin):
    for j in np.arange(EST.nbin):
        R_at_ell[i,j] = EST.R_vr_unwise(lmax, i, j, 2)
SN,R_P = EST.pmode_vrvr(lmax=lmax,L=Ls,fine=False,cal=False)
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

theorynoise = np.zeros((EST.nbin, EST.nbin))
for alpha in np.arange(EST.nbin):
    for gamma in np.arange(EST.nbin):
        theorynoise[alpha,gamma] = EST.Noise_vr_matrix(lmax,alpha,gamma,2)

plt.figure()
plt.loglog(np.arange(1,3*nsideout ), hp.anafast(real_out_map)[1:], label='weighted signal')
#plt.loglog(np.arange(1,3*nsideout ), hp.anafast(real_actual_map)[1:], label= 'weighted true signal')
plt.loglog(np.arange(1,3*nsideout ), hp.anafast(real_noise)[1:], label= 'weighted noise')
plt.loglog([0,nsideout*3], [fsky*np.dot(np.dot(weights,theorynoise),weights.T)]*2,label='theory noise')
plt.legend()
#plt.ylim([1e-8,5e-7])
plt.title('Reconstruction quality of weighted sums')
plt.savefig(outdir + 'reconstruction_quality')







################################
########                 #######
########    TEST AREA    #######
########                 #######
################################
#newnoise = np.zeros((EST.nbin, EST.nbin))
#for alpha in np.arange(EST.nbin):
    #newnoise[alpha,alpha] = EST.Noise_vr_matrix(lmax,alpha,alpha,2)
##    for gamma in np.arange(EST.nbin):
##        newnoise[alpha,gamma] = EST.Noise_vr_matrix(lmax,alpha,gamma,2)
#noise_vector = np.diag(newnoise)
##cheat_weights = 1 / noise_vector / np.linalg.norm(1/noise_vector)
#noisemaps = np.zeros((EST.nbin,12*nsideout**2))
#for i in np.arange(EST.nbin):
    #noisemaps[i] = hp.synfast([noise_vector[i] for l in np.arange(3*nsideout)],nsideout)

#plt.figure()
#l1, = plt.loglog(np.arange(1,3*nsideout ), hp.anafast(np.sum(noisemaps*weights[:,np.newaxis],axis=0))[1:], label= 'weighted noise (diag)')
##l2, = plt.loglog(np.arange(1,3*nsideout ), hp.anafast(np.sum(noisemaps*cheat_weights[:,np.newaxis],axis=0)), label= 'weighted cheat noise')
#l3, = plt.loglog(np.arange(1,3*nsideout ), hp.anafast(real_noise)[1:], label= 'weighted noise (full)')
#plt.loglog([1,3*nsideout],         [fsky*np.sum(noise_vector*(weights**2))] * 2,               c=l1.get_c(), ls='--')
##plt.loglog([1,3*nsideout],         [fsky*np.sum(noise_vector*cheat_weights)] * 2,         c=l2.get_c())
#plt.loglog(np.arange(1,3*nsideout ), fsky*np.sum(theorynoise*(weights[:,np.newaxis])**2,axis=0)[1:], c=l3.get_c(), ls=':')
#plt.legend()
##plt.ylim([hp.anafast(real_noise)[3:].min(),hp.anafast(real_actual_map).max()])
##plt.title('Reconstruction quality of weighted sums')
#plt.savefig(outdir + 'noise_offset_plot')






# Plot original reconstruction in bin 1
#bin0_recon = lowpass(vrecon[0,:],lfilt)
#bin0_actual_rot = lowpass(vactual[0,:],lfilt)
hp.mollview(vrecon[0,:],title='Reconstructed bin 1/%d'%EST.nbin)
plt.savefig(outdir + 'qe_recon_bin0')
hp.mollview(vactual[0,:],title='Actual bin 1/%d'%EST.nbin)
plt.savefig(outdir + 'qe_actual_bin0')

C1,C2,C3,C4 =EST.load_clqe_sim(0,'vr',nside , nsideout, 0, mask = mask) 
plt.figure()
plt.loglog(np.arange(1,3*nsideout ), C1[1:,0], label= 'reconstructed signal')
#plt.loglog(np.arange(3*nsideout ), C2[:,0], label= 'true signal')
plt.loglog([1,3*nsideout], [fsky*theorynoise[0,0]]*2,label='theory noise')
plt.loglog(np.arange(1,3*nsideout ), C4[1:,0], label= 'reconstruction noise')
plt.legend()
#plt.ylim([C4[3:,0].min(),C2[3:,0].max()*3])
plt.title('Bin 1/%d' % EST.nbin)
plt.savefig(outdir + 'reconstruction_quality_bin0')

# Plot principle components
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
for i in np.arange(3):
    l, = ax1.plot(EST.zb.zbins_zcentral, R_P[0,-(i+1),:],label='PC%d'%(i+1))
    ax2.plot(EST.zb.zbins_zcentral, R_P[0,-(i+1),:]/np.linalg.norm(R_P[0,-(i+1),:]),c = l.get_c())
fig.legend(bbox_to_anchor=(1.,0.5))
fig.suptitle('Principle Components')
ax1.set_xlabel('z',fontsize=12)
ax2.set_xlabel('z')
ax1.set_title('unnormalized',fontsize=12)
ax2.set_title('normalized',fontsize=12)
fig.tight_layout(rect=[0,0,.9,1])
fig.savefig(outdir + 'principle_components')

## Plot input vs theory spectra
lssmap = c.load(c.get_basic_conf(conf), 'lss_vr_1024_real=0_bin=0', dir_base=EST.estim_dir+'sims')[0]  # unWISE has only 1 lss bin
kszmap = c.load(c.get_basic_conf(conf), 'ksz_1024_real=0', dir_base=EST.estim_dir+'sims')
Tmap = c.load(EST.basic_conf_dir,EST.Ttag+'_'+str(nside)+'_real=0_bin='+str(0), dir_base = EST.estim_dir+'sims')

plt.figure()
plt.loglog(np.arange(100,3072), hp.anafast(kszmap)[100:])
plt.loglog(np.arange(100,3073), EST.Cls['kSZ-kSZ'][100:,0,0])
plt.title('kSZ spectra')
plt.savefig(outdir+'kszspec')

plt.figure()
plt.loglog(np.arange(100,3072), hp.anafast(kszmap+Tmap)[100:])
plt.loglog(np.arange(100,3073), EST.Cls['T-T'][100:,0,0])
plt.title('Temperature spectrum')
plt.savefig(outdir + 'Tspec')

plt.figure()
plt.loglog(np.arange(100,3072), hp.anafast(lssmap)[100:])
plt.loglog(np.arange(100,3073), EST.Cls['lss-lss'][100:,0,0])
plt.title('LSS spectra')
plt.savefig(outdir+'lssspec')

plt.figure()
for b in [0,conf.N_bins//2,conf.N_bins-1]:
    l, = plt.loglog(np.arange(100,3073), EST.Cls['taud-taud'][100:,b,b],ls='--')
    plt.loglog(np.arange(100,3072), hp.anafast(c.load(c.get_basic_conf(conf),'taudmaps_1024_64_real=0_bin=%d'%b,EST.estim_dir+'sims'))[100:],c=l.get_c(), label='bin %d'%(b+1)) 
plt.legend()
plt.title('tau spec')
plt.savefig(outdir+'taudspec')

fig, ((ax1, ax2, ax3)) = plt.subplots(1,3,sharey=True,figsize=(18,6))
lmin_plot = 1
lmax_plot = 190
spacing = {64 : [[0,5,10],[26,31,36],[53,58,63]], 32 : [[0, 3, 6], [12,15,18],[25,28,31]], 8 : [[0,1,2],[2,3,4],[5,6,7]], 4 : [[0],[1],[2,3]]}
for b in spacing[conf.N_bins][0]:
    l, = ax1.loglog(np.arange(lmin_plot,lmax_plot), hp.anafast(vactual[b])[lmin_plot:lmax_plot], label='bin %d'%(b+1))
    ax1.loglog(np.arange(lmin_plot,lmax_plot+1), EST.Cls['vr-vr'][lmin_plot:lmax_plot+1,b,b],ls='--',c=l.get_c())
for b in spacing[conf.N_bins][1]:
    l, = ax2.loglog(np.arange(lmin_plot,lmax_plot), hp.anafast(vactual[b])[lmin_plot:lmax_plot], label='bin %d'%(b+1))
    ax2.loglog(np.arange(lmin_plot,lmax_plot+1), EST.Cls['vr-vr'][lmin_plot:lmax_plot+1,b,b],ls='--',c=l.get_c())
for b in spacing[conf.N_bins][2]:
    l, = ax3.loglog(np.arange(lmin_plot,lmax_plot), hp.anafast(vactual[b])[lmin_plot:lmax_plot], label='bin %d'%(b+1))
    ax3.loglog(np.arange(lmin_plot,lmax_plot+1), EST.Cls['vr-vr'][lmin_plot:lmax_plot+1,b,b],ls='--',c=l.get_c())

for ax in [ax1,ax2,ax3]:
    ax.set_xlabel(r'$\ell$')
    ax.legend()

plt.suptitle('vr spec')
plt.savefig(outdir+'vr_actualspec')

# Plot maps
hp.mollview(lssmap,title='lss')
plt.savefig(outdir+'lss')
plt.close()
hp.mollview(kszmap,title='kSZ')
plt.savefig(outdir+'ksz')
plt.close()
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


