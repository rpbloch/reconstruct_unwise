import numpy as np
import matplotlib.pyplot as plt
from spectra import *
plt.switch_backend('agg')

def n(z):
    ng = 40  # 40 / arcmin**2
    z0 = 0.3
    return (ng/(2*z0)) * (z/z0)**2 * np.exp(-z/z0)


fq1=fq2=None
tag1=tag2='g'
Pks = power.pks(conf_module = conf)
Pks.get_pks(tag1,tag2,fq1,fq2)
Pks.start_halo()
MthreshHODstellar = interp1d(Pks.hmod.z,Pks.mthreshHODstellar)(zb.zbins_zcentral)
ngalMpc3z = Pks.hmod.nbar_galaxy(zb.zbins_zcentral,Pks.hmod.logmmin,Pks.hmod.logmmax,MthreshHODstellar)

dz = np.diff(zb.zbins_z) #get the density over the red shift bin (assuming it is constant within the bin).
ngalarcmin2Binned_nodz = Pks.hmod.convert_n_mpc3_arcmin2(ngalMpc3z,zb.zbins_zcentral)
ngalarcmin2Binned = Pks.hmod.convert_n_mpc3_arcmin2(ngalMpc3z,zb.zbins_zcentral) * dz

ns = n(zb.zbins_zcentral)

plt.figure()
plt.loglog(zb.zbins_zcentral,ns, label='Theory')
plt.loglog(zb.zbins_zcentral, ngalarcmin2Binned, label='HMOD*dz')
plt.loglog(zb.zbins_zcentral, ngalarcmin2Binned_nodz, label='HMOD')
plt.xlabel('z')
plt.ylabel('n(z)')
plt.legend()
plt.savefig('plots/LSST_number_density')



























'''
UNWISE DEBUG HISTORY

import common as c
import cosmology
import config as conf
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import loginterp
plt.switch_backend('agg')

with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
    lines = FILE.readlines()
    
ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float').astype('int')
clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')[:np.where(ells>3073)[0][0]]
ells = ells[:np.where(ells>3073)[0][0]]

#recco = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf), 'Cl_g_g_noshot_lmax=3072',dir_base='Cls/LSS=unwise_blue/'), EST.load_L())[ells,0,0]
#recco_full = EST.Cls['lss-lss'][ells,0,0]
#recco = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_noshot_lmax=3072','Cls/LSS=unwise_blue/'),EST.load_L())[ells,0,0]
recco_CAMBpk = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_CAMB_lmax=3072','Cls/LSS=unwise_blue/'),EST.load_L())[ells,0,0]
recco_HMODpk = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_NOTCAMB_lmax=3072','Cls/LSS=unwise_blue/'),EST.load_L())[ells,0,0]
#recco_test = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_lmax=3072','Cls/LSS=unwise_blue/'),EST.load_L())[ells,0,0]

plt.figure()
plt.loglog(ells,clgg,label='unWISE')
plt.loglog(ells, recco_CAMBpk, label='ReCCO (CAMB Pk)')
plt.loglog(ells, recco_HMODpk, label='ReCCO (ReCCO Pk)')
#plt.loglog(ells, recco_test, label='ReCCO (ReCCO Pk, safe interp)')
#plt.loglog(ells,recco+9.2e-8,label='ReCCO + unWISE shot noise',ls='--')
plt.legend()
plt.savefig('plots/Clgg mismatch')









import numpy as np
import camb
import config as conf
import cosmology
from scipy.interpolate import interp1d, interp2d

csm = cosmology.cosmology(conf_module=conf)

cambpars = camb.CAMBparams()
cambpars.set_cosmology(H0=conf.h*100,ombh2=conf.ombh2,omch2=conf.omch2,mnu=conf.mnu,omk=conf.Omega_K,tau=conf.tau,TCMB=conf.T_CMB/1e6)
cambpars.InitPower.set_params(As=conf.As*1e-9,ns=conf.ns,r=0)
cambpars.NonLinear = camb.model.NonLinear_both
cambpars.max_eta_k = 14000.0*conf.ks_hm[-1]

z = conf.zs_hm
k = conf.ks_hm
chi = csm.chi_from_z(z)

cambpars.set_matter_power(redshifts=z.tolist(), kmax=k[-1],k_per_logint=20)
camb_PK_nonlin = camb.get_matter_power_interpolator(cambpars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=k[-1], zmax=z[-1])

pgg = interp2d(k,z,camb_PK_nonlin.P(z,k)*(((0.8+1.2*z)[:,np.newaxis])**2),kind='linear',bounds_error=False,fill_value=0.)
camb_PK_lin = camb.get_matter_power_interpolator(cambpars, nonlinear=False,hubble_units=False, k_hunit=False, kmax=k[-1], zmax=z[-1])
pgg1 = interp2d(k,z,camb_PK_lin.P(z,k)*(((0.8+1.2*z)[:,np.newaxis])**2),kind='linear',bounds_error=False,fill_value=0.)


with open('data/unWISE/blue.txt', 'r') as FILE:
    x = FILE.readlines()
zs = np.array([float(l.split(' ')[0]) for l in x])
dndz = np.array([float(l.split(' ')[1]) for l in x])
window_g = interp1d(zs,dndz, kind= 'linear',bounds_error=False,fill_value=0)(z)*csm.H_z(z) 




lmax = 3072
clgg_manual = np.zeros(lmax+1)
for ell in np.arange(lmax):
    pgg_limber_ell = interp1d(chi, np.diag(pgg(((ell+0.5))/chi, z)+pgg1(((ell+0.5))/chi, z)))
    clgg_manual[ell] = np.trapz(window_g**2 * pgg_limber_ell(chi) / chi**2, chi)
    


















import camb
from camb import model
import config as conf
from scipy.interpolate import interp1d, interp2d
import numpy as np
import common as c
import matplotlib.pyplot as plt
import loginterp
plt.switch_backend('agg')

pars = camb.CAMBparams()
pars.set_cosmology(H0=conf.H0, ombh2=conf.ombh2, omch2=conf.omch2)
pars.InitPower.set_params(ns=conf.ns)
pars.set_matter_power(redshifts=conf.zs_hm, kmax=conf.ks_hm.max())
pars.NonLinear = model.NonLinear_both
data = camb.get_background(pars)

chis_hm = data.comoving_radial_distance(conf.zs_hm)
Pk_camb = camb.get_matter_power_interpolator(pars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=conf.ks_hm.max(), zmax=conf.zs_hm.max())
Pmm_camb_sampled = Pk_camb.P(conf.zs_hm, conf.ks_hm, grid=True)
Pmm_camb_sampled *= ((0.8+1.2*conf.zs_hm)[:,np.newaxis])**2
Pgg_camb = interp2d(conf.ks_hm, conf.zs_hm, Pmm_camb_sampled, kind='linear', bounds_error=False, fill_value=0.)
limber = lambda ell : np.diag(Pgg_camb((ell+0.5)/chis_hm,conf.zs_hm))




from camb.sources import GaussianSourceWindow, SplinedSourceWindow
pars.SourceTerms.counts_redshift=False
pars.SourceTerms.counts_velocity=False
pars.SourceTerms.time_delay=False
pars.SourceTerms.ISW=False
pars.SourceTerms.potential=False
pars.SourceWindows=[SplinedSourceWindow(bias_z=(0.8+1.2*zs),dlog10Ndm=+0.2,z=zs,W=dndz)]
results=camb.get_results(pars)
clgg_manual = results.get_source_cls_dict()['W1xW1']
f = ells[np.where(ells<=2501)]



with open('data/unWISE/blue.txt', 'r') as FILE:
    x = FILE.readlines()
zs = np.array([float(l.split(' ')[0]) for l in x])
dndz = np.array([float(l.split(' ')[1]) for l in x])
window_g = interp1d(zs,dndz, kind= 'linear',bounds_error=False,fill_value=0)(conf.zs_hm)*data.h_of_z(conf.zs_hm)
stuff  = window_g/np.trapz(window_g,chis_hm)


lmax = 3072
clgg_manual = np.zeros(lmax+1)
for ell in np.arange(lmax):
    clgg_manual[ell] = np.trapz(stuff*stuff* limber(ell) / chis_hm**2, chis_hm)





with open('data/unWISE/Bandpowers_Auto_Sample1.dat','rb') as FILE:
    lines = FILE.readlines()
    
ells = np.array(lines[0].decode('utf-8').split(' ')).astype('float').astype('int')
clgg = np.array(lines[1].decode('utf-8').split(' ')).astype('float')[:np.where(ells>3073)[0][0]]
ells = ells[:np.where(ells>3073)[0][0]]

recco_CAMBpk = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_CAMB_lmax=3072','Cls/LSS=unwise_blue/'),c.load(c.get_basic_conf(conf),'L_sample_lmax=3072','Cls'))[ells,0,0]
recco_HMODpk = loginterp.log_interpolate_matrix(c.load(c.get_basic_conf(conf),'Cl_g_g_NOTCAMB_lmax=3072','Cls/LSS=unwise_blue/'),c.load(c.get_basic_conf(conf),'L_sample_lmax=3072','Cls'))[ells,0,0]

plt.figure()
plt.loglog(ells,clgg,label='unWISE')
plt.loglog(ells, recco_CAMBpk, label='ReCCO (CAMB Pk)')
plt.loglog(ells, recco_HMODpk, label='ReCCO (ReCCO Pk)')
plt.loglog(f, (clgg_manual[f]/(f*(f+1)/2/np.pi))+9.2e-8, label='Manual Calc')
plt.loglog(ells,[9.2e-8 for item in ells], ls='--',c='grey',alpha=0.75)
plt.legend()
plt.savefig('plots/Clgg manual')







































csm = cosmology.cosmology(conf_module=conf)

recco_pk = c.load(c.get_basic_conf(conf), 'p_gg_f1=None_f2 =None','pks')(conf.ks_hm, conf.zs_hm)
camb_pk = csm.camb_Pk_nonlin(conf.ks_hm,conf.zs_hm) * ((0.8+1.2*conf.zs_hm)[:,np.newaxis])**2
f = camb.get_matter_power_interpolator(pars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=conf.ks_hm.max(), zmax=conf.zs_hm.max())

indices = [0,44,77,135]
fig, ax = plt.subplots(1,1)
for i in indices:
    #l, = ax.loglog(conf.ks_hm, recco_pk[i,:], label='z=%.2f'%conf.zs_hm[i],ls='-')
    #l, = ax.loglog(conf.ks_hm, pgg_camb(conf.ks_hm, conf.zs_hm[i]), ls='-')
    l, = ax.loglog(conf.ks_hm, camb_pk[i,:], ls='-', label='z=%.2f'%conf.zs_hm[i])
    colour = l.get_c()
    ax.loglog(conf.ks_hm, Pgg_camb(conf.ks_hm, conf.zs_hm[i]), ls='--',c=colour)
plt.legend(title='solid = interpolator\ndashed = calculator')
ax.set_ylim([1e-4,ax.get_ylim()[-1]])
plt.savefig('plots/camb_diff')













data = np.load('/home/richard/Desktop/SZCOSMOS.npz')
sz_cosmo_1bin = loginterp.log_interpolate_matrix(data['clgg'],EST.load_L())[ells,0,0]
sz_cosmo_1bin_noshot = loginterp.log_interpolate_matrix(data['noshot'],EST.load_L())[ells,0,0]+9.2e-8

#cosmoplot = np.array([np.diag(szcosmo[ell,:,:]) for ell in ells])
plt.figure()
plt.loglog(ells,clgg,label='unWISE')
plt.loglog(ells,sz_cosmo_1bin,label='SZ Cosmo')
plt.loglog(ells,sz_cosmo_1bin_noshot,label='SZ Cosmo + uNWISE shot noise', ls='--')
#plt.loglog(ells, recco_full, label='ReCCO')
#plt.loglog(ells,recco+9.2e-8,label='ReCCO + unWISE shot noise',ls='--')
plt.legend()
plt.savefig('plots/SZ_Cosmo mismatch')
'''