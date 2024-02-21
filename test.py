
maplist.processed_alms['COMMANDER'] = maplist.mask_and_debeam(maplist.input_COMMANDER, np.ones(maplist.mask.size), maplist.SMICAbeam)
maplist.processed_alms['unWISE'] = hp.map2alm(maplist.input_unWISE, lmax=4000)

maplist.Cls['COMMANDER'] = maplist.alm2cl(maplist.mask_and_debeam(maplist.input_COMMANDER, maplist.mask_planck, maplist.SMICAbeam), maplist.fsky_planck)
maplist.Cls['unWISE'] = hp.alm2cl(maplist.processed_alms['unWISE'])

print('    Computing fiducial reconstructions')
noises['COMMANDER'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=maplist.Cls['COMMANDER'], clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
reconstructions['COMMANDER'] = combine_alm(, , maplist.mask, , , , noises['COMMANDER'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
recon_Cls['COMMANDER'] = maplist.alm2cl(hp.map2alm(reconstructions['COMMANDER'], lmax=recon_lmax), maplist.fsky)

# compare_to_off: variable stored holding power spec of Matt's method that
#                 gives same as our method.
#                 Use to compare to below to see if manual also matches or not.

Clgg_unWISE = hp.anafast(maplist.input_unWISE)
unwiselms = hp.map2alm(maplist.input_unWISE, lmax=maplist.lmax)

mapin_com = maplist.input_COMMANDER.copy()

mapin_planckmask = hp.ma(mapin_com)
mapin_planckmask.mask = np.logical_not(maplist.mask_planck)
ClTT_COMMANDER = hp.alm2cl(hp.almxfl(hp.map2alm(mapin_planckmask, lmax=maplist.lmax), 1/maplist.SMICAbeam)) / maplist.fsky_planck
theorynoise_com = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=ClTT_COMMANDER, clgg_binned=Clgg_unWISE, cltaudg_binned=cltaug_fiducial)

mapin_debeam_com = hp.almxfl(hp.map2alm(mapin_com, lmax=maplist.lmax), 1/maplist.SMICAbeam)

ClTT_filter = ClTT_COMMANDER.copy()[:maplist.lmax+1]
Clgg_filter = Clgg_unWISE.copy()[:maplist.lmax+1]
Cltaudg = cltaug_fiducial.copy()[:maplist.lmax+1]
ClTT_filter[:100] = 1e15
Clgg_filter[:100] = 1e15
dTlm_xi = hp.almxfl(mapin_debeam_com, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
dlm_zeta = hp.almxfl(unwiselms, np.divide(Cltaudg, Clgg_filter, out=np.zeros_like(Cltaudg), where=Clgg_filter!=0))
Tmap_filtered = hp.alm2map(dTlm_xi, lmax=maplist.lmax, nside=maplist.nside)
lssmap_filtered = hp.alm2map(dlm_zeta, lmax=maplist.lmax, nside=maplist.nside)
outmap_filtered = Tmap_filtered*lssmap_filtered
outmap_com = -outmap_filtered * theorynoise_com

outmap_masked_com = hp.ma(outmap_com)
outmap_masked_com.mask = np.logical_not(maplist.mask)

#newmethod_dipole_Cls_com = hp.anafast(outmap_masked_com,lmax=maplist.lmax) / maplist.fsky
newmethod_nodipole_Cls_com = hp.anafast(hp.remove_dipole(outmap_masked_com), lmax=maplist.lmax) / maplist.fsky

plt.figure()
#plt.semilogy(newmethod_dipole_Cls_com[2:150], label='new method\nwith dipole')
plt.semilogy(newmethod_nodipole_Cls_com[2:150], label='new method\nno dipole')
#plt.semilogy(compare_to_off[2:150] / maplist.fsky, label='previous result', ls='--')
plt.semilogy(np.ones(200)[2:150] * theorynoise_com, c='k')
plt.legend()
plt.xlim(2,50)
plt.ylim(1e-9,2e-8)
plt.savefig(outdir+'comparisons_COMMANDER')

bandpowers = lambda spectrum : np.array([spectrum[2:][1+(5*i):1+(5*(i+1))].mean() for i in np.arange(spectrum.size//5)])
x_ells = bandpowers(np.arange(4003))



plt.figure()
plt.semilogy(x_ells, bandpowers(newmethod_nodipole_Cls_com), label='new method\nno dipole')
plt.semilogy(x_ells, np.ones(x_ells.size) * theorynoise_com, c='k')
plt.legend()
plt.xlim(2,50)
plt.ylim(1e-9,2e-8)
plt.savefig(outdir+'comparisons_bandpowers_COMMANDER')















































mapin_smic = maplist.input_SMICA.copy()

mapin_planckmask = hp.ma(mapin_smic)
mapin_planckmask.mask = np.logical_not(maplist.mask_planck)
ClTT_SMICA = hp.alm2cl(hp.almxfl(hp.map2alm(mapin_planckmask, lmax=maplist.lmax), 1/maplist.SMICAbeam)) / maplist.fsky_planck

mapin_debeam_smic = hp.almxfl(hp.map2alm(mapin_smic, lmax=maplist.lmax), 1/maplist.SMICAbeam)
theorynoise_smic = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=hp.alm2cl(mapin_debeam_smic), clgg_binned=Clgg_unWISE, cltaudg_binned=cltaug_fiducial)

ClTT_filter = ClTT_SMICA.copy()[:maplist.lmax+1]
Clgg_filter = Clgg_unWISE.copy()[:maplist.lmax+1]
Cltaudg = cltaug_fiducial.copy()[:maplist.lmax+1]
ClTT_filter[:100] = 1e15
Clgg_filter[:100] = 1e15
dTlm_xi = hp.almxfl(mapin_debeam_smic, np.divide(np.ones(ClTT_filter.size), ClTT_filter, out=np.zeros_like(np.ones(ClTT_filter.size)), where=ClTT_filter!=0))
dlm_zeta = hp.almxfl(unwiselms, np.divide(Cltaudg, Clgg_filter, out=np.zeros_like(Cltaudg), where=Clgg_filter!=0))
Tmap_filtered = hp.alm2map(dTlm_xi, lmax=maplist.lmax, nside=maplist.nside)
lssmap_filtered = hp.alm2map(dlm_zeta, lmax=maplist.lmax, nside=maplist.nside)
outmap_filtered = Tmap_filtered*lssmap_filtered
outmap_smic = -outmap_filtered * theorynoise_smic

outmap_masked_smic = hp.ma(outmap_smic)
outmap_masked_smic.mask = np.logical_not(maplist.mask)

#newmethod_dipole_Cls_smic = hp.anafast(outmap_masked_smic,lmax=maplist.lmax) / maplist.fsky
newmethod_nodipole_Cls_smic = hp.anafast(hp.remove_dipole(outmap_masked_smic), lmax=maplist.lmax) / maplist.fsky

plt.figure()
#plt.semilogy(newmethod_dipole_Cls_smic[2:150], label='new method\nwith dipole')
plt.semilogy(newmethod_nodipole_Cls_smic[2:150], label='new method\nno dipole')
#plt.semilogy(compare_to_off[2:150] / maplist.fsky, label='previous result', ls='--')
plt.semilogy(np.ones(200)[2:150] * theorynoise_smic, c='k')
plt.legend()
plt.xlim(2,150)
plt.ylim(1e-9,2e-8)
plt.savefig(outdir+'comparisons_SMICA_smicafix')

bandpowers = lambda spectrum : np.array([spectrum[2:][1+(5*i):1+(5*(i+1))].mean() for i in np.arange(spectrum.size//5)])
x_ells = bandpowers(np.arange(4003))



plt.figure()
plt.semilogy(x_ells, bandpowers(newmethod_nodipole_Cls_smic), label='new method\nno dipole')
plt.semilogy(x_ells, np.ones(x_ells.size) * theorynoise_smic, c='k')
plt.legend()
plt.xlim(2,150)
plt.ylim(1e-9,2e-8)
plt.savefig(outdir+'comparisons_bandpowers_SMICA_smicafix')

