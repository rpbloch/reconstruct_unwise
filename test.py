




print('Processing CMB reconstructions...')
map_container = {'100GHz' : maplist.input_T100,
				 '143GHz' : maplist.input_T143,
				 '217GHz' : maplist.input_T217,
				 '353GHz' : maplist.input_T353,
				 '100GHz_noSMICA' : maplist.input_T100_noCMB_SMICA,
				 '143GHz_noSMICA' : maplist.input_T143_noCMB_SMICA,
				 '217GHz_noSMICA' : maplist.input_T217_noCMB_SMICA,
				 '353GHz_noSMICA' : maplist.input_T353_noCMB_SMICA,
				 '100GHz_noCOMMANDER' : maplist.input_T100_noCMB_COMMANDER,
				 '143GHz_noCOMMANDER' : maplist.input_T143_noCMB_COMMANDER,
				 '217GHz_noCOMMANDER' : maplist.input_T217_noCMB_COMMANDER,
				 '353GHz_noCOMMANDER' : maplist.input_T353_noCMB_COMMANDER,
				 '100GHz_thermaldust' : maplist.input_T100_thermaldust,
				 '143GHz_thermaldust' : maplist.input_T143_thermaldust,
				 '217GHz_thermaldust' : maplist.input_T217_thermaldust,
				 '353GHz_thermaldust' : maplist.input_T353_thermaldust,
				 '353GHz_CIB' : maplist.input_T353_CIB}

beam_container = {'100GHz' : maplist.T100beam,
				  '143GHz' : maplist.T143beam,
				  '217GHz' : maplist.T217beam,
				  '353GHz' : maplist.T353beam,
				  '100GHz_noSMICA' : maplist.T100beam,
				  '143GHz_noSMICA' : maplist.T143beam,
				  '217GHz_noSMICA' : maplist.T217beam,
				  '353GHz_noSMICA' : maplist.T353beam,
				  '100GHz_noCOMMANDER' : maplist.T100beam,
				  '143GHz_noCOMMANDER' : maplist.T143beam,
				  '217GHz_noCOMMANDER' : maplist.T217beam,
				  '353GHz_noCOMMANDER' : maplist.T353beam,
				  '100GHz_thermaldust' : maplist.T100beam,
				  '143GHz_thermaldust' : maplist.T143beam,
				  '217GHz_thermaldust' : maplist.T217beam,
				  '353GHz_thermaldust' : maplist.T353beam,
				  '353GHz_CIB' : maplist.SMICAbeam}

reconstructions = {}
noises = {}
recon_Cls = {}




maplist.processed_alms['unWISE'] = hp.map2alm(maplist.input_unWISE, lmax=4000)


maplist.Cls['unWISE'] = hp.alm2cl(maplist.processed_alms['unWISE'])



for key in map_container:
	print('    Preprocessing %s' % key)
	maplist.processed_alms[key] = maplist.mask_and_debeam(map_container[key], maplist.mask, beam_container[key])
	maplist.processed_alms[key+'_CIBmask'] = maplist.mask_and_debeam(map_container[key], maplist.mask_huge, beam_container[key])
	maplist.Cls[key] = maplist.alm2cl(maplist.processed_alms[key], maplist.fsky)
	maplist.Cls[key+'_CIBmask'] = maplist.alm2cl(maplist.processed_alms[key+'_CIBmask'], maplist.fsky_huge)





for key in map_container:
	print('    Reconstructing %s' % key)
	master_cltt = maplist.Cls[key.split('_')[0]]  # Theory ClTT should be on the full frequency sky
	noises[key] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=master_cltt, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial)
	noises[key+'_mm'] = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=master_cltt, clgg_binned=maplist.Cls['unWISE'], cltaudg_binned=cltaug_fiducial_mm)
	if '353GHz' not in key:
		convert_K_flag = True if '100GHz' in key else False
		reconstructions[key] = combine_alm(maplist.processed_alms[key], maplist.processed_alms['unWISE'], maplist.mask, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial, noises[key], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=convert_K_flag)
		reconstructions[key+'_mm'] = combine_alm(maplist.processed_alms[key], maplist.processed_alms['unWISE'], maplist.mask, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial_mm, noises[key+'_mm'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=convert_K_flag)
		recon_Cls[key] = maplist.alm2cl(hp.map2alm(reconstructions[key], lmax=recon_lmax), maplist.fsky)
		recon_Cls[key+'_mm'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_mm'], lmax=recon_lmax), maplist.fsky)
	if ('217GHz' in key) or ('353GHz' in key):
		reconstructions[key+'_CIBmask'] = combine_alm(maplist.processed_alms[key+'_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial, noises[key], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
		reconstructions[key+'_mm_CIBmask'] = combine_alm(maplist.processed_alms[key+'_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], cltaug_fiducial_mm, noises[key+'_mm'], lmax=maplist.lmax, nside_out=maplist.nside, convert_K=False)
		recon_Cls[key+'_CIBmask'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_CIBmask'], lmax=recon_lmax), maplist.fsky_huge)
		recon_Cls[key+'_mm_CIBmask'] = maplist.alm2cl(hp.map2alm(reconstructions[key+'_mm_CIBmask'], lmax=recon_lmax), maplist.fsky_huge)




recon_lmax = 200
master_cltt = maplist.Cls['353GHz'].copy()
recon_noise = noises['353GHz']
dchi = np.trapz(np.trapz(window_v['353GHz'],chis,axis=1),chis)


recon_353ghz_alm = combine_alm(maplist.processed_alms['353GHz_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], np.load('cltaug_fiducial.npy')*dchi, recon_noise, lmax=4000, nside_out=2048, convert_K=False)
recon_353ghz_Cl = maplist.alm2cl(hp.map2alm(recon_353ghz_alm, lmax=recon_lmax), maplist.fsky_huge)

recon_353ghz_thermaldust_alm = combine_alm(maplist.processed_alms['353GHz_thermaldust_CIBmask'], maplist.processed_alms['unWISE'], maplist.mask_huge, master_cltt, maplist.Cls['unWISE'], np.load('cltaug_fiducial.npy')*dchi, recon_noise, lmax=4000, nside_out=2048, convert_K=False)
recon_353ghz_thermaldust_Cl = maplist.alm2cl(hp.map2alm(recon_353ghz_thermaldust_alm, lmax=recon_lmax), maplist.fsky_huge)

gal_fsky60 = hp.reorder(fits.open('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits')[1].data['GAL060'],n2r=True)
cltt_mask = gal_fsky60 * maplist.mask_planck
fsky_debug = np.sum(cltt_mask) / cltt_mask.size

cltt_353 = hp.alm2cl(hp.almxfl(hp.map2alm(maplist.input_T353 * cltt_mask), 1/maplist.T353beam)) / fsky_debug
clgg_debug = hp.anafast(maplist.input_unWISE * maplist.mask_unwise) / (np.sum(maplist.mask_unwise)/maplist.mask_unwise.size)
cltaug_debug = np.load('cltaug_fiducial.npy')*dchi
noise_debug = Noise_vr_diag(lmax=ls.max(), alpha=0, gamma=0, ell=2, cltt=cltt_353, clgg_binned=clgg_debug, cltaudg_binned=cltaug_debug)

recon_353debug_alm = combine_alm(hp.map2alm(maplist.input_T353,lmax=4000), hp.map2alm(maplist.input_unWISE,lmax=4000), maplist.mask, cltt_353, clgg_debug, cltaug_debug, noise_debug, lmax=4000, nside_out=2048, convert_K=False)
recon_353debug_Cl = hp.anafast(recon_353debug_alm, lmax=recon_lmax) / maplist.fsky

recon_353debug_thermaldust_alm = combine_alm(hp.map2alm(maplist.input_T353_thermaldust,lmax=4000), hp.map2alm(maplist.input_unWISE,lmax=4000), maplist.mask, cltt_353, clgg_debug, cltaug_debug, noise_debug, lmax=4000, nside_out=2048, convert_K=False)
recon_353debug_thermaldust_Cl = hp.anafast(recon_353debug_thermaldust_alm, lmax=recon_lmax) / maplist.fsky

plt.figure()
plt.loglog(np.arange(200)[2:100], recon_353ghz_Cl[2:100], label='353Ghz')
plt.loglog(np.arange(200)[2:100], recon_353ghz_thermaldust_Cl[2:100], label='thermal dust')
plt.loglog(np.arange(200)[2:100], recon_353debug_Cl[2:100], label='353Ghz debug')
plt.loglog(np.arange(200)[2:100], recon_353debug_thermaldust_Cl[2:100], label='thermal dust debug')
plt.legend()
plt.savefig(outdir+'debug')
