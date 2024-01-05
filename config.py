



MASTER_DATA_DIR = 'data/planck_data_testing/'

from collections import namedtuple

fileinfo = namedtuple('MapFileInfo', ['name', 'loc', 'fitshead', 'conform_type'])

map_files = [ \
             fileinfo(name='SMICA',     loc=MASTER_DATA_DIR+'maps/COM_CMB_IQU-smica_2048_R3.00_full.fits', fitshead='I_STOKES', conform_type='n2r'),
             fileinfo(name='COMMANDER', loc=MASTER_DATA_DIR+'maps/COM_CMB_IQU-commander_2048_R3.00_full.fits', fitshead='I_STOKES', conform_type='n2r')
]




input_unWISE = conform(read_file('data/unWISE/numcounts_map1_2048-r1-v2_flag.fits', 'T'), 'flat')
input_T100 = conform(read_file('maps/HFI_SkyMap_100_2048_R3.01_full.fits', 'I_STOKES'), 'n2r')
input_T143 = conform(read_file('maps/HFI_SkyMap_143_2048_R3.01_full.fits', 'I_STOKES'), 'n2r') / 2.7255
input_T217 = conform(read_file('maps/HFI_SkyMap_217_2048_R3.01_full.fits', 'I_STOKES'), 'n2r') / 2.7255
input_T353 = conform(read_file('maps/HFI_SkyMap_353-psb_2048_R3.01_full.fits', 'I_STOKES'), 'n2r') / 2.7255
input_T100_noCMB_SMICA = conform(read_file('maps/HFI_CompMap_Foregrounds-smica-100_R3.00.fits', 'INTENSITY'), 'flat')
input_T143_noCMB_SMICA = conform(read_file('maps/HFI_CompMap_Foregrounds-smica-143_R3.00.fits', 'INTENSITY'), 'flat') / 2.7255
input_T217_noCMB_SMICA = conform(read_file('maps/HFI_CompMap_Foregrounds-smica-217_R3.00.fits', 'INTENSITY'), 'flat') / 2.7255
input_T353_noCMB_SMICA = conform(read_file('maps/HFI_CompMap_Foregrounds-smica-353_R3.00.fits', 'INTENSITY'), 'flat') / 2.7255
input_T100_noCMB_COMMANDER = conform(read_file('maps/HFI_CompMap_Foregrounds-commander-100_R3.00.fits', 'INTENSITY'), 'flat')
input_T143_noCMB_COMMANDER = conform(read_file('maps/HFI_CompMap_Foregrounds-commander-143_R3.00.fits', 'INTENSITY'), 'flat') / 2.7255
input_T217_noCMB_COMMANDER = conform(read_file('maps/HFI_CompMap_Foregrounds-commander-217_R3.00.fits', 'INTENSITY'), 'flat') / 2.7255
input_T353_noCMB_COMMANDER = conform(read_file('maps/HFI_CompMap_Foregrounds-commander-353_R3.00.fits', 'INTENSITY'), 'flat') / 2.7255
input_T100_thermaldust = conform(read_file('foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-100_2048_R3.00_full.fits', 'TEMPERATURE'), 'flat')
input_T143_thermaldust = conform(read_file('foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-143_2048_R3.00_full.fits', 'TEMPERATURE'), 'flat')  / 2.7255
input_T217_thermaldust = conform(read_file('foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-217_2048_R3.00_full.fits', 'TEMPERATURE'), 'flat')  / 2.7255
input_T353_thermaldust = conform(read_file('foregrounds/COM_SimMap_thermaldust-ffp10-skyinbands-353_2048_R3.00_full.fits', 'TEMPERATURE'), 'flat')  / 2.7255
input_T353_CIB = np.nan_to_num(read_file('CIB/cib_fullmission_353.hpx.fits', 'CIB'))
mask_planck = hp.reorder(read_file('COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', 'TMASK'),n2r=True)
mask_unwise = read_file('data/mask_unWISE_thres_v10.npy')
mask_GAL020 = conform(read_file('data/masks/planck/HFI_Mask_GalPlane-apo0_2048_R2.00.fits', 'GAL020'), 'n2r')
mask_CIB = read_file('CIB/cib_fullmission_353.hpx.fits', 'CIB')
