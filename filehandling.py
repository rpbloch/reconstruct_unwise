import numpy as np
import healpy as hp
from astropy.io import fits

MASTER_DATA_DIR = 'data/planck_data_testing/'

def read_file(filename, fitshead=None):
	if filename.lower().endswith('.fits'):
		return fits.open(conf.MASTER_DATA_DIR + filename)[1].data[fitshead]
	elif filename.lower().endswith('.npy'):
		return np.load(conf.MASTER_DATA_DIR + filename)
	else:
		raise TypeError(f"File type not recognized: '{filename[filename.rfind('.'):]}'")

def conform(filedata, style):
	if style == 'n2r':
		return hp.reorder(filedata, n2r=True)
	elif style == 'flat':
		return filedata.flatten()
	else:
		raise ValueError(f"Invalid argument for style: '{style}' not recognized")

def load(filename, fitshead):
	return conform(read_file(filename, fitshead))

# K_CMB units are removed from all temperature maps except frequency maps for 100 GHz.
# Arithmetic operations on that map or any copies thereof generate numerical garbage.
# Instead we carry units of K_CMB through and remove them from the reconstruction it


