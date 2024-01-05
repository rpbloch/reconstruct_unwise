from astropy import units as u

def convert_units(Map, T_CMB, CIB_frequency=None):
	# Converts input map to unitless. If CIB_frequency is provided, performs MJy/sr to T_CMB conversion first.
	if CIB_frequency:
		Map *= (u.MJy / u.sr).to(1. * u.K, equivalencies=u.thermodynamic_temperature(CIB_frequency * u.GHz, T_cmb=T_CMB*u.K))
	Map /= T_CMB