import sys
from astropy.io import fits as pyfits
import numpy as np

import util


def rmsynth(self):
	"""
	Executes the whole RM-Synthesis with the parameters from the config file for an Apertif observation
	"""
	# Generate the primary beam corrected Faraday cubes
	print('Generating Faraday cubes of primary beam corrected data')
	Qdata, Udata, freq, chanwidth = read_cubes(self)
	check_data(Qdata, Udata, freq)
	phi_array = gen_FDaxis(self.fd_low, self.fd_high, self.fd_dphi)
	p_cube, FD_cube_empty = gen_complex_cubes(Qdata, Udata, phi_array)
	p_cube_blanked = blank_nonvalid(p_cube)
	FD_cube = do_rmsynth(p_cube_blanked, FD_cube_empty, phi_array, freq, chanwidth)
	phi_array, RMTF = util.calc_rmtf(phi_array, freq, chanwidth)
	width, max_scale, max_FD, lam2, lam02 = util.calc_rmsynth_params(freq, chanwidth)
	print('Resolution in Faraday space is ' + str(width) + ' rad/m^2')
	print('Maximum observable scale in Faraday space is ' + str(max_scale) + ' rad/m^2')
	print('Maximum observable Faraday Depth is ' + str(max_FD) + ' rad/m^2')
	write_FDcubes(self, FD_cube, RMTF, phi_array)
	write_FD_RMS_images(self)
	# Generate the non-primary beam corrected Faraday cubes
#	print('Generating Faraday cubes of non-primary beam corrected data')
#	Qdata_uncorr, Udata_uncorr, freq_uncorr, chanwidth_uncorr = read_cubes_uncorr(self)
#	check_data(Qdata_uncorr, Udata_uncorr, freq_uncorr)
#	phi_array_uncorr = gen_FDaxis(self.fd_low, self.fd_high, self.fd_dphi)
#	p_cube_uncorr, FD_cube_empty_uncorr = gen_complex_cubes(Qdata_uncorr, Udata_uncorr, phi_array_uncorr)
#	p_cube_blanked_uncorr = blank_nonvalid(p_cube_uncorr)
#	FD_cube_uncorr = do_rmsynth(p_cube_blanked_uncorr, FD_cube_empty_uncorr, phi_array_uncorr, freq_uncorr, chanwidth_uncorr)
#	phi_array_uncorr, RMTF_uncorr = util.calc_rmtf(phi_array_uncorr, freq_uncorr, chanwidth_uncorr)
#	write_FDcubes_uncorr(self, FD_cube_uncorr, RMTF_uncorr, phi_array_uncorr)


# def calc_med_image(self):
# 	"""
# 	Calculates a median image for pybdsf for better source finding
# 	"""
# 	Qdata, Udata, freq, chanwidth = read_cubes(self)
# 	phi_array_chk = gen_FDaxis(-10240.0, 10240.0, 0.1)
# 	phi_array_chk, RMTF_chk = util.calc_rmtf(phi_array_chk, freq, chanwidth)
# 	RMTF_min, phi_min = util.find_min_resonance(phi_array_chk, RMTF_chk)
# 	print('Minimum value of ' + str(RMTF_min) + ' encountered at ' + str(phi_min) + ' rad/m^2')
# 	phi_array_med = gen_FDaxis(phi_min, phi_min, 0.1)
# 	p_cube, FD_cube_empty = gen_complex_cubes(Qdata, Udata, phi_array_med)
# 	p_cube_blanked = blank_nonvalid(p_cube)
# 	FD_cube_med = do_rmsynth(p_cube_blanked, FD_cube_empty, phi_array_med, freq, chanwidth)
# 	write_medimage(self, FD_cube_med, phi_array_med)
# 	write_medRMTF(self, RMTF_chk, phi_array_chk)


def read_cubes(self):
	"""
	Reads the Stokes Q- and U-cubes and the frequency files into a python array
	returns (array, array, array, float): Q-cube, U-cube, frequency file as a numpy array, channel width as a float
	"""
	Q_cube = pyfits.open(self.polmosaicdir + '/Qcube.fits')
	Q_cube_hdr = Q_cube[0].header
	chanwidth = Q_cube_hdr['CDELT3']
	Q_cube_data = Q_cube[0].data
	U_cube = pyfits.open(self.polmosaicdir + '/Ucube.fits')
	U_cube_data = U_cube[0].data
	freq = np.loadtxt(self.polmosaicdir + '/freq.txt')
	return Q_cube_data, U_cube_data, freq, chanwidth


def read_cubes_uncorr(self):
	"""
	Reads the Stokes Q- and U-cubes and the frequency files of the non-primary beam corrected and the frequency files into a python array
	returns (array, array, array, float): Q-cube, U-cube, frequency file as a numpy array, channel width as a float
	"""
	Q_cube = pyfits.open(self.polmosaicdir + '/Qcube_uncorr.fits')
	Q_cube_hdr = Q_cube[0].header
	chanwidth = Q_cube_hdr['CDELT3']
	Q_cube_data = Q_cube[0].data
	U_cube = pyfits.open(self.polmosaicdir + '/Ucube_uncorr.fits')
	U_cube_data = U_cube[0].data
	freq = np.loadtxt(self.polmosaicdir + '/freq.txt')
	return Q_cube_data, U_cube_data, freq, chanwidth


def check_data(Qcube, Ucube, freq):
	"""
	Checks if the cubes and frequency files have the same length
	Qcube (array): Stokes Q-cube as numpy array
	Ucube (array): Stokes U-cube as numpy array
	freq (array): Frequencies of the planes in the cubes as numpy array
	"""
	length = freq.shape[0]
	if Qcube.shape[0] == Ucube.shape[0] == length:
		print('Image and frequency files successfully read!')
	else:
		print('Q-cube, U-cube and frequency file have different shapes!')
		print('Unsuccessful run of RMsynth! No output generated!')
		sys.exit()


def gen_FDaxis(low, high, dphi):
	"""
	Generates the FD axis
	return (array): The Faraday axis as a numpy array
	"""
	phi_array = np.arange(low, high+dphi, dphi)
	if len(phi_array) % 2 == 0:
		print('Even number of frames in RM-cube! If you are going to clean the spectrum, you may want to reconsider and produce a cube with uneven number of frames due to better sampling of the RMTF!')
	return phi_array


def gen_complex_cubes(Qdata, Udata, phi_array):
	"""
	Combine the Stokes Q and U cubes into a complex valued cube and create an empty final FD-cube
	Qdata (array): Stokes Q-data cube as numpy array
	Udata (array): Stokes U-data cube as numpy array
	phi_array (array): FD axis as an array
	returns(array, array): Complex valued Stokes cube and empty FD-cube as an array
	"""
	p_cube = Qdata + 1j * Udata
	FD_cube = np.zeros((len(phi_array), Qdata.shape[1], Qdata.shape[2]), dtype=np.complex64)
	return p_cube, FD_cube


def blank_nonvalid(p_cube):
	"""
	Blanks any invalid pixels due to different masking at the different input frequencies in the input Stokes Q and U cubes
	p_cube(array): Complex valued Stokes Q + iU cube
	returns(array): Complex valued blanked Stokes Q + iU cube
	"""
	if np.any(np.isnan(p_cube)):
		print('Different blanking of pixels for several images. Those pixels are blanked in the final cube!')
		p_nan = np.ones_like(p_cube[:, 0, 0]) * np.nan
		for m in range(p_cube.shape[2]):
			for n in range(p_cube.shape[1]):
				if np.any(np.isnan(p_cube[:, n, m])):
					if np.all(np.isnan(p_cube[:, n, m])):
						continue
					else:
						p_cube[:, n, m] = p_nan
	return p_cube


def do_rmsynth(p_cube, FD_cube, phi_array, freq, chanwidth):
	"""
	Do the RM-synthesis
	p_cube(array): Complex valued Stokes Q+iU cube
	FD_cube(array): Faraday Depth cube
	phi_array(array): FD axis as an array
	freq(array): Frequencies of the planes in the cubes as numpy array
	chanwidth(float): Channel width in Hertz
	returns(array): FD_cube as an array
	"""
	width, max_scale, max_FD, lam2, lam02 = util.calc_rmsynth_params(freq, chanwidth)
	phis = len(phi_array)
	lam2mlam02 = lam2 - lam02
	length = freq.shape[0]
	for i, phi in enumerate(phi_array):
		print('Processing frame ' + str(i + 1) + '/' + str(phis) + ' with phi = ' + str(phi))
		phases = (np.exp(-2.0j * phi * lam2mlam02))[:, np.newaxis, np.newaxis]
		FD_cube[i, :, :] = np.sum(p_cube * phases, axis=0) / length
	return FD_cube


def write_FDcubes(self, FDcube, RMTF, phi_array):
	"""
	Writes out the results from the RM-Synthesis into the target directory
	FDcube(array): Calculated Faraday cube
	RMTF(array): Calculated RMTF
	"""
	# Get the header from the original data
	qcube = pyfits.open(self.polmosaicdir + '/Qcube.fits')
	qhdu = qcube[0]
	qhdr = qhdu.header

	# Write the Faraday Q-cube
	qhdr['NAXIS3']=len(phi_array)
	qhdr['CRPIX3']=1.0
	qhdr['CRVAL3']=phi_array[0]
	qhdr['CDELT3']=self.fd_dphi
	qhdr['CTYPE3']='Faraday depth'
	qhdr['CUNIT3']='rad m^{2}'
	qhdr['POL'] = 'Q'
	pyfits.writeto(self.polanalysisdir + '/FD_Q.fits', data=np.real(FDcube), header=qhdr, overwrite=True)

	# Write the Faraday U-cube
	qhdr['POL'] = 'U'
	pyfits.writeto(self.polanalysisdir + '/FD_U.fits', data=np.imag(FDcube), header=qhdr, overwrite=True)

	# Write the Faraday P-cube
	qhdr['POL'] = 'P'
	pyfits.writeto(self.polanalysisdir + '/FD_P.fits', data=np.absolute(FDcube), header=qhdr, overwrite=True)

	# Write the RMTF to an ascii-file
	RMTF_text = np.zeros((4, len(phi_array)), dtype=np.float32)
	RMTF_text[0,:] = phi_array
	RMTF_text[1,:] = np.real(RMTF)
	RMTF_text[2,:] = np.imag(RMTF)
	RMTF_text[3,:] = np.absolute(RMTF)
	np.savetxt(self.polanalysisdir + '/RMTF.txt', np.rot90(RMTF_text))


def write_FD_RMS_images(self):
	"""
	Calculates the RMS along the Faraday Q-, U- and PI-cubes
	FDcube(array): Calculated Faraday cube
	"""
	qcube = pyfits.open(self.polanalysisdir + '/FD_Q.fits')
	qhdu = qcube[0]
	qhdr = qhdu.header
	qhdr['NAXIS'] = 2
	del qhdr['NAXIS3']
	del qhdr['CRVAL3']
	del qhdr['CDELT3']
	del qhdr['CRPIX3']
	qdata = qhdu.data
	q_rms = np.std(qdata, axis=0)
	pyfits.writeto(self.polanalysisdir + '/FD_Q_RMS.fits', data=q_rms, header=qhdr, overwrite=True)

	ucube = pyfits.open(self.polanalysisdir + '/FD_U.fits')
	uhdu = ucube[0]
	uhdr = uhdu.header
	uhdr['NAXIS'] = 2
	del uhdr['NAXIS3']
	del uhdr['CRVAL3']
	del uhdr['CDELT3']
	del uhdr['CRPIX3']
	udata = uhdu.data
	u_rms = np.std(udata, axis=0)
	pyfits.writeto(self.polanalysisdir + '/FD_U_RMS.fits', data=u_rms, header=uhdr, overwrite=True)

	pcube = pyfits.open(self.polanalysisdir + '/FD_P.fits')
	phdu = pcube[0]
	phdr = phdu.header
	phdr['NAXIS'] = 2
	del phdr['NAXIS3']
	del phdr['CRVAL3']
	del phdr['CDELT3']
	del phdr['CRPIX3']
	pdata = np.abs(phdu.data)
	p_rms = np.std(pdata, axis=0)
	# calculate the MED
	# c is the constant from MED to std
	c = 0.6745
	d = np.median(pdata, axis = 0)
	p_med = np.median(np.fabs(pdata - d) / c, axis = 0)
	pyfits.writeto(self.polanalysisdir + '/FD_P_RMS.fits', data=p_rms, header=phdr, overwrite=True)
	pyfits.writeto(self.polanalysisdir + '/FD_P_MED.fits', data=p_med, header=phdr, overwrite=True)


def write_FDcubes_uncorr(self, FDcube, RMTF, phi_array):
	"""
	Writes out the results from the RM-Synthesis into the target directory
	FDcube(array): Calculated Faraday cube
	RMTF(array): Calculated RMTF
	"""
	# Get the header from the original data
	qcube = pyfits.open(self.polmosaicdir + '/Qcube_uncorr.fits')
	qhdu = qcube[0]
	qhdr = qhdu.header

	# Write the Faraday Q-cube
	qhdr['NAXIS3']=len(phi_array)
	qhdr['CRPIX3']=1.0
	qhdr['CRVAL3']=phi_array[0]
	qhdr['CDELT3']=self.fd_dphi
	qhdr['CTYPE3']='Faraday depth'
	qhdr['CUNIT3']='rad m^{2}'
	qhdr['POL'] = 'Q'
	pyfits.writeto(self.polanalysisdir + '/FD_Q_uncorr.fits', data=np.real(FDcube), header=qhdr, overwrite=True)

	# Write the Faraday U-cube
	qhdr['POL'] = 'U'
	pyfits.writeto(self.polanalysisdir + '/FD_U_uncorr.fits', data=np.imag(FDcube), header=qhdr, overwrite=True)

	# Write the Faraday P-cube
	qhdr['POL'] = 'P'
	pyfits.writeto(self.polanalysisdir + '/FD_P_uncorr.fits', data=np.absolute(FDcube), header=qhdr, overwrite=True)

	# Write the RMTF to an ascii-file
	RMTF_text = np.zeros((4, len(phi_array)), dtype=np.float32)
	RMTF_text[0,:] = phi_array
	RMTF_text[1,:] = np.real(RMTF)
	RMTF_text[2,:] = np.imag(RMTF)
	RMTF_text[3,:] = np.absolute(RMTF)
	np.savetxt(self.polanalysisdir + '/RMTF_uncorr.txt', np.rot90(RMTF_text))


def write_medimage(self, FDcube_med, phi_array_med):
	"""
	Writes out the background noise image atr the lowest resonance of the RMTF
	FDcube(array): Calculated Faraday cube
	phi_array_med(array): Calculated RMTF
	"""
	# Get the header from the original data
	qcube = pyfits.open(self.polmosaicdir + '/Qcube.fits')
	qhdu = qcube[0]
	qhdr = qhdu.header

	qhdr['NAXIS'] = 2
	del qhdr['NAXIS3']
	del qhdr['CDELT3']
	del qhdr['CRPIX3']
	del qhdr['CRVAL3']
	del qhdr['CTYPE3']
	qhdr['BUNIT'] = 'Jy beam-1'
	print(FDcube_med.shape)
	pyfits.writeto(self.polanalysisdir + '/med_P.fits', data=np.squeeze(np.absolute(FDcube_med)), header=qhdr, overwrite=True)


def write_medRMTF(self, RMTF_med, phi_array_med):
	"""
	Writes out the background noise image atr the lowest resonance of the RMTF
	FDcube(array): Calculated Faraday cube
	phi_array_med(array): Calculated RMTF
	"""
	RMTF_text = np.zeros((4, len(phi_array_med)), dtype=np.float32)
	RMTF_text[0,:] = phi_array_med
	RMTF_text[1,:] = np.real(RMTF_med)
	RMTF_text[2,:] = np.imag(RMTF_med)
	RMTF_text[3,:] = np.absolute(RMTF_med)
	np.savetxt(self.polanalysisdir + '/RMTF_med.txt', np.rot90(RMTF_text))