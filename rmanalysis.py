from astropy.io import fits as pyfits
import numpy as np
import glob
import shutil

import util
import misc

from radio_beam import Beam


def rmanalysis(self):
    """
    Runs the analysis of the Faraday cubes and generates PI, RM, PA, and PA0 images as well as their error images
    """
    FD_P_data, FD_Q_data, FD_U_data = load_cubes(self)
    PI_arr, RM_arr, PA_arr, PA0_arr = calc_maps(self, FD_P_data, FD_Q_data, FD_U_data)
#    rmsynth.calc_med_image(self)
#    PI_bias_corr = sub_bias(self, PI_arr)
    write_maps(self, PI_arr, RM_arr, PA_arr, PA0_arr)
    PI_err, RM_err, PA_err = calc_error_maps(self, PI_arr)
    write_error_maps(self, PI_err, RM_err, PA_err)
    calc_FP(self)
#    FD_P_data_uncorr, FD_Q_data_uncorr, FD_U_data_uncorr = load_cubes_uncorr(self)
#    PI_arr_uncorr, RM_arr_uncorr, PA_arr_uncorr, PA0_arr_uncorr = calc_maps(self, FD_P_data_uncorr, FD_Q_data_uncorr, FD_U_data_uncorr)
#    write_maps_uncorr(self, PI_arr_uncorr, RM_arr_uncorr, PA_arr_uncorr, PA0_arr_uncorr)


def load_cubes(self):
    """
    Loads the Faraday P, Q, and U cubes and returns their data as numpy array
    returns(np.array, np.array, np.array): PI, Q and U numpy data arrays
    """
    FD_P = pyfits.open(self.polanalysisdir + '/FD_P.fits')
    FD_P_data = FD_P[0].data
    FD_Q = pyfits.open(self.polanalysisdir + '/FD_Q.fits')
    FD_Q_data = FD_Q[0].data
    FD_U = pyfits.open(self.polanalysisdir + '/FD_U.fits')
    FD_U_data = FD_U[0].data
    return FD_P_data, FD_Q_data, FD_U_data


def load_cubes_uncorr(self):
    """
    Loads the Faraday P, Q, and U cubes and returns their data as numpy array
    returns(np.array, np.array, np.array): PI, Q and U numpy data arrays
    """
    FD_P = pyfits.open(self.polanalysisdir + '/FD_P_uncorr.fits')
    FD_P_data = FD_P[0].data
    FD_Q = pyfits.open(self.polanalysisdir + '/FD_Q_uncorr.fits')
    FD_Q_data = FD_Q[0].data
    FD_U = pyfits.open(self.polanalysisdir + '/FD_U_uncorr.fits')
    FD_U_data = FD_U[0].data
    return FD_P_data, FD_Q_data, FD_U_data


def calc_maps(self, FD_P_data, FD_Q_data, FD_U_data):
    """
    Calculates the polarised intensity, Rotation Measure, Polarisation angle and derotated Polarisation angle arrays
    FD_P_data (np.array): Faraday P-cube
    FD_Q_data (np.array): Faraday Q-cube
    FD_U_data (np.array): Faraday U-cube
    returns(np.array, np.array, np.array, np.array): PI , RM, PA, and PA0 2D numpy arrays
    """
    xlen = FD_P_data.shape[-1]
    ylen = FD_P_data.shape[-2]
    zlen = FD_P_data.shape[-3]

    PI_arr = np.nan * np.zeros((ylen, xlen), dtype=np.float32)
    RM_arr = np.nan * np.zeros((ylen, xlen), dtype=np.float32)
    RM_index = np.nan * np.zeros((ylen, xlen), dtype=np.float32)
    Q_arr = np.nan * np.zeros((ylen, xlen), dtype=np.float32)
    U_arr = np.nan * np.zeros((ylen, xlen), dtype=np.float32)

    for x in range(xlen):
        for y in range(ylen):
            FD_values = FD_P_data[:,y,x]
            if np.any(np.isnan(FD_values)):
                pass
            else:
                # Get the index of the maximum along the Faraday axis
                ind = np.argmax(FD_values)
                # If the index is at the edge, shift it by one towards the centre. Otherwise parabolic fitting would not work.
                if ind<=0:
                    ind = 1
                elif ind >= (zlen-1):
                    ind = zlen-2
                try:
                    RM_ind, PI = misc.parabolic(FD_values, ind)
                except FloatingPointError:
                    continue

                if RM_ind<=0:
                    RM_ind = 1
                    PI = FD_values[RM_ind]
                elif RM_ind >= (zlen-1):
                    RM_ind = zlen-2
                    PI = FD_values[RM_ind]
                PI_arr[y, x] = PI
                RM_arr[y, x] = self.fd_dphi * RM_ind + self.fd_low
                RM_index[y, x] = RM_ind
                try:
                    Q_med, Q_high = FD_Q_data[int(RM_index[y, x]), y, x], FD_Q_data[int(RM_index[y, x]) + 1, y, x]
                    Q_arr[y, x] = Q_med + ((Q_high - Q_med) * (RM_index[y, x] - float(int(RM_index[y, x]))))
                    U_med, U_high = FD_U_data[int(RM_index[y, x]), y, x], FD_U_data[int(RM_index[y, x]) + 1, y, x]
                    U_arr[y, x] = U_med + ((U_high - U_med) * (RM_index[y, x] - float(int(RM_index[y, x]))))
                except (ValueError, IndexError):
                    Q_arr[y,x] = np.nan
                    U_arr[y,x] = np.nan
    RM_arr = -1.0 * RM_arr
    PA_arr = np.degrees(0.5 * np.arctan2(U_arr, Q_arr))
    freqs, chanwidth = util.get_freqs_chanwidth(self)
    width, max_scale, max_FD, lam2, lam02 = util.calc_rmsynth_params(freqs, chanwidth)
    PA0_arr = PA_arr - np.degrees(RM_arr * lam02)
    return PI_arr, RM_arr, PA_arr, PA0_arr


def sub_bias(self, PI_arr):
    """
    Subtracts the bias from the PI map using the FD_P map generated at the lowest responses of the RMTF
    PI_arr(np.array): Not bias corrected PI array as np array
    returns(np.array): Bias corrected PI array as np array
    """
    print('To be done...')


def write_maps(self, PI_arr, RM_arr, PA_arr, PA0_arr):
    """
    Writes out the 2D arrays for PI, RM, PA and PA0 as FITS-files
    PI_arr(np.array): PI array as np-array
    RM_arr(np.array): RM array as np-array
    PA_arr(np.array): PA array as np-array
    PA0_arr(np.array): PA0 array as np-array
    """
    # Load and modify the header
    header = misc.load_header(self.polanalysisdir + '/FD_P.fits')
    header['NAXIS'] = 2
    del header['NAXIS3']
    del header['CDELT3']
    del header['CRPIX3']
    del header['CRVAL3']
    del header['CTYPE3']
    del header['CUNIT3']
    header['BUNIT'] = 'rad m-2'

    # Write the RM map
    pyfits.writeto(self.polanalysisdir + '/RM.fits', data=RM_arr, header=header, overwrite=True)

    # Write the PI map
    header['BUNIT'] = 'Jy beam-1'
    pyfits.writeto(self.polanalysisdir + '/PI.fits', data=PI_arr, header=header, overwrite=True)

    # Write the PA map
    header['BUNIT'] = 'deg'
    pyfits.writeto(self.polanalysisdir + '/PA.fits', data=PA_arr, header=header, overwrite=True)

    # Write the PA0 map
    pyfits.writeto(self.polanalysisdir + '/PA0.fits', data=PA0_arr, header=header, overwrite=True)


def write_maps_uncorr(self, PI_arr, RM_arr, PA_arr, PA0_arr):
    """
    Writes out the 2D arrays for PI, RM, PA and PA0 as FITS-files
    PI_arr(np.array): PI array as np-array
    RM_arr(np.array): RM array as np-array
    PA_arr(np.array): PA array as np-array
    PA0_arr(np.array): PA0 array as np-array
    """
    # Load and modify the header
    header = misc.load_header(self.polanalysisdir + '/FD_P_uncorr.fits')
    header['NAXIS'] = 2
    del header['NAXIS3']
    del header['CDELT3']
    del header['CRPIX3']
    del header['CRVAL3']
    del header['CTYPE3']
    del header['CUNIT3']

    # Write the PI map
    header['BUNIT'] = 'Jy beam-1'
    pyfits.writeto(self.polanalysisdir + '/PI_uncorr.fits', data=PI_arr, header=header, overwrite=True)


def calc_error_maps(self, PI_arr):
    """
    Calculates the PI, RM and PA error maps
    freqs (np.array): Frequencies used in the creation of the RM-Synthesis
    PI_arr (np.array):PI array as np-array
    RM_arr(np.array): RM array as np-array
    PA_arr(np.array): PA array as np-array
    returns (np.array, np.array, np.array): PI-error, RM-error, and PA-error arrays
    """
    freqs, chanwidth = util.get_freqs_chanwidth(self)
    # Number of frequencies used
    nf = len(freqs)
    # Calculate sigma_lambda^2
    sumpow4 = np.sum(np.power(299792458.0/freqs,4.0))
    sumpow2 = np.sum(np.power(299792458.0/freqs,2.0))
    sigma_l_2 = (1.0 / (nf - 1.0)) * (sumpow4 - (1.0 / nf) * np.power(sumpow2, 2))
    # Get the median rms of the input Q and U images
    Q_cube = pyfits.open(self.polmosaicdir + '/Qcube.fits')
    Q_data = Q_cube[0].data
    U_cube = pyfits.open(self.polmosaicdir + '/Ucube.fits')
    U_data = U_cube[0].data
    all_data = np.concatenate((Q_data, U_data))
    rms = np.nanstd(all_data)
    # Calculate the PI error array
    sigma_PI_arr = rms * np.sqrt(nf) * np.ones_like(PI_arr)
    # Calculate the RM error array
    sigma_RM_arr = np.sqrt(np.power(rms, 2.0) / (4.0 * (nf - 2.0) * np.power(PI_arr, 2.0) * sigma_l_2))
    # Calculate the PA error array
    sigma_PA_arr = (np.sqrt(np.power(rms, 2.0) / (4.0 * (nf - 2.0) * np.power(PI_arr, 2.0)) * (((nf - 1) / nf) + (sumpow4 / sigma_l_2)))) * (180.0 / np.pi)
    return sigma_PI_arr, sigma_RM_arr, sigma_PA_arr


def write_error_maps(self, PI_err, RM_err, PA_err):
    # Load and modify the header
    header = misc.load_header(self.polanalysisdir + '/RM.fits')
    header['BUNIT'] = 'rad m-2'

    # Write the RM error map
    pyfits.writeto(self.polanalysisdir + '/RM_err.fits', data=RM_err, header=header, overwrite=True)

    # Write the PI error map
    header['BUNIT'] = 'Jy beam-1'
    pyfits.writeto(self.polanalysisdir + '/PI_err.fits', data=PI_err, header=header, overwrite=True)

    # Write the PA error map
    header['BUNIT'] = 'deg'
    pyfits.writeto(self.polanalysisdir + '/PA_err.fits', data=PA_err, header=header, overwrite=True)


def calc_FP(self):
    # Calculate the FP map and its error map

    # Copy the total power map to the polarisation directory
    # get the source id
    tpimage = glob.glob(self.contmosaicdir + '/*.fits')[0].split('/')[-1]
    # copy the TP image and reproject to the PI image
    shutil.copy(self.contmosaicdir + '/' + tpimage, self.polanalysisdir + '/TP.fits')
    util.reproject_image(self.polanalysisdir + '/TP.fits', self.polanalysisdir + '/PI.fits', self.polanalysisdir + '/TP_repr.fits')
    # Convolve the total power image to the same beam as the polarised intensity image
    with pyfits.open(self.polanalysisdir + '/PI.fits') as hdul:
        hdr = hdul[0].header
        tppsf = Beam.from_fits_header(hdr)
    util.fits_reconvolve_image(self.polanalysisdir + '/TP_repr.fits', tppsf, out=self.polanalysisdir + '/TP_repr_cv.fits')
    # Get the data of the two arrays and calculate the FP-image
    PI_arr = misc.load_data(self.polanalysisdir + '/PI.fits')
    TP_arr = misc.load_data(self.polanalysisdir + '/TP_repr_cv.fits')
    FP_arr = PI_arr/TP_arr
    # Write the FP image
    header = misc.load_header(self.polanalysisdir + '/PI.fits')
    header['BUNIT'] = 'FP'
    pyfits.writeto(self.polanalysisdir + '/FP.fits', data=FP_arr, header=header, overwrite=True)