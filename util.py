import os
import numpy as np

from configparser import ConfigParser

from astropy.table import Table
from astropy.io import fits as pyfits
from reproject import reproject_interp
from radio_beam import Beam
from radio_beam import EllipticalGaussian2DKernel
from scipy.fft import ifft2, ifftshift
from astropy.convolution import convolve
import astropy.units as u

import misc


def load_config(config_object, file_=None):
    """
    Function to load the config file
    """
    config = ConfigParser()  # Initialise the config parser
    config.read_file(open(file_))
    for s in config.sections():
        for o in config.items(s):
            setattr(config_object, o[0], eval(o[1]))
    return config  # Save the loaded config file as defaults for later usage


def set_dirs(self):
    """
    Creates the directory names for the subdirectories to make scripting easier
    """
    self.polworkdir = os.path.join(self.basedir, self.obsid, self.mossubdir, self.mospoldir)
    self.contworkdir = os.path.join(self.basedir, self.obsid, self.mossubdir, self.moscontdir)
    self.contmosaicdir = os.path.join(self.contworkdir, 'mosaic')
    self.polmosaicdir = os.path.join(self.polworkdir, 'mosaic')
    self.polanalysisdir = os.path.join(self.polworkdir, 'analysis')
    self.polanalysisplotdir = os.path.join(self.polanalysisdir, 'plots')


def gen_dirs(self):
    """
    Creates the necessary directories for the analysis
    """
    if os.path.isdir(self.polanalysisdir):
        pass
    else:
        os.makedirs(self.polanalysisdir)
    if os.path.isdir(self.polanalysisplotdir):
        pass
    else:
        os.makedirs(self.polanalysisplotdir)


def get_freqs_chanwidth(self):
    """
    Get the frequencies and channel width of the input cubes
    return:
    """
    header = misc.load_header(self.polmosaicdir + '/Qcube.fits')
    chanwidth = header['CDELT3']
    freq = np.loadtxt(self.polmosaicdir + '/freq.txt')
    return freq, chanwidth


def calc_rmsynth_params(freq, chanwidth):
    """
    Calculates the parameters for the RM-Synthesis given a frequency coverage
    freq(array): numpy array read from the frequency file
    chanwidth(float): Channel width in Hertz
    returns(float, float, float, float): resolution, max observable scale, max observable Faraday Depth, lambda_0**2 in rad/m^2
    """
    c = 299792458.0  # Speed of light
    lam2 = (c / freq) ** 2.0
    lam02 = np.mean(lam2)
    minl2 = np.min(lam2)
    maxl2 = np.max(lam2)
    chwidth2 = (c / (np.mean(freq) - 0.5 * chanwidth)) ** 2.0 - (c / (np.mean(freq) + 0.5 * chanwidth)) ** 2.0
    width = (2.0 * np.sqrt(3.0)) / (maxl2 - minl2)
    max_scale = np.pi / minl2
    max_FD = np.sqrt(3.0) / chwidth2
    return width, max_scale, max_FD, lam2, lam02


def calc_rmtf(phi_array, freq, chanwidth):
    """
    Calculates the RMTF for a given input observation
    """
    length = freq.shape[0]
    width, max_scale, max_FD, lam2, lam02 = calc_rmsynth_params(freq, chanwidth)
    RMTF_comp = np.zeros((len(phi_array), length), dtype=np.complex64)
    for phi in range(len(phi_array)):
        for i in range(0, length):
            RMTF_comp[phi, i] = np.exp(-2.0j * phi_array[phi] * (lam2[i] - lam02))
    RMTF = np.sum((RMTF_comp), axis=1) / length
    return phi_array, RMTF


def find_min_resonance(phi_array, RMTF):
    """
    Finds the minimum absolute value along a given RMTF and determines the FD at this position
    phi_array(array): The sampled FD values
    RMTF(array): real, imaginary and absolute values of the RMTF
    returns(float,float): Location of the minimum in rad/m**2 and value of the minimum
    """
    RMTF_abs = np.absolute(RMTF)
    min, argmin = np.min(RMTF_abs), np.argmin(RMTF_abs)
    FD_min = phi_array[argmin]
    return min, FD_min


def load_catalogue(catalogue):
    """
    Loads a catalogue file using astropy.table
    catalogue: Catalogue textfile created by pybdsf
    returns: catalogue as astropy table array
    """
    f = open(catalogue)
    lines = f.readlines()
    header_line = lines[5].rstrip('\n').lstrip('# ')
    col_names = header_line.split(' ')
    t = Table.read(catalogue, format='ascii', names=col_names)
    return t


def load_pointing_centres(pfile, nan=False):
    """
    Loads the pointing centre file and removes all nan values
    pfile: Pointing centre file from the mosaicking
    returns: Pointing centres with RA and DEC as two numpy arrays
    """
    parray = np.loadtxt(pfile)
    ra = parray[:,1]
    dec = parray[:,2]
    if nan:
        pass
    else:
        ra = ra[~np.isnan(ra)]
        dec = dec[~np.isnan(dec)]
    return ra, dec


def get_imageextent(self):
    with pyfits.open(self.polanalysisdir + '/PI.fits') as hdu:
        hdu_header = hdu[0].header
        ra_cent = hdu_header['CRVAL1']
        dec_cent = hdu_header['CRVAL2']
        ra_size = np.absolute(hdu_header['NAXIS1'] * hdu_header['CDELT1'])
        dec_size = np.absolute(hdu_header['NAXIS2'] * hdu_header['CDELT2'])
        return ra_cent, dec_cent, ra_size, dec_size


def get_beam(self):
    with pyfits.open(self.polanalysisdir + '/PI.fits') as hdu:
        hdu_header = hdu[0].header
        bmaj = hdu_header['BMAJ']
        bmin = hdu_header['BMIN']
    return bmaj, bmin


def reproject_image(infile, template, outfile):
    with pyfits.open(infile) as infile_hdu:
        with pyfits.open(template) as template_hdu:
            array, footprint = reproject_interp(infile_hdu[0], template_hdu[0].header)
            pyfits.writeto(outfile, array, header=template_hdu[0].header, overwrite=True)


def remove_dims(infile, outfile):
    with pyfits.open(infile) as infile_hdu:
        infile_data = infile_hdu[0].data
        infile_hdr = infile_hdu[0].header
        outfile_data = np.squeeze(infile_data)
        infile_hdr['NAXIS'] = 2
        for keyword in ['CRVAL3','CDELT3','CRPIX3','CUNIT3','CTYPE3','NAXIS3','CRVAL4','CDELT4','CRPIX4','CUNIT4','CTYPE4','NAXIS4']:
            try:
                del infile_hdr[keyword]
            except:
                pass
        pyfits.writeto(outfile, outfile_data, header=infile_hdr, overwrite=True)


def fft_psf(bmaj, bmin, bpa, size=3073):
    SIGMA_TO_FWHM = np.sqrt(8*np.log(2))
    fmaj = size / (bmin / SIGMA_TO_FWHM) / 2 / np.pi
    fmin = size / (bmaj / SIGMA_TO_FWHM) / 2 / np.pi
    fpa = bpa + 90
    angle = np.deg2rad(90+fpa)
    fkern = EllipticalGaussian2DKernel(fmaj, fmin, angle, x_size=size, y_size=size)
    fkern.normalize('peak')
    fkern = fkern.array
    return fkern


def reconvolve_gaussian_kernel(img, old_maj, old_min, old_pa, new_maj, new_min, new_pa):
    """
    convolve image with a gaussian kernel without FFTing it
    bmaj, bmin -- in pixels,
    bpa -- in degrees from top clockwise (like in Beam)
    inverse -- use True to deconvolve.
    NOTE: yet works for square image without NaNs
    """
    size = len(img)
    imean = img.mean()
    img -= imean
    fimg = np.fft.fft2(img)
    krel = fft_psf(new_maj, new_min, new_pa, size) / fft_psf(old_maj, old_min, old_pa, size)
    fconv = fimg * ifftshift(krel)
    return ifft2(fconv).real + imean


def fits_reconvolve_image(fitsfile, newpsf, out=None):
    """ Convolve image with deconvolution of (newpsf, oldpsf) """
    newparams = newpsf.to_header_keywords()
    if out is None:
        out = fitsfile
    with pyfits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        currentpsf = Beam.from_fits_header(hdr)
        if currentpsf != newpsf:
            kmaj1 = (currentpsf.major.to('deg').value / hdr['CDELT2'])
            kmin1 = (currentpsf.minor.to('deg').value / hdr['CDELT2'])
            kpa1 = currentpsf.pa.to('deg').value
            kmaj2 = (newpsf.major.to('deg').value / hdr['CDELT2'])
            kmin2 = (newpsf.minor.to('deg').value / hdr['CDELT2'])
            kpa2 = newpsf.pa.to('deg').value
            norm = newpsf.to_value() / currentpsf.to_value()
            if len(hdul[0].data.shape) == 4:
                conv_data = hdul[0].data[0, 0, ...]
            elif len(hdul[0].data.shape) == 2:
                conv_data = hdul[0].data
            conv_data = norm * reconvolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1, kmaj2, kmin2, kpa2)
            if len(hdul[0].data.shape) == 4:
                hdul[0].data[0, 0, ...] = conv_data
            elif len(hdul[0].data.shape) == 2:
                hdul[0].data = conv_data
            hdr.set('BMAJ', newparams['BMAJ'])
            hdr.set('BMIN', newparams['BMIN'])
            hdr.set('BPA', newparams['BPA'])
        pyfits.writeto(out, data=hdul[0].data, header=hdr, overwrite=True)
    return out


def blank_image(template, outfile):
    """Generates an image from a template file filled with NaNs"""
    with pyfits.open(template) as hdul:
        data = hdul[0].data
        nandata = np.full_like(data, np.nan)
    pyfits.writeto(outfile, data=nandata, header=hdul[0].header)