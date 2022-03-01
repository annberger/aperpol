import numpy as np
import copy

from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


def parabolic(f, x):
    """
    Defines a parabolic function to approximate the maximum along the FD-axis
    f(array): 1D array of the FD-axis values
    x(array): index
    returns(float, float): x- and y-position of the maximum of the parabola
    """
    xv = 0.25 * (f[x-1] - f[x+1]) / (f[x-1] - 2.0 * f[x] + f[x+1]) + x
    yv = f[x] - 0.25 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def load_header(image):
    """
    Returns the header of a fits file
    image: Image or cube in FITS-format
    return(FITS header object): Header of a FITS-file
    """
    hdu = pyfits.open(image)
    header = hdu[0].header
    return header


def load_data(image):
    """
    Returns the data of a fits file
    image: Image or cube in FITS-format
    return(FITS data array): Data of a FITS-file
    """
    hdu = pyfits.open(image)
    data = hdu[0].data
    return data


def make_cutout(image, ra, dec, size):
    """
    Generates a cutout with an updated header for an image
    """
    with pyfits.open(image, memmap=0) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        skycoord = SkyCoord(ra, dec, unit="deg")
        cutout = Cutout2D(hdu.data, position=skycoord, size=size, wcs=wcs, mode='partial', fill_value=np.nan)
        hdu.data = cutout.data
        hdu.header.update(cutout.wcs.to_header())
        cutout_filename = image.rstrip('.fits') + '_cutout.fits'
        hdu.writeto(cutout_filename, overwrite=True)


def calc_sigmas(data):
    meandis = np.nanmean(data)
    if np.isnan(meandis):
        mindis = 0.0
        maxdis = 1.0
    else:
        oldstd = 0.000011
        std = np.nanstd(data)
        while not 0.95 * oldstd < std < 1.05 * oldstd:
            newdata = copy.deepcopy(data)
            newdata[np.where(newdata > meandis + 7 * std)] = float('NaN')
            newdata[np.where(newdata < meandis - 7 * std)] = float('NaN')
            oldstd = std
            std = np.nanstd(newdata)
            meandis = np.nanmean(newdata)
        mindis = meandis - 3 * std
        maxdis = meandis + 7 * std
    return mindis, maxdis


def make_source_id(ra, dec, prefix):
    """
    Generate the name of the source using its ra and dec coordinates
    ra (float): RA in deg
    dec (float): DEC in deg
    return (string): source name
    """
    RA_str = str(np.around(ra, decimals=3))
    DEC_str = str(np.around(dec, decimals=3))
    RA_list = RA_str.split('.')
    DEC_list = DEC_str.split('.')
    len_RA_str = len(RA_list[-1])
    len_DEC_str = len(DEC_list[-1])
    last_string = '{:<03}'
    if len_RA_str < 3:
        RA = str(RA_list[0]) + '.' + last_string.format(str(RA_list[1]))
    else:
        RA = RA_str
    if len_DEC_str < 3:
        DEC = str(DEC_list[0]) + '.' + last_string.format(str(DEC_list[1]))
    else:
        DEC = DEC_str
    sourcestring = prefix + '_' + RA + '+' + DEC
    return sourcestring


def shift_marker(coords, shift):
    if coords[0] < 150.0:
        x = coords[0] + shift
    else:
        x = coords[0] - shift
    return x, coords[1]


def define_markersize(text):
    if len(text) == 1:
        ms = 8
    elif len(text) == 2:
        ms = 12
    elif len(text) == 3:
        ms = 18
    elif len(text) == 4:
        ms = 24
    return ms