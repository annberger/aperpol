import os
from astropy.io import fits as pyfits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np
import glob
import shutil

from astropy.table import Table
import bdsf

import util
import misc


def leakage(self):
    gen_leakdirs(self)
#    copy_imagesV(self)
#    generate_imagesFP(self)
#    write_pointing_centres(self)
#    generate_catalogue_I(self)
    calc_leakage_sources(self)


def gen_leakdirs(self):
    # Generate the directories
    self.polanalysisleakagedir = os.path.join(self.polanalysisdir, 'leakage')
    self.polanalysisleakagedirV = os.path.join(self.polanalysisleakagedir, 'V')
    self.polanalysisleakagedirQ = os.path.join(self.polanalysisleakagedir, 'Q')
    if not os.path.isdir(self.polanalysisleakagedir):
        os.makedirs(self.polanalysisleakagedir)
    if not os.path.isdir(self.polanalysisleakagedirV):
        os.makedirs(self.polanalysisleakagedirV)
    if not os.path.isdir(self.polanalysisleakagedirQ):
        os.makedirs(self.polanalysisleakagedirQ)


def copy_imagesV(self):
    for beam in range(40):
        clist = glob.glob(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'continuum') + '/image_mf_[0-9][0-9].fits')
        # vlist = glob.glob(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'continuum') + '/image_mf_V.fits')
        if len(clist) > 0 and os.path.isfile(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits'):
        # if os.path.isfile(self.contworkdir + '/beams/B' + str(beam).zfill(2) + '.fits'):
            # Copy the total power images
            tpimage = glob.glob(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'continuum') + '/image_mf_[0-9][0-9].fits')
            shutil.copy(tpimage[0], self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits')
            # Copy the Stokes V images
            shutil.copy(os.path.join(self.basedir, self.obsid, str(beam).zfill(2), 'polarisation') + '/image_mf_V.fits', self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            # Save a copy of the Stokes V image with all positive values for the source finder
            with pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits') as hdu:
                hdu_header = hdu[0].header
                hdu_data = np.fabs(hdu[0].data)
                pyfits.writeto(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_pos.fits', hdu_data, header=hdu_header, overwrite=True)
        else:
            print('Not all information available for beam ' + str(beam).zfill(2) + '. Discarding beam.')


def generate_imagesFP(self):
    for beam in range(40):
        if os.path.isfile(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits') and os.path.isfile(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits'):
            with pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits') as V:
                V_data = V[0].data
                with pyfits.open(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits') as I:
                    I_header = I[0].header
                    I_data = I[0].data
                    V_data[V_data == 0] = np.nan
                    I_data[I_data == 0] = np.nan
                    FP_data = np.divide(V_data, I_data)
                    pyfits.writeto(self.polanalysisleakagedirV + '/FP_' + str(beam).zfill(2) + '.fits', FP_data, header=I_header, overwrite=True)


def write_pointing_centres(self):
    # Write a file with the central coordinates of each pointing used
    coord_arr = np.full((40, 3), np.nan)
    coord_arr[:, 0] = np.arange(0, 40, 1)
    for b in range(40):
        if os.path.isfile(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation/image_mf_V.fits')):
            vim = pyfits.open(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation/image_mf_V.fits'))[0]
            vim_hdr = vim.header
            coord_arr[b, 1] = vim_hdr['CRVAL1']
            coord_arr[b, 2] = vim_hdr['CRVAL2']
    np.savetxt(self.polanalysisleakagedirV + '/pointings.txt', coord_arr, fmt=['%2s', '%1.13e', '%1.13e'], delimiter='\t')


def generate_catalogue_I(self):
    ra, dec = util.load_pointing_centres(self.polanalysisleakagedirV + '/pointings.txt', nan=True)
    for beam in range(40):
        if os.path.isfile(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits'):
            try:
                source_list_i = bdsf.process_image(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits', adaptive_rms_box=True, thresh_isl=7.0, thresh_pix=5.0, quiet=True)
                source_list_i.write_catalog(outfile=self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', clobber=True)
                cat_i = util.load_catalogue(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt')
                cat_i = cat_i['RA', 'E_RA', 'DEC', 'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'Xposn', 'E_Xposn', 'Yposn', 'E_Yposn', 'Maj', 'E_Maj', 'Min', 'E_Min', 'PA', 'E_PA', 'Isl_rms']
                cat_i.write(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')
            except:
                print(str(beam).zfill(2) + ' was not successfully processed!')


def calc_leakage_sources(self):
    for beam in range(40):
        # Load the Stokes I catalogue
        if os.path.isfile(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt'):
            cat_i = Table.read(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')
            keys = cat_i.meta['comments'][4].split(' ')
            cat_i = Table.read(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', names=keys, format='ascii')
            # Load the corresponding Stokes V image
    #        V_data = misc.load_data(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            V_header = misc.load_header(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            util.remove_dims(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits', self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_rd.fits')
            V_arr = np.full(len(cat_i), np.nan)
            for n, s in enumerate(cat_i):
                # Generate a cutout for each source
                co_size = np.round(np.fabs(s['Maj']/V_header['CDELT1']) + 3.0)
                misc.make_cutout(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_rd.fits', s['RA'], s['DEC'], co_size)
                skycoord = SkyCoord(s['RA'], s['DEC'], unit="deg")
                co_data = misc.load_data(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_rd_cutout.fits')
                co_hdr = misc.load_header(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_rd_cutout.fits')
                w = WCS(co_hdr)
                x, y = w.world_to_pixel(skycoord)
                # Calculate the maximum value inside the ellipse of the component
                maj = np.fabs(s['Maj']/co_hdr['CDELT1'])
                min = np.fabs(s['Min']/co_hdr['CDELT1'])
                in_ellipse = np.vectorize(util.in_ellipse)
                V_mask = in_ellipse(*np.indices(np.squeeze(co_data).shape), x, y, maj/2.0, min/2.0, np.deg2rad(-1.0*s['PA']))
                V_array = np.where(V_mask, co_data, np.nan)
                try:
                    max_pos = np.nanargmax(np.abs(V_array), keepdims=True)
                    V_arr[n] = np.float(np.ndarray.flatten(V_array)[max_pos])
                except ValueError:
                    print('Component too small to calculate leakage!')
                    pass
            cat_i['V_flux'] = V_arr
            cat_i.write(self.polanalysisleakagedirV + '/IV_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', overwrite=True)
