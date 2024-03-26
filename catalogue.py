import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table

import bdsf

import util


def catalogue(self):
    source_finding(self)
    remove_pointing_centres(self)
    RM_PA(self)


def source_finding(self):
    print('Doing source finding')
    source_list = gen_source_list(self)
    gen_catalogue(self, source_list)
    gen_mask(self, source_list)
    gen_normsnr(self, source_list)
    limit_images(self)
    correct_RM_PA_PA0(self)


def remove_pointing_centres(self):
    print('Removing sources at the pointing centres')
    cat = util.load_catalogue(self.polanalysisdir + '/PI_cat.txt')
    ra, dec = util.load_pointing_centres(self.polmosaicdir + '/pointings.txt')
    cat_clean = match_and_remove(self, cat, ra, dec)
    write_cat_clean(self, cat_clean)


def RM_PA(self):
    print('Getting Rotation Measure of sources')
    get_RM(self)


def gen_source_list(self):
    source_list = bdsf.process_image(self.polanalysisdir + '/PI.fits', adaptive_rms_box=True, adaptive_thresh=10.0, rms_box_bright=(20,8), advanced_opts=True, mean_map='map', rms_box=(60,20), rms_map=True, thresh_isl=5.0, thresh_pix=6.25, quiet=True)
    return source_list


def gen_catalogue(self, source_list):
    source_list.write_catalog(outfile=self.polanalysisdir + '/PI_cat.txt', format='ascii', clobber=True)


def gen_mask(self, source_list):
    source_list.export_image(outfile=self.polanalysisdir + '/mask.fits', clobber=True, img_type='island_mask')


def gen_normsnr(self, source_list):
    source_list.export_image(outfile=self.polanalysisdir + '/PI_rms.fits', clobber=True, img_type='rms')
    source_list.export_image(outfile=self.polanalysisdir + '/PI_ch0.fits', clobber=True, img_type='ch0')
    source_list.export_image(outfile=self.polanalysisdir + '/PI_mean.fits', clobber=True, img_type='mean')
    rms_hdu = pyfits.open(self.polanalysisdir + '/PI_rms.fits')[0]
    ch0_hdu = pyfits.open(self.polanalysisdir + '/PI_ch0.fits')[0]
    mean_hdu = pyfits.open(self.polanalysisdir + '/PI_mean.fits')[0]
    rms_data = rms_hdu.data
    ch0_data = ch0_hdu.data
    mean_data = mean_hdu.data
    norm_data = (ch0_data - mean_data)/rms_data
    pyfits.writeto(self.polanalysisdir + '/PI_norm.fits', data=norm_data, header=mean_hdu.header, overwrite=True)


def limit_images(self):
    maskfile = pyfits.open(self.polanalysisdir + '/mask.fits')
    maskfile_data = maskfile[0].data
    maskfile_data[maskfile_data == 0] = np.nan
    # Open the other images and remove all the noise pixels
    for f in ['RM.fits','PA.fits','PA0.fits','RM_err.fits','PA_err.fits','FP.fits']:
        fi = pyfits.open(self.polanalysisdir + '/' + f)
        fi_header = fi[0].header
        fi_data = fi[0].data
        fi_data_blanked = fi_data * maskfile_data
        pyfits.writeto(self.polanalysisdir + '/' + f.strip('.fits') + '_blanked.fits', data=fi_data_blanked, header=fi_header, overwrite=True)


def correct_RM_PA_PA0(self):
    RM_blanked = pyfits.open(self.polanalysisdir + '/RM_blanked.fits')
    PA_blanked = pyfits.open(self.polanalysisdir + '/PA_blanked.fits')
    PA0_blanked = pyfits.open(self.polanalysisdir + '/PA0_blanked.fits')
    RM_blanked_data = RM_blanked[0].data
    PA_blanked_data = PA_blanked[0].data
    PA0_blanked_data = PA0_blanked[0].data
    RM_blanked_header = RM_blanked[0].header
    PA_blanked_header = PA_blanked[0].header
    PA0_blanked_header = PA0_blanked[0].header
    RM_blanked_corr_data = RM_blanked_data * (-1.0)
    PA_blanked_corr_data = (90.0) - PA_blanked_data
    PA0_blanked_corr_data = (90.0) - PA0_blanked_data
    pyfits.writeto(self.polanalysisdir + '/RM_corr_blanked.fits', data=RM_blanked_corr_data, header=RM_blanked_header, overwrite=True)
    pyfits.writeto(self.polanalysisdir + '/PA_corr_blanked.fits', data=PA_blanked_corr_data, header=PA_blanked_header, overwrite=True)
    pyfits.writeto(self.polanalysisdir + '/PA0_corr_blanked.fits', data=PA0_blanked_corr_data, header=PA0_blanked_header, overwrite=True)


def match_and_remove(self, cat, ra, dec):
    # Initialise a table for removing sources
    remove_list = []
    # Get the major beam size of the mosaic
    PI_hdu = pyfits.open(self.polanalysisdir + '/PI.fits')
    PI_hdr = PI_hdu[0].header
    bmaj = PI_hdr['BMAJ']
    cat_ra = cat['RA']
    cat_dec = cat['DEC']
    for s, source in enumerate(cat_ra):
        dist_arr = np.sqrt(np.square(ra - cat_ra[s]) + np.square(dec - cat_dec[s]))
        if np.any(dist_arr<(bmaj/2.0)):
            remove_list.append(s)
    cat.remove_rows([remove_list])
    return cat


def write_cat_clean_TP(self, cat_clean):
    cat_clean.write(self.polanalysisdir + '/TP_cat_clean.txt', format='ascii', overwrite=True)


def match_and_remove_TP(infits, cat, ra, dec):
    # Initialise a table for removing sources
    remove_list = []
    # Get the major beam size of the mosaic
    PI_hdu = pyfits.open(infits)
    PI_hdr = PI_hdu[0].header
    bmaj = PI_hdr['BMAJ']
    cat_ra = cat['RA']
    cat_dec = cat['DEC']
    for s, source in enumerate(cat_ra):
        dist_arr = np.sqrt(np.square(ra - cat_ra[s]) + np.square(dec - cat_dec[s]))
        if np.any(dist_arr<(bmaj/2.0)):
            remove_list.append(s)
    cat.remove_rows([remove_list])
    return cat


def write_cat_clean(self, cat_clean):
    cat_clean.write(self.polanalysisdir + '/PI_cat_clean.txt', format='ascii', overwrite=True)


def get_RM(self):
    cat = Table.read(self.polanalysisdir + '/PI_cat_clean.txt', format='ascii')
    RM_hdu = pyfits.open(self.polanalysisdir + '/RM_corr_blanked.fits')[0]
    RMerr_hdu = pyfits.open(self.polanalysisdir + '/RM_err_blanked.fits')[0]
    RM_data = RM_hdu.data[0,0,:]
    RMerr_data = RMerr_hdu.data[0,0,:]
    RM_arr = np.full(len(cat), np.nan)
    RMerr_arr = np.full(len(cat), np.nan)
    for c,comp in enumerate(cat):
        RM_arr[c] = RM_data[int(np.round(comp['Yposn'])),int(np.round(comp['Xposn']))]
        RMerr_arr[c] = RMerr_data[int(np.round(comp['Yposn'])), int(np.round(comp['Xposn']))]
    cat['RM_Comp'] = RM_arr
    cat['RM_Comp_err'] = RMerr_arr
    write_cat_clean(self, cat)