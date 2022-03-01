import os
from astropy.io import fits as pyfits
import numpy as np
import glob
import shutil

from astropy.table import Table
import bdsf

import util


def leakage(self):
    gen_leakdirs(self)
    copy_imagesV(self)
    generate_imagesFP(self)
    generate_catalogue_I(self)
    generate_catalogue_V(self)
    cross_id_stokes_V(self)


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
        if os.path.isfile(self.contworkdir + '/beams/B' + str(beam).zfill(2) + '.fits'):
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


def generate_catalogue_I(self):
    ra, dec = util.load_pointing_centres(self.polmosaicdir + '/pointings.txt', nan=True)
    for beam in range(40):
        if os.path.isfile(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits'):
            source_list_i = bdsf.process_image(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits', adaptive_rms_box=True, thresh_isl=7.0, thresh_pix=5.0, quiet=True)
            source_list_i.write_catalog(outfile=self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', clobber=True)
            cat_i = util.load_catalogue(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt')
            # Initialise a table for removing sources
            remove_list = []
            # Get the major beam size of the image
            I_hdu = pyfits.open(self.polanalysisleakagedirV + '/I_' + str(beam).zfill(2) + '.fits')
            I_hdr = I_hdu[0].header
            bmaj = I_hdr['BMAJ']
            cat_ra = cat_i['RA']
            cat_dec = cat_i['DEC']
            for s, source in enumerate(cat_ra):
                dist_arr = np.sqrt(np.square(ra[beam] - cat_ra[s]) + np.square(dec[beam] - cat_dec[s]))
                if np.any(dist_arr < (bmaj / 2.0)):
                    remove_list.append(s)
            cat_i.remove_rows([remove_list])
            cat_i = cat_i['RA', 'E_RA', 'DEC', 'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux']
            cat_i.write(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')


def generate_catalogue_V(self):
    ra, dec = util.load_pointing_centres(self.polmosaicdir + '/pointings.txt', nan=True)
    for beam in range(40):
        if os.path.isfile(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_pos.fits'):
            source_list_v = bdsf.process_image(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_pos.fits', adaptive_rms_box=True, thresh_isl=6.25, thresh_pix=7.0, quiet=True)
            source_list_v.write_catalog(outfile=self.polanalysisleakagedirV + '/V_cat_B' + str(beam).zfill(2) + '.txt', format='ascii', clobber=True)
            source_list_v.export_image(outfile=self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_mean.fits', clobber=True, img_type='mean')
            cat_v = util.load_catalogue(self.polanalysisleakagedirV + '/V_cat_B' + str(beam).zfill(2) + '.txt')
            # Initialise a table for removing sources
            remove_list = []
            # Get the major beam size of the image
            V_hdu = pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            V_hdr = V_hdu[0].header
            bmaj = V_hdr['BMAJ']
            cat_ra = cat_v['RA']
            cat_dec = cat_v['DEC']
            for s, source in enumerate(cat_ra):
                dist_arr = np.sqrt(np.square(ra[beam] - cat_ra[s]) + np.square(dec[beam] - cat_dec[s]))
                if np.any(dist_arr < (bmaj / 2.0)):
                    remove_list.append(s)
            cat_v.remove_rows([remove_list])
            V_im = pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits')
            V_im_data = V_im[0].data
            V_mean = pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '_mean.fits')
            V_mean_data = V_mean[0].data
            V_FP_pix_arr = np.full(len(cat_v), np.nan)
            V_FP_im = pyfits.open(self.polanalysisleakagedirV + '/FP_' + str(beam).zfill(2) + '.fits')
            V_FP_im_data = V_FP_im[0].data
            for v, vsource in enumerate(cat_v):
                x_coord = vsource['Xposn']
                y_coord = vsource['Yposn']
                V_FP_pix_arr[v] = V_FP_im_data[0, 0, int(np.round(y_coord)), int(np.round(x_coord))]
                if V_im_data[0, 0, int(np.round(y_coord)), int(np.round(x_coord))] > 0:
                    cat_v['Total_flux'][v] = cat_v['Total_flux'][v] - V_mean_data[0, 0, int(np.round(y_coord)), int(np.round(x_coord))]
                    cat_v['Peak_flux'][v] = cat_v['Peak_flux'][v] - V_mean_data[0, 0, int(np.round(y_coord)), int(np.round(x_coord))]
                else:
                    cat_v['Total_flux'][v] = -1.0 * (cat_v['Total_flux'][v] + V_mean_data[0, 0, int(np.round(y_coord)), int(np.round(x_coord))])
                    cat_v['Peak_flux'][v] = -1.0 * (cat_v['Peak_flux'][v] + V_mean_data[0, 0, int(np.round(y_coord)), int(np.round(x_coord))])
            cat_v = cat_v['RA', 'E_RA', 'DEC', 'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux']
            delta_ra = cat_v['RA'] - ra[beam]
            delta_dec = cat_v['DEC'] - dec[beam]
            dist_arr = np.sqrt(np.square(delta_ra) + np.square(delta_dec))
            cat_v['Dist'] = dist_arr
            cat_v['V_RA_delta'] = delta_ra
            cat_v['V_DEC_delta'] = delta_dec
            cat_v['FP_pix_V'] = V_FP_pix_arr
            cat_v.write(self.polanalysisleakagedirV + '/V_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')


def cross_id_stokes_V(self):
    for beam in range(40):
        if os.path.isfile(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt') and os.path.isfile(self.polanalysisleakagedirV + '/V_cat_B' + str(beam).zfill(2) + '.txt'):
            # Load the two catalogues
            cat_i = Table.read(self.polanalysisleakagedirV + '/I_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')
            cat_v = Table.read(self.polanalysisleakagedirV + '/V_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')
            TP_arr = np.full(len(cat_v), np.nan)
            TPerr_arr = np.full(len(cat_v), np.nan)
            TP_Peak_arr = np.full(len(cat_v), np.nan)
            TP_Peakerr_arr = np.full(len(cat_v), np.nan)
            TP_RA_arr = np.full(len(cat_v), np.nan)
            TP_RAerr_arr = np.full(len(cat_v), np.nan)
            TP_DEC_arr = np.full(len(cat_v), np.nan)
            TP_DECerr_arr = np.full(len(cat_v), np.nan)
            V_FP_arr = np.full(len(cat_v), np.nan)
            V_FPerr_arr = np.full(len(cat_v), np.nan)
            Beam_arr = np.full(len(cat_v), beam)
            obsid_arr = np.full(len(cat_v), self.obsid)
            for v, vsource in enumerate(cat_v):
                ra_source = vsource['RA']
                dec_source = vsource['DEC']
                dist_arr = np.sqrt(np.square(cat_i['RA'] - ra_source) + np.square(cat_i['DEC'] - dec_source))
                min_idx = np.argmin(np.abs(dist_arr))
                # Calculate maximum distance for a match
                with pyfits.open(self.polanalysisleakagedirV + '/V_' + str(beam).zfill(2) + '.fits') as hdu:
                    hdu_header = hdu[0].header
                    bmaj = hdu_header['BMAJ']
                    bmin = hdu_header['BMIN']
                dist = (np.max([bmaj, bmin]) * 1.0)
                if dist_arr[min_idx] <= dist:
                    TP_arr[v] = cat_i[min_idx]['Total_flux']
                    TPerr_arr[v] = cat_i[min_idx]['E_Total_flux']
                    TP_Peak_arr[v] = cat_i[min_idx]['Peak_flux']
                    TP_Peakerr_arr[v] = cat_i[min_idx]['E_Peak_flux']
                    TP_RA_arr[v] = cat_i[min_idx]['RA']
                    TP_RAerr_arr[v] = cat_i[min_idx]['E_RA']
                    TP_DEC_arr[v] = cat_i[min_idx]['DEC']
                    TP_DECerr_arr[v] = cat_i[min_idx]['E_DEC']
                    V_FP_arr[v] = vsource['Peak_flux'] / cat_i[min_idx]['Peak_flux']
                    V_FPerr_arr[v] = np.sqrt((1.0 / cat_i[min_idx]['Peak_flux']) ** 2.0 * vsource['E_Peak_flux'] ** 2.0 + (vsource['Peak_flux'] / (cat_i[min_idx]['Peak_flux'] ** 2.0)) * cat_i[min_idx]['E_Peak_flux'] ** 2.0)
                else:
                    pass
            cat_v['I_flux'] = TP_arr
            cat_v['I_flux_err'] = TPerr_arr
            cat_v['Ip_flux'] = TP_Peak_arr
            cat_v['Ip_flux_err'] = TP_Peakerr_arr
            cat_v['I_RA'] = TP_RA_arr
            cat_v['I_RA_err'] = TP_RAerr_arr
            cat_v['I_DEC'] = TP_DEC_arr
            cat_v['I_DEC_err'] = TP_DECerr_arr
            cat_v['FP_V'] = V_FP_arr
            cat_v['FP_V_err'] = V_FPerr_arr
            cat_v['Beam'] = Beam_arr
            cat_v['Obsid'] = obsid_arr
            cat_v.rename_column('RA', 'V_RA')
            cat_v.rename_column('E_RA', 'V_RA_err')
            cat_v.rename_column('DEC', 'V_DEC')
            cat_v.rename_column('E_DEC', 'V_DEC_err')
            cat_v.rename_column('Total_flux', 'V_flux')
            cat_v.rename_column('E_Total_flux', 'V_flux_err')
            cat_v.rename_column('Peak_flux', 'Vp_flux')
            cat_v.rename_column('E_Peak_flux', 'Vp_flux_err')
            cat_v.write(self.polanalysisleakagedirV + '/FP_cat_B' + str(beam).zfill(2) + '.txt', format='ascii')
        else:
            pass