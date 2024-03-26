import numpy as np

from astroquery.irsa import Irsa
from astroquery.sdss import SDSS
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table
import bdsf
from astropy.io import fits as pyfits

import os
import glob

import util
import catalogue
import misc


def cross_id(self):
    PI_filt = load_filter_rename_PIcat(self)
    TP_cat = gen_TP_cat(self)
    PI_TP = cross_match_TP(self, PI_filt, TP_cat)
    print(PI_TP.colnames)
    PI_wise = cross_match_wise(self, PI_TP)
    PI_nvss = cross_match_nvss(self, PI_wise)
    PI_sdss = cross_match_sdss(self, PI_nvss)
    write_cross_matched_cat(self, PI_sdss)


def load_filter_rename_PIcat(self):
    PI_cat = Table.read(self.polanalysisdir + '/PI_cat_clean.txt', format='ascii')
    PI_cat_filtered = PI_cat['Isl_id','RA','E_RA','DEC','E_DEC','Isl_Total_flux','E_Isl_Total_flux','Total_flux','E_Total_flux','Peak_flux','E_Peak_flux','S_Code','RM_Comp','RM_Comp_err','Isl_rms','DC_Maj','DC_Min','DC_PA']
    PI_cat_filtered.rename_column('RA', 'RA_Comp')
    PI_cat_filtered.rename_column('E_RA', 'RA_Comp_err')
    PI_cat_filtered.rename_column('DEC', 'DEC_Comp')
    PI_cat_filtered.rename_column('E_DEC', 'DEC_Comp_err')
    PI_cat_filtered.rename_column('Total_flux', 'PI_Comp_fit') # Changed from PI_Comp
    PI_cat_filtered.rename_column('E_Total_flux', 'PI_Comp_fit_err') # Changed from PI_Comp_err
    PI_cat_filtered.rename_column('Isl_Total_flux', 'PI_Comp_Isl') # This is no component flux! Only taken as this, to not loose the ISL flux and for cases where multiple Isl get combined!
    PI_cat_filtered.rename_column('E_Isl_Total_flux', 'PI_Comp_Isl_err')
    PI_cat_filtered.rename_column('Peak_flux', 'PI_Comp_Peak')
    PI_cat_filtered.rename_column('E_Peak_flux', 'PI_Comp_Peak_err')
    PI_cat_filtered.rename_column('Isl_rms', 'PI_rms')
    # Generate the source names and the correct RA, DEC, fluxes etc. for the sources by combining components
    source_ids = np.full(len(PI_cat_filtered), '', dtype='S20')
    RA_arr = np.full(len(PI_cat_filtered), np.nan)
    RAerr_arr = np.full(len(PI_cat_filtered), np.nan)
    DEC_arr = np.full(len(PI_cat_filtered), np.nan)
    DECerr_arr = np.full(len(PI_cat_filtered), np.nan)

    PI_arr = np.full(len(PI_cat_filtered), np.nan)
    PIerr_arr = np.full(len(PI_cat_filtered), np.nan)
    PIIsl_arr = np.full(len(PI_cat_filtered), np.nan)
    PIIslerr_arr = np.full(len(PI_cat_filtered), np.nan)
    PIrms_arr = np.full(len(PI_cat_filtered), np.nan)
    islands = np.unique(PI_cat_filtered['Isl_id'])
    for isl in islands:
        islidxs = np.where(isl == PI_cat_filtered['Isl_id'])[0] # grouping sources by Isl
        if len(islidxs) == 1:
            # single component case
            RA_arr[islidxs] = PI_cat_filtered['RA_Comp'][islidxs]
            RAerr_arr[islidxs] = PI_cat_filtered['RA_Comp_err'][islidxs]
            DEC_arr[islidxs] = PI_cat_filtered['DEC_Comp'][islidxs]
            DECerr_arr[islidxs] = PI_cat_filtered['DEC_Comp_err'][islidxs]
            PI_arr[islidxs] = PI_cat_filtered['PI_Comp_fit'][islidxs]
            PIerr_arr[islidxs] = PI_cat_filtered['PI_Comp_fit_err'][islidxs]
            PIIsl_arr[islidxs] = PI_cat_filtered['PI_Comp_Isl'][islidxs]
            PIIslerr_arr[islidxs] = PI_cat_filtered['PI_Comp_Isl_err'][islidxs]
            PIrms_arr[islidxs] = PI_cat_filtered['PI_rms'][islidxs]
            # Generate the source name
            src_name = misc.make_source_id(RA_arr[islidxs][0], DEC_arr[islidxs][0], self.prefix)
            source_ids[islidxs] = src_name
        else:
            # multi component case
            RA_arr[islidxs] = np.mean(PI_cat_filtered['RA_Comp'][islidxs])
            RAerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(PI_cat_filtered['RA_Comp_err'][islidxs])))))
            DEC_arr[islidxs] = np.mean(PI_cat_filtered['DEC_Comp'][islidxs])
            DECerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(PI_cat_filtered['DEC_Comp_err'][islidxs])))))
            PI_arr[islidxs] = np.full(len(islidxs), np.sum(PI_cat_filtered['PI_Comp_fit'][islidxs]))
            PIerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(PI_cat_filtered['PI_Comp_fit_err'][islidxs])))))
            # Isl flux is always all Comp in Isl
            PIIsl_arr[islidxs] = np.full(len(islidxs), PI_cat_filtered['PI_Comp_Isl'][islidxs[0]])
            PIIslerr_arr[islidxs] = list(np.full(len(islidxs), PI_cat_filtered['PI_Comp_Isl_err'][islidxs[0]]))
            PIrms_arr[islidxs] = PI_cat_filtered['PI_rms'][islidxs[0]]
            # Generate the source name
            src_name = misc.make_source_id(RA_arr[islidxs][0], DEC_arr[islidxs][0], self.prefix)
            source_ids[islidxs] = np.full(len(islidxs), src_name)
    for s in PI_cat_filtered:
        if (s['DC_Maj'] > 0.0) or (s['DC_Min'] > 0.0) or (s['DC_PA'] > 0.0) or s['S_Code'] != 'S':
            s['S_Code'] = 'E'
        else:
            s['S_Code'] = 'S'
    PI_cat_filtered['ID'] = source_ids # assumes one Isl is one source!
    PI_cat_filtered['RA'] = RA_arr
    PI_cat_filtered['RA_err'] = RAerr_arr
    PI_cat_filtered['DEC'] = DEC_arr
    PI_cat_filtered['DEC_err'] = DECerr_arr
    PI_cat_filtered['PI_fit'] = PI_arr # previously PI
    PI_cat_filtered['PI_fit_err'] = PIerr_arr
    PI_cat_filtered['PI_Isl'] = PIIsl_arr
    PI_cat_filtered['PI_Isl_err'] = PIIslerr_arr
    PI_cat_filtered['PI_rms'] = PIrms_arr
    #PI_cat_filtered.remove_column('Isl_id')
    PI_cat_filtered.remove_column('DC_Maj')
    PI_cat_filtered.remove_column('DC_Min')
    PI_cat_filtered.remove_column('DC_PA')
    return PI_cat_filtered


def gen_TP_cat(self):
    contimagename = sorted(glob.glob(self.contmosaicdir + '/*.fits'))[0]
    contuncorrimagename = sorted(glob.glob(self.contmosaicdir + '/*_uncorr.fits'))[0]
#    source_list = bdsf.process_image(contimagename, detection_image=contuncorrimagename, adaptive_rms_box=True, thresh_isl=4.0, thresh_pix=5.0, quiet=True, rms_box=(100,10), rms_box_bright=(20,2), rms_map=True, group_by_isl=False, group_tol=10.0, output_opts=True, output_all=True, advanced_opts=True, blank_limit=None, mean_map='zero', spline_rank=1)
    source_list = bdsf.process_image(contimagename, detection_image=contuncorrimagename, adaptive_rms_box=True, thresh_isl=4.0, thresh_pix=5.0, quiet=True, rms_box=(100,10), rms_box_bright=(20,2), rms_map=True, group_by_isl=False, group_tol=10.0, advanced_opts=True, blank_limit=None, mean_map='zero', spline_rank=1)
    source_list.export_image(outfile=self.polanalysisdir + '/TP_rms.fits', clobber=True, img_type='rms')
    source_list.export_image(outfile=self.polanalysisdir + '/TP_ch0.fits', clobber=True, img_type='ch0')
    source_list.export_image(outfile=self.polanalysisdir + '/TP_mean.fits', clobber=True, img_type='mean')
    source_list.export_image(outfile=self.polanalysisdir + '/TP_mask.fits', clobber=True, img_type='island_mask')
    rms_hdu = pyfits.open(self.polanalysisdir + '/TP_rms.fits')[0]
    ch0_hdu = pyfits.open(self.polanalysisdir + '/TP_ch0.fits')[0]
    mean_hdu = pyfits.open(self.polanalysisdir + '/TP_mean.fits')[0]
    rms_data = rms_hdu.data
    ch0_data = ch0_hdu.data
    mean_data = mean_hdu.data
    norm_data = (ch0_data - mean_data)/rms_data
    pyfits.writeto(self.polanalysisdir + '/TP_norm.fits', data=norm_data, header=mean_hdu.header, overwrite=True)
    source_list.write_catalog(outfile=self.polanalysisdir + '/TP_cat.txt', format='ascii', clobber=True)
    cat = util.load_catalogue(self.polanalysisdir + '/TP_cat.txt')
    print('Removing components at the pointing centres for continuum catalogue')
    ra, dec = util.load_pointing_centres(self.polmosaicdir + '/pointings.txt')
    cat_clean = catalogue.match_and_remove_TP(contimagename, cat, ra, dec)
    catalogue.write_cat_clean_TP(self, cat_clean)
    TP_cat = Table.read(self.polanalysisdir + '/TP_cat_clean.txt', format='ascii')
    TP_cat_filtered = TP_cat['Isl_id','RA','E_RA','DEC','E_DEC','Total_flux','E_Total_flux','Peak_flux','E_Peak_flux','S_Code','Isl_rms']
    TP_cat_filtered.rename_column('RA', 'RA_Comp_TP')
    TP_cat_filtered.rename_column('E_RA', 'RA_Comp_TP_err')
    TP_cat_filtered.rename_column('DEC', 'DEC_Comp_TP')
    TP_cat_filtered.rename_column('E_DEC', 'DEC_Comp_TP_err')
    TP_cat_filtered.rename_column('Total_flux', 'TP_Comp')
    TP_cat_filtered.rename_column('E_Total_flux', 'TP_Comp_err')
    TP_cat_filtered.rename_column('Peak_flux', 'TP_Comp_Peak')
    TP_cat_filtered.rename_column('E_Peak_flux', 'TP_Comp_Peak_err')
    TP_cat_filtered.rename_column('Isl_rms','TP_rms')
    # Correct RA, DEC, fluxes etc. for the sources by combining components
    RA_arr = np.full(len(TP_cat_filtered), np.nan)
    RAerr_arr = np.full(len(TP_cat_filtered), np.nan)
    DEC_arr = np.full(len(TP_cat_filtered), np.nan)
    DECerr_arr = np.full(len(TP_cat_filtered), np.nan)
    PI_arr = np.full(len(TP_cat_filtered), np.nan)
    PIerr_arr = np.full(len(TP_cat_filtered), np.nan)
    TPrms_arr = np.full(len(TP_cat_filtered), np.nan)
    islands = np.unique(TP_cat_filtered['Isl_id'])
    for isl in islands:
        islidxs = np.where(isl == TP_cat_filtered['Isl_id'])[0]
        if len(islidxs) == 1:
            RA_arr[islidxs] = TP_cat_filtered['RA_Comp_TP'][islidxs]
            RAerr_arr[islidxs] = TP_cat_filtered['RA_Comp_TP_err'][islidxs]
            DEC_arr[islidxs] = TP_cat_filtered['DEC_Comp_TP'][islidxs]
            DECerr_arr[islidxs] = TP_cat_filtered['DEC_Comp_TP_err'][islidxs]
            PI_arr[islidxs] = TP_cat_filtered['TP_Comp'][islidxs]
            PIerr_arr[islidxs] = TP_cat_filtered['TP_Comp_err'][islidxs]
            TPrms_arr[islidxs] = TP_cat_filtered['TP_rms'][islidxs]
        else:
            RA_arr[islidxs] = np.mean(TP_cat_filtered['RA_Comp_TP'][islidxs])
            RAerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(TP_cat_filtered['RA_Comp_TP_err'][islidxs])))))
            DEC_arr[islidxs] = np.mean(TP_cat_filtered['DEC_Comp_TP'][islidxs])
            DECerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(TP_cat_filtered['DEC_Comp_TP_err'][islidxs])))))
            PI_arr[islidxs] = np.full(len(islidxs), np.sum(TP_cat_filtered['TP_Comp'][islidxs]))
            PIerr_arr[islidxs] = list(np.full(len(islidxs), np.sqrt(np.sum(np.square(TP_cat_filtered['TP_Comp_err'][islidxs])))))
            TPrms_arr[islidxs] = np.mean(TP_cat_filtered['TP_rms'][islidxs])
    TP_cat_filtered['RA_TP'] = RA_arr
    TP_cat_filtered['RA_TP_err'] = RAerr_arr
    TP_cat_filtered['DEC_TP'] = DEC_arr
    TP_cat_filtered['DEC_TP_err'] = DECerr_arr
    TP_cat_filtered['TP'] = PI_arr
    TP_cat_filtered['TP_err'] = PIerr_arr
    TP_cat_filtered['TP_rms'] = TPrms_arr
    # Generate source IDs for total power
    TP_ID_arr = np.full(len(TP_cat_filtered), '', dtype='S20')
    for s, source in enumerate(TP_cat_filtered):
        TP_ID_arr[s] = misc.make_source_id(TP_cat_filtered['RA_TP'][s], TP_cat_filtered['DEC_TP'][s], 'TP')
        TP_cat_filtered['TP_ID'] = TP_ID_arr
    TP_cat_filtered.remove_column('Isl_id')
    new_order = ['TP_ID','RA_TP','RA_TP_err','DEC_TP','DEC_TP_err','TP','TP_err','S_Code','RA_Comp_TP','RA_Comp_TP_err','DEC_Comp_TP','DEC_Comp_TP_err','TP_Comp','TP_Comp_err','TP_Comp_Peak','TP_Comp_Peak_err','TP_rms']
    tp_new_order = TP_cat_filtered[new_order]
    tp_new_order.write(self.polanalysisdir + '/TP_cat_final.txt', format='ascii', overwrite=True)
    return TP_cat_filtered


def cross_match_TP(self, PI_cat, TP_cat):
    # Generate the columns for the TP information
    TP_id_arr = np.full(len(PI_cat), '', dtype='S20')
    TP_arr = np.full(len(PI_cat), np.nan)
    TP_err_arr = np.full(len(PI_cat), np.nan)
    TP_ra_arr = np.full(len(PI_cat), np.nan)
    TP_dec_arr = np.full(len(PI_cat), np.nan)
    TP_ra_err_arr = np.full(len(PI_cat), np.nan)
    TP_dec_err_arr = np.full(len(PI_cat), np.nan)
    FPfit_arr = np.full(len(PI_cat), np.nan)
    FPfit_err_arr = np.full(len(PI_cat), np.nan)
    FPIsl_arr = np.full(len(PI_cat), np.nan)
    FPIsl_err_arr = np.full(len(PI_cat), np.nan)
    for s, source in enumerate(PI_cat):
        ra_source = source['RA']
        dec_source = source['DEC']
        dist_arr = np.sqrt(np.square(TP_cat['RA_TP']-ra_source) + np.square(TP_cat['DEC_TP']-dec_source))
        min_idx = np.argmin(np.abs(dist_arr))
        # Calculate maximum distance for a match
        bmaj, bmin = util.get_beam(self)
        dist = (np.max([bmaj, bmin]) * 1.0)
        if dist_arr[min_idx] <= dist:
            TP_id_arr[s] = TP_cat[min_idx]['TP_ID']
            TP_arr[s] = TP_cat[min_idx]['TP']
            TP_err_arr[s] = TP_cat[min_idx]['TP_err']
            TP_ra_arr[s] = TP_cat[min_idx]['RA_TP']
            TP_ra_err_arr[s] = TP_cat[min_idx]['RA_TP_err']
            TP_dec_arr[s] = TP_cat[min_idx]['DEC_TP']
            TP_dec_err_arr[s] = TP_cat[min_idx]['DEC_TP_err']
            FPfit_arr[s] = source['PI_fit']/TP_cat[min_idx]['TP']
            FPfit_err_arr[s] = np.sqrt((1.0/TP_cat[min_idx]['TP'])**2.0 * source['PI_fit_err']**2.0 + (source['PI_fit'] / (TP_cat[min_idx]['TP']**2.0)) * TP_cat[min_idx]['TP_err']**2.0)
            FPIsl_arr[s] = source['PI_Isl'] / TP_cat[min_idx]['TP']
            FPIsl_err_arr[s] = np.sqrt((1.0 / TP_cat[min_idx]['TP']) ** 2.0 * source['PI_Isl_err'] ** 2.0 + (
                        source['PI_Isl'] / (TP_cat[min_idx]['TP'] ** 2.0)) * TP_cat[min_idx]['TP_err'] ** 2.0)
        else:
            pass
    PI_cat['TP_ID'] = TP_id_arr
    PI_cat['TP'] = TP_arr
    PI_cat['TP_err'] = TP_err_arr
    PI_cat['RA_TP'] = TP_ra_arr
    PI_cat['RA_TP_err'] = TP_ra_err_arr
    PI_cat['DEC_TP'] = TP_dec_arr
    PI_cat['DEC_TP_err'] = TP_dec_err_arr
    PI_cat['FP_fit'] = FPfit_arr
    PI_cat['FP_fit_err'] = FPfit_err_arr
    PI_cat['FP_Isl'] = FPIsl_arr
    PI_cat['FP_Isl_err'] = FPIsl_err_arr
    # Combine different polarised sources, which have the same total power counterpart
    for tp_source_ra in np.unique(TP_ra_arr):
        nsources = np.where(TP_ra_arr == tp_source_ra)
        if len(nsources[0]) < 2:
            pass
        else:
            if len(np.unique(PI_cat['ID'][nsources])) > 1: # check if multi match is with multi Isl
                for pis in nsources:
                    PI_cat['RA'][pis] = np.mean(PI_cat['RA'][nsources])
                    PI_cat['RA_err'][pis] = np.sqrt(np.sum(np.square(PI_cat['RA_err'][pis])))
                    PI_cat['DEC'][pis] = np.mean(PI_cat['DEC'][nsources])
                    PI_cat['DEC_err'][pis] = np.sqrt(np.sum(np.square(PI_cat['DEC_err'][pis])))
                    PI_cat['ID'][pis] = misc.make_source_id(PI_cat['RA'][pis][0], PI_cat['DEC'][pis][0], self.prefix)
                    PI_cat['PI_fit'][pis] = np.sum(PI_cat['PI_fit'][nsources])
                    PI_cat['PI_fit_err'][pis] = np.sqrt(np.sum(np.square(PI_cat['PI_fit_err'][nsources])))
                    un_IDs = np.unique(PI_cat['ID'][nsources])
                    un_idx = []
                    for un_ID in un_IDs:
                        un_idx.append(np.where(PI_cat['ID'] == un_ID)[0])
                    PI_cat['PI_Isl'][pis] = np.sum(PI_cat['PI_fit'][un_idx])
                    PI_cat['PI_fit_err'][pis] = np.sqrt(np.sum(np.square(PI_cat['PI_fit_err'][un_idx])))
                    PI_cat['S_Code'][pis] = 'E'
                    PI_cat['FP_fit'][pis] = PI_cat['PI_fit'][pis] / PI_cat['TP'][pis]
                    PI_cat['FP_fit_err'][pis] = np.sqrt((1.0 / PI_cat['TP'][nsources]) ** 2.0 * PI_cat['PI_fit_err'][nsources] ** 2.0 + (PI_cat['PI_fit'][nsources] / (PI_cat['TP'][nsources] ** 2.0)) * PI_cat['TP_err'][nsources] ** 2.0)
                    PI_cat['FP_Isl'][pis] = PI_cat['PI_Isl'][pis] / PI_cat['TP'][pis]
                    PI_cat['FP_Isl_err'][pis] = np.sqrt((1.0 / PI_cat['TP'][nsources]) ** 2.0 * PI_cat['PI_Isl_err'][nsources] ** 2.0 + (
                                    PI_cat['PI_Isl'][nsources] / (PI_cat['TP'][nsources] ** 2.0)) * PI_cat['TP_err'][nsources] ** 2.0)
            else:
                pass
    return PI_cat


def cross_match_wise(self, cat):
    # Calculate maximum distance for a match
    bmaj, bmin = util.get_beam(self)
    dist = (np.max([bmaj, bmin])/2.0)*3600.0
    # Generate the list for the additional information
    dist_arr = np.full(len(cat), np.nan)
    designation_arr = np.full(len(cat), np.nan, dtype='U20')
    wise_ra_arr = np.full(len(cat), np.nan)
    wise_ra_err_arr = np.full(len(cat), np.nan)
    wise_dec_arr = np.full(len(cat), np.nan)
    wise_dec_err_arr = np.full(len(cat), np.nan)
    w1mpro_arr = np.full(len(cat), np.nan)
    w1mpro_err_arr = np.full(len(cat), np.nan)
    w1snr_arr = np.full(len(cat), np.nan)
    w2mpro_arr = np.full(len(cat), np.nan)
    w2mpro_err_arr = np.full(len(cat), np.nan)
    w2snr_arr = np.full(len(cat), np.nan)
    w3mpro_arr = np.full(len(cat), np.nan)
    w3mpro_err_arr = np.full(len(cat), np.nan)
    w3snr_arr = np.full(len(cat), np.nan)
    w4mpro_arr = np.full(len(cat), np.nan)
    w4mpro_err_arr = np.full(len(cat), np.nan)
    w4snr_arr = np.full(len(cat), np.nan)
    # Cross match WISE sources with Apertif source catalogue
    for s, source in enumerate(cat):
        # First match with the total power source if available otherwise with the PI source
        if np.isnan(source['RA_TP']):
            match = []
        else:
            try:
                match = Irsa.query_region(coord.SkyCoord(source['RA_TP'], source['DEC_TP'], unit=(u.deg, u.deg), frame='icrs'), catalog='allwise_p3as_psd', radius=dist * u.arcsec)
            except:
                match = []
        if len(match) == 0:
            pass
        else:
            dist_arr[s] = match['dist'][0]
            designation_arr[s] = match['designation'][0]
            wise_ra_arr[s] = match['ra'][0]
            wise_ra_err_arr[s] = match['sigra'][0]
            wise_dec_arr[s] = match['dec'][0]
            wise_dec_err_arr[s] = match['sigdec'][0]
            w1mpro_arr[s] = match['w1mpro'][0]
            w1mpro_err_arr[s] = match['w1sigmpro'][0]
            w1snr_arr[s] = match['w1snr'][0]
            w2mpro_arr[s] = match['w2mpro'][0]
            w2mpro_err_arr[s] = match['w2sigmpro'][0]
            w2snr_arr[s] = match['w2snr'][0]
            w3mpro_arr[s] = match['w3mpro'][0]
            w3mpro_err_arr[s] = match['w3sigmpro'][0]
            w3snr_arr[s] = match['w3snr'][0]
            w4mpro_arr[s] = match['w4mpro'][0]
            w4mpro_err_arr[s] = match['w4sigmpro'][0]
            w4snr_arr[s] = match['w4snr'][0]
#    cat['WISE_Dist'] = dist_arr
    cat['WISE_ID'] = designation_arr
    cat['WISE_RA'] = wise_ra_arr
    cat['WISE_RA_err'] = wise_ra_err_arr
    cat['WISE_DEC'] = wise_dec_arr
    cat['WISE_DEC_err'] = wise_dec_err_arr
    cat['WISE_Flux_3.4'] = w1mpro_arr
    cat['WISE_Flux_3.4_err'] = w1mpro_err_arr
    cat['WISE_SNR_3.4'] = w1snr_arr
    cat['WISE_Flux_4.6'] = w2mpro_arr
    cat['WISE_Flux_4.6_err'] = w2mpro_err_arr
    cat['WISE_SNR_4.6'] = w2snr_arr
    cat['WISE_Flux_12'] = w3mpro_arr
    cat['WISE_Flux_12_err'] = w3mpro_err_arr
    cat['WISE_SNR_12'] = w3snr_arr
    cat['WISE_Flux_22'] = w4mpro_arr
    cat['WISE_Flux_22_err'] = w4mpro_err_arr
    cat['WISE_SNR_22'] = w4snr_arr
    return cat


def cross_match_nvss(self, cat):
    # Calculate maximum distance for a match
    bmaj, bmin = util.get_beam(self)
    limit = (np.max([bmaj, bmin])/2.0)*3600.0
    # Load the NVSS RM-catalogue
    nvss_cat = Table.read((os.path.dirname(util.__file__)) + '/catalogues/RMCatalogue.txt', format='ascii')
    # Convert the coordinates of the RM catalogue to the needed format
    ra_str_arr = np.full(len(nvss_cat), np.nan, dtype='S11')
    dec_str_arr = np.full(len(nvss_cat), np.nan, dtype='S12')
    for r in range(len(nvss_cat)):
        ra_str_arr[r] = str(nvss_cat['col1'][r]).zfill(2) + ' ' + str(nvss_cat['col2'][r]).zfill(2) + ' ' + str(nvss_cat['col3'][r])
        dec_str_arr[r] = str(nvss_cat['col6'][r]) + ' ' + str(nvss_cat['col7'][r]).zfill(2) + ' ' + str(nvss_cat['col8'][r])
    nvss_cat['ra'] = ra_str_arr
    nvss_cat['dec'] = dec_str_arr
    # Cross match NVSS source catalogue with Apertif component catalogue
    coords_nvss = coord.SkyCoord(ra=nvss_cat['ra'], dec=nvss_cat['dec'], unit=(u.hourangle, u.deg))
    coords_pi = coord.SkyCoord(ra=cat['RA_Comp'], dec=cat['DEC_Comp'], unit=(u.deg, u.deg))
    idx, d2d, d3d = coords_pi.match_to_catalog_sky(coords_nvss)
    dist = (d2d * u.deg * 3600) / (u.deg * u.deg)
    match = np.where(dist <= limit)  # Index of sources with match
    idx_match = idx[match]
    # Generate the lists for the additional NVSS information
    nvss_ra_arr = np.full(len(cat), np.nan)
    nvss_raerr_arr = np.full(len(cat), np.nan)
    nvss_dec_arr = np.full(len(cat), np.nan)
    nvss_decerr_arr = np.full(len(cat), np.nan)
    nvss_I_arr = np.full(len(cat), np.nan)
    nvss_Ierr_arr = np.full(len(cat), np.nan)
    nvss_PI_arr = np.full(len(cat), np.nan)
    nvss_PIerr_arr = np.full(len(cat), np.nan)
    nvss_FP_arr = np.full(len(cat), np.nan)
    nvss_FPerr_arr = np.full(len(cat), np.nan)
    nvss_RM_arr = np.full(len(cat), np.nan)
    nvss_RMerr_arr = np.full(len(cat), np.nan)
    for s,source in enumerate(match[0]):
        nvss_ra_arr[source] = ((coords_nvss.ra[idx_match[s]] * u.deg)/(u.deg*u.deg))
        nvss_raerr_arr[source] = nvss_cat['col5'][idx_match[s]]
        nvss_dec_arr[source] = ((coords_nvss.dec[idx_match[s]] * u.deg)/(u.deg*u.deg))
        nvss_decerr_arr[source] = nvss_cat['col10'][idx_match[s]]
        nvss_I_arr[source] = nvss_cat['col13'][idx_match[s]] / 1000.0
        nvss_Ierr_arr[source] = nvss_cat['col15'][idx_match[s]] / 1000.0
        nvss_PI_arr[source] = nvss_cat['col16'][idx_match[s]] / 1000.0
        nvss_PIerr_arr[source] = nvss_cat['col18'][idx_match[s]] / 1000.0
        nvss_FP_arr[source] = nvss_cat['col19'][idx_match[s]] / 100.0
        nvss_FPerr_arr[source] = nvss_cat['col21'][idx_match[s]] / 100.0
        nvss_RM_arr[source] = nvss_cat['col22'][idx_match[s]]
        nvss_RMerr_arr[source] = nvss_cat['col24'][idx_match[s]]
    # Add the columns to the catalogue
    cat['NVSS_RA'] = nvss_ra_arr
    cat['NVSS_RA_err'] = nvss_raerr_arr
    cat['NVSS_DEC'] = nvss_dec_arr
    cat['NVSS_DEC_err'] = nvss_decerr_arr
    cat['NVSS_I'] = nvss_I_arr
    cat['NVSS_I_err'] = nvss_Ierr_arr
    cat['NVSS_PI'] = nvss_PI_arr
    cat['NVSS_PI_err'] = nvss_PIerr_arr
    cat['NVSS_FP'] = nvss_FP_arr
    cat['NVSS_FP_err'] = nvss_FPerr_arr
    cat['NVSS_RM'] = nvss_RM_arr
    cat['NVSS_RM_err'] = nvss_RMerr_arr
    return cat


def cross_match_sdss(self, cat):
    # Calculate maximum distance for a match
    bmaj, bmin = util.get_beam(self)
    dist = (np.max([bmaj, bmin])/2.0)*3600.0
    # Generate the list for the additional information
    objid_arr = np.full(len(cat), np.nan, dtype='U18')
    sdss_ra_arr = np.full(len(cat), np.nan)
    sdss_dec_arr = np.full(len(cat), np.nan)
    sdss_u_arr = np.full(len(cat), np.nan)
    sdss_uerr_arr = np.full(len(cat), np.nan)
    sdss_g_arr = np.full(len(cat), np.nan)
    sdss_gerr_arr = np.full(len(cat), np.nan)
    sdss_r_arr = np.full(len(cat), np.nan)
    sdss_rerr_arr = np.full(len(cat), np.nan)
    sdss_i_arr = np.full(len(cat), np.nan)
    sdss_ierr_arr = np.full(len(cat), np.nan)
    sdss_z_arr = np.full(len(cat), np.nan)
    sdss_zerr_arr = np.full(len(cat), np.nan)
    sdss_rs_arr = np.full(len(cat), np.nan)
    sdss_rserr_arr = np.full(len(cat), np.nan)
    # Cross match SDSS sources with WISE coordinates of Apertif polarised source catalogue
    for s, source in enumerate(cat):
        if np.isnan(source['WISE_RA']):
            pass
        else:
            match = SDSS.query_region(coord.SkyCoord(source['WISE_RA'], source['WISE_DEC'], unit=(u.deg, u.deg), frame='icrs'), photoobj_fields=['objid','ra','dec','u','err_u','g','err_g','r','err_r','i','err_i','z','err_z','type'], spectro=False, radius=3.2 * u.arcsec, data_release=16)
            #print('TEST',match)
            if match is not None: #if (len(np.array(match).flatten()) != None): #if match != None:
                match_gal = match[np.where(match['type'] == 3)]
                if len(match_gal) != 0:
                    dist = np.sqrt(np.square(match_gal['ra']-source['WISE_RA']) + np.square(match_gal['dec']-source['WISE_DEC']))
                    minidx = np.argmin(dist)
                    sdss_src = match_gal[minidx]
                    objid_arr[s] = sdss_src['objid']
                    sdss_ra_arr[s] = sdss_src['ra']
                    sdss_dec_arr[s] = sdss_src['dec']
                    if sdss_src['u'] < 0.0:
                        sdss_u_arr[s] = np.nan
                    else:
                        sdss_u_arr[s] = sdss_src['u']
                    if sdss_src['err_u'] < 0.0:
                        sdss_uerr_arr[s] = np.nan
                    else:
                        sdss_uerr_arr[s] = sdss_src['err_u']
                    if sdss_src['g'] < 0.0:
                        sdss_g_arr[s] = np.nan
                    else:
                        sdss_g_arr[s] = sdss_src['g']
                    if sdss_src['err_g'] < 0.0:
                        sdss_gerr_arr[s] = np.nan
                    else:
                        sdss_gerr_arr[s] = sdss_src['err_g']
                    if sdss_src['r'] < 0.0:
                        sdss_r_arr[s] = np.nan
                    else:
                        sdss_r_arr[s] = sdss_src['r']
                    if sdss_src['err_r'] < 0.0:
                        sdss_rerr_arr[s] = np.nan
                    else:
                        sdss_rerr_arr[s] = sdss_src['err_r']
                    if sdss_src['i'] < 0.0:
                        sdss_i_arr[s] = np.nan
                    else:
                        sdss_i_arr[s] = sdss_src['i']
                    if sdss_src['err_i'] < 0.0:
                        sdss_ierr_arr[s] = np.nan
                    else:
                        sdss_ierr_arr[s] = sdss_src['err_i']
                    if sdss_src['z'] < 0.0:
                        sdss_z_arr[s] = np.nan
                    else:
                        sdss_z_arr[s] = sdss_src['z']
                    if sdss_src['err_z'] < 0.0:
                        sdss_zerr_arr[s] = np.nan
                    else:
                        sdss_zerr_arr[s] = sdss_src['err_z']
                    match_spec = SDSS.query_region(coord.SkyCoord(source['WISE_RA'], source['WISE_DEC'], unit=(u.deg, u.deg), frame='icrs'), photoobj_fields=['objid'], specobj_fields=['ra','dec','z','zerr'], spectro=True, radius=3.2 * u.arcsec, data_release=16)
                    if match_spec is not None: #if (len(np.array(match_spec).flatten()) != None): #if match_spec != None:
                        match_spec_crossid = match_spec[np.where(match_spec['objid'] == sdss_src['objid'])]
                        if len(match_spec_crossid) == 0:
                            sdss_rs_arr[s] = np.nan
                            sdss_rserr_arr[s] = np.nan
                        elif len(match_spec_crossid) == 1:
                            sdss_rs_arr[s] = match_spec_crossid['z']
                            sdss_rserr_arr[s] = match_spec_crossid['zerr']
                        else:
                            sdss_rs_arr[s] = match_spec_crossid['z'][0]
                            sdss_rserr_arr[s] = match_spec_crossid['zerr'][0]
                    else:
                        sdss_rs_arr[s] = np.nan
                        sdss_rserr_arr[s] = np.nan
                else:
                    objid_arr[s] = np.nan
                    sdss_ra_arr[s] = np.nan
                    sdss_dec_arr[s] = np.nan
                    sdss_u_arr[s] = np.nan
                    sdss_uerr_arr[s] = np.nan
                    sdss_g_arr[s] = np.nan
                    sdss_gerr_arr[s] = np.nan
                    sdss_r_arr[s] = np.nan
                    sdss_rerr_arr[s] = np.nan
                    sdss_i_arr[s] = np.nan
                    sdss_ierr_arr[s] = np.nan
                    sdss_z_arr[s] = np.nan
                    sdss_zerr_arr[s] = np.nan
                    sdss_rs_arr[s] = np.nan
                    sdss_rserr_arr[s] = np.nan
                    sdss_rs_arr[s] = np.nan
                    sdss_rserr_arr[s] = np.nan
            else:
                objid_arr[s] = np.nan
                sdss_ra_arr[s] = np.nan
                sdss_dec_arr[s] = np.nan
                sdss_u_arr[s] = np.nan
                sdss_uerr_arr[s] = np.nan
                sdss_g_arr[s] = np.nan
                sdss_gerr_arr[s] = np.nan
                sdss_r_arr[s] = np.nan
                sdss_rerr_arr[s] = np.nan
                sdss_i_arr[s] = np.nan
                sdss_ierr_arr[s] = np.nan
                sdss_z_arr[s] = np.nan
                sdss_zerr_arr[s] = np.nan
                sdss_rs_arr[s] = np.nan
                sdss_rserr_arr[s] = np.nan
                sdss_rs_arr[s] = np.nan
                sdss_rserr_arr[s] = np.nan
    cat['SDSS_ID'] = objid_arr
    cat['SDSS_RA'] = sdss_ra_arr
    cat['SDSS_DEC'] = sdss_dec_arr
    cat['SDSS_Flux_U'] = sdss_u_arr
    cat['SDSS_Flux_U_err'] = sdss_uerr_arr
    cat['SDSS_Flux_G'] = sdss_g_arr
    cat['SDSS_Flux_G_err'] = sdss_gerr_arr
    cat['SDSS_Flux_R'] = sdss_r_arr
    cat['SDSS_Flux_R_err'] = sdss_rerr_arr
    cat['SDSS_Flux_I'] = sdss_i_arr
    cat['SDSS_Flux_I_err'] = sdss_ierr_arr
    cat['SDSS_Flux_Z'] = sdss_z_arr
    cat['SDSS_Flux_Z_err'] = sdss_zerr_arr
    cat['SDSS_z'] = sdss_rs_arr
    cat['SDSS_z_err'] = sdss_rserr_arr
    return cat


def write_cross_matched_cat(self, cat):
    # Reorder the columns of the catalogue
    new_order = ['ID','RA','RA_err','DEC','DEC_err','PI_Isl','PI_Isl_err','PI_fit','PI_fit_err','S_Code','RA_Comp','RA_Comp_err','DEC_Comp','DEC_Comp_err','PI_Comp_Isl','PI_Comp_Isl_err','PI_Comp_fit','PI_Comp_fit_err','PI_Comp_Peak','PI_Comp_Peak_err','PI_rms','Isl_id','RM_Comp','RM_Comp_err','TP_ID','TP','TP_err','RA_TP','RA_TP_err','DEC_TP','DEC_TP_err','FP_Isl','FP_Isl_err','FP_fit','FP_fit_err','NVSS_I','NVSS_I_err','NVSS_PI','NVSS_PI_err','NVSS_FP','NVSS_FP_err','NVSS_RM','NVSS_RM_err','NVSS_RA','NVSS_RA_err','NVSS_DEC','NVSS_DEC_err','WISE_ID','WISE_RA','WISE_RA_err','WISE_DEC','WISE_DEC_err','WISE_Flux_3.4','WISE_Flux_3.4_err','WISE_SNR_3.4','WISE_Flux_4.6','WISE_Flux_4.6_err','WISE_SNR_4.6','WISE_Flux_12','WISE_Flux_12_err','WISE_SNR_12','WISE_Flux_22','WISE_Flux_22_err','WISE_SNR_22','SDSS_ID','SDSS_RA','SDSS_DEC','SDSS_Flux_U','SDSS_Flux_U_err','SDSS_Flux_G','SDSS_Flux_G_err','SDSS_Flux_R','SDSS_Flux_R_err','SDSS_Flux_I','SDSS_Flux_I_err','SDSS_Flux_Z','SDSS_Flux_Z_err','SDSS_z','SDSS_z_err']
    t_new_order = cat[new_order]
    # Write out the catalogue
    t_new_order.write(self.polanalysisdir + '/PI_cat_cm.txt', format='ascii', overwrite=True)
    # Remove the old header
    with open(self.polanalysisdir + '/PI_cat_cm.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open(self.polanalysisdir + '/PI_cat_final.txt', 'w') as fout:
        fout.writelines(data[4:])