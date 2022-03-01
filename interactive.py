import numpy as np
import os
from matplotlib.cbook import flatten

from astropy.table import Table
from astroquery.irsa import Irsa
from astroquery.sdss import SDSS
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord

import misc
import util
import make_plots


def load_catalogue_PI(self):
    """
    Interactive function to load the polarised intensity catalogue from the analysis into an astropy table
    returns (astropy table): Polarised intensity catalogues in astropy table format
    """
    PI_cat = Table.read(self.polanalysisdir + '/PI_cat_final.txt', format='ascii')
    return PI_cat


def load_catalogue_TP(self):
    """
    Interactive function to load the total power catalogue from the analysis into an astropy table
    returns (astropy table): The total power catalogue in astropy table format
    """
    TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final.txt', format='ascii')
    return TP_cat


def load_catalogues(self):
    """
    Load the total power and polarised intensity catalogues
    """
    PI_cat = Table.read(self.polanalysisdir + '/PI_cat_final.txt', format='ascii')
    TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final.txt', format='ascii')
    return TP_cat, PI_cat


def load_catalogue_PI_interactive(self):
    """
    Interactive function to load the polarised intensity catalogue from a previous run of the interactive scripts as an astropy table
    returns (astropy table): The polarised intensity catalogue in astropy table format
    """
    PI_cat = Table.read(self.polanalysisdir + '/PI_cat_final_interactive.txt', format='ascii')
    return PI_cat


def load_catalogue_TP_interactive(self):
    """
    Interactive function to load the total power catalogue from a previous run of the interactive scripts as an astropy table
    returns (astropy table): The total power catalogue in astropy table format
    """
    TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final_interactive.txt', format='ascii')
    return TP_cat


def load_catalogues_interactive(self):
    """
    Load the total power and polarised intensity catalogues from a previous run of the interactive scripts as astropy tables
    """
    PI_cat = Table.read(self.polanalysisdir + '/PI_cat_final_interactive.txt', format='ascii')
    TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final_interactive.txt', format='ascii')
    return TP_cat, PI_cat


def combine_PI(self, sourceids, tpcat, picat):
    """
    Interactive function to combine two polarised sources into one and search for the total power, WISE and SDSS counterpart at the new central position
    sourceids (list of strings): Source ids to combine
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    com_idxs = []
    for sourceid in sourceids:
        com_idxs.append(list(np.where(picat['ID'] == sourceid)))
    idxs_arr = np.array(list(flatten(com_idxs)))
    picat['RA'][idxs_arr] = np.mean(picat['RA_Comp'][idxs_arr])
    picat['RA_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['RA_Comp_err'][idxs_arr])))
    picat['DEC'][idxs_arr] = np.mean(picat['DEC_Comp'][idxs_arr])
    picat['DEC_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['DEC_Comp_err'][idxs_arr])))
    picat['PI'][idxs_arr] = np.sum(picat['PI_Comp'][idxs_arr])
    picat['PI_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['PI_Comp_err'][idxs_arr])))
    picat['PI_rms'][idxs_arr] = np.mean(picat['PI_rms'][idxs_arr])
    picat['S_Code'][idxs_arr] = 'E'
    src_name = misc.make_source_id(picat['RA'][idxs_arr][0], picat['DEC'][idxs_arr][0], self.prefix)
    picat['ID'][idxs_arr] = src_name
    print('New Source ID from combination of ' + ' and '.join(sourceids) + ' is: ' + src_name)
    # check for new total power counterpart
    ra, dec, pi, pi_err = picat['RA'][idxs_arr][0], picat['DEC'][idxs_arr][0], picat['PI'][idxs_arr][0], picat['PI_err'][idxs_arr][0]
    picat['TP_ID'][idxs_arr], picat['TP'][idxs_arr], picat['TP_err'][idxs_arr], picat['RA_TP'][idxs_arr], picat['RA_TP_err'][idxs_arr], picat['DEC_TP'][idxs_arr], picat['DEC_TP_err'][idxs_arr], picat['FP'][idxs_arr], picat['FP_err'][idxs_arr] = check_TP(self, tpcat, ra, dec, pi, pi_err)
    ra_tp, dec_tp = picat['RA_TP'][idxs_arr][0], picat['DEC_TP'][idxs_arr][0]
    # check for new wise counterpart
    picat['WISE_ID'][idxs_arr], picat['WISE_RA'][idxs_arr], picat['WISE_RA_err'][idxs_arr], picat['WISE_DEC'][idxs_arr], picat['WISE_DEC_err'][idxs_arr], picat['WISE_Flux_3.4'][idxs_arr], picat['WISE_Flux_3.4_err'][idxs_arr], picat['WISE_SNR_3.4'][idxs_arr], picat['WISE_Flux_4.6'][idxs_arr], picat['WISE_Flux_4.6_err'][idxs_arr], picat['WISE_SNR_4.6'][idxs_arr], picat['WISE_Flux_12'][idxs_arr], picat['WISE_Flux_12_err'][idxs_arr], picat['WISE_SNR_12'][idxs_arr], picat['WISE_Flux_22'][idxs_arr], picat['WISE_Flux_22_err'][idxs_arr], picat['WISE_SNR_22'][idxs_arr] = check_wise(self, ra_tp, dec_tp)
    # check the SDSS counterpart
    ra_wise, dec_wise = picat['WISE_RA'][idxs_arr][0], picat['WISE_DEC'][idxs_arr][0]
    picat['SDSS_ID'][idxs_arr], picat['SDSS_RA'][idxs_arr], picat['SDSS_DEC'][idxs_arr], picat['SDSS_Flux_U'][idxs_arr], picat['SDSS_Flux_U_err'][idxs_arr], picat['SDSS_Flux_G'][idxs_arr], picat['SDSS_Flux_G_err'][idxs_arr], picat['SDSS_Flux_R'][idxs_arr], picat['SDSS_Flux_R_err'][idxs_arr], picat['SDSS_Flux_I'][idxs_arr], picat['SDSS_Flux_I_err'][idxs_arr], picat['SDSS_Flux_Z'][idxs_arr], picat['SDSS_Flux_Z_err'][idxs_arr], picat['SDSS_z'][idxs_arr], picat['SDSS_z_err'][idxs_arr] = check_sdss(self, ra_wise, dec_wise)
    return tpcat, picat


def combine_TP(self, tpids, piid, tpcat, picat, ra_hms=None, dec_hms=None):
    """
    Interactive function to combine total power sources and connect the combines source to a PI-id. If ra_hms and dec_hms are given WISE and SDSS counterparts aare seached at this position, otherwise the new combined TP central position is used for the search
    tpids (list of strings): Total power source ids to combine (start with "TP_")
    piid (string): PI-id to connect the sources to
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    ra_hms: Right Ascension in dd:mm:ss format
    dec_hms: Declination in hh:mm:ss format
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    # Combine the total power source and update the total power catalogue
    if len(tpids) > 1:
        com_idxs = []
        for tpid in tpids:
            com_idxs.append(list(np.where(tpcat['TP_ID'] == tpid)))
        idxs_arr = np.array(list(flatten(com_idxs)))
    else:
        com_idxs = np.where(tpcat['TP_ID'] == tpids[0])
        idxs_arr = np.array(list(flatten(com_idxs)))
    tpcat['RA_TP'][idxs_arr] = np.mean(tpcat['RA_Comp_TP'][idxs_arr])
    tpcat['RA_TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(tpcat['RA_Comp_TP_err'][idxs_arr])))
    tpcat['DEC_TP'][idxs_arr] = np.mean(tpcat['DEC_Comp_TP'][idxs_arr])
    tpcat['DEC_TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(tpcat['DEC_Comp_TP_err'][idxs_arr])))
    tpcat['TP'][idxs_arr] = np.sum(tpcat['TP_Comp'][idxs_arr])
    tpcat['TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(tpcat['TP_Comp_err'][idxs_arr])))
    tpcat['S_Code'][idxs_arr] = 'E'
    src_name = misc.make_source_id(tpcat['RA_TP'][idxs_arr][0], tpcat['DEC_TP'][idxs_arr][0], 'TP')
    tpcat['TP_ID'][idxs_arr] = src_name
    # Find the given source id in the PI catalogue and enter the new TP parameters
    pi_idxs = np.where(picat['ID'] == piid)
    picat['TP_ID'][pi_idxs] = tpcat['TP_ID'][idxs_arr][0]
    picat['TP'][pi_idxs] = tpcat['TP'][idxs_arr][0]
    picat['TP_err'][pi_idxs] = tpcat['TP_err'][idxs_arr][0]
    picat['RA_TP'][pi_idxs] = tpcat['RA_TP'][idxs_arr][0]
    picat['RA_TP_err'][pi_idxs] =  tpcat['RA_TP_err'][idxs_arr][0]
    picat['DEC_TP'][pi_idxs] = tpcat['DEC_TP'][idxs_arr][0]
    picat['DEC_TP_err'][pi_idxs] = tpcat['DEC_TP_err'][idxs_arr][0]
    picat['FP'][pi_idxs] = picat['PI'][pi_idxs][0] / tpcat['TP'][idxs_arr][0]
    picat['FP_err'][pi_idxs] = np.sqrt((1.0/tpcat['TP'][idxs_arr][0])**2.0 * picat['PI_err'][pi_idxs][0]**2.0 + (picat['PI'][pi_idxs][0] / (tpcat['TP'][idxs_arr][0]**2.0)) * tpcat['TP_err'][idxs_arr][0]**2.0)
    print('Total power sources ' + ' and '.join(tpids) + ' combined to ' + src_name + ' and connected to PI source ' + piid + '!')
    # Check for new WISE and SDSS counterparts
    if ra_hms == None and dec_hms == None:
        picat['WISE_ID'][pi_idxs], picat['WISE_RA'][pi_idxs], picat['WISE_RA_err'][pi_idxs], picat['WISE_DEC'][pi_idxs], picat['WISE_DEC_err'][pi_idxs], picat['WISE_Flux_3.4'][pi_idxs], picat['WISE_Flux_3.4_err'][pi_idxs], picat['WISE_SNR_3.4'][pi_idxs], picat['WISE_Flux_4.6'][pi_idxs], picat['WISE_Flux_4.6_err'][pi_idxs], picat['WISE_SNR_4.6'][pi_idxs], picat['WISE_Flux_12'][pi_idxs], picat['WISE_Flux_12_err'][pi_idxs], picat['WISE_SNR_12'][pi_idxs], picat['WISE_Flux_22'][pi_idxs], picat['WISE_Flux_22_err'][pi_idxs], picat['WISE_SNR_22'][pi_idxs] = check_wise(self, picat['RA_TP'][pi_idxs][0], picat['DEC_TP'][pi_idxs][0])
    else:
        cd = SkyCoord(ra_hms, dec_hms)
        ra_deg = cd.ra.deg
        dec_deg = cd.dec.deg
        picat['WISE_ID'][pi_idxs], picat['WISE_RA'][pi_idxs], picat['WISE_RA_err'][pi_idxs], picat['WISE_DEC'][pi_idxs], picat['WISE_DEC_err'][pi_idxs], picat['WISE_Flux_3.4'][pi_idxs], picat['WISE_Flux_3.4_err'][pi_idxs], picat['WISE_SNR_3.4'][pi_idxs], picat['WISE_Flux_4.6'][pi_idxs], picat['WISE_Flux_4.6_err'][pi_idxs], picat['WISE_SNR_4.6'][pi_idxs], picat['WISE_Flux_12'][pi_idxs], picat['WISE_Flux_12_err'][pi_idxs], picat['WISE_SNR_12'][pi_idxs], picat['WISE_Flux_22'][pi_idxs], picat['WISE_Flux_22_err'][pi_idxs], picat['WISE_SNR_22'][pi_idxs] = check_wise(self,ra_deg, dec_deg)
    ra_wise, dec_wise = picat['WISE_RA'][pi_idxs][0], picat['WISE_DEC'][pi_idxs][0]
    picat['SDSS_ID'][pi_idxs], picat['SDSS_RA'][pi_idxs], picat['SDSS_DEC'][pi_idxs], picat['SDSS_Flux_U'][pi_idxs], picat['SDSS_Flux_U_err'][pi_idxs], picat['SDSS_Flux_G'][pi_idxs], picat['SDSS_Flux_G_err'][pi_idxs], picat['SDSS_Flux_R'][pi_idxs], picat['SDSS_Flux_R_err'][pi_idxs], picat['SDSS_Flux_I'][pi_idxs], picat['SDSS_Flux_I_err'][pi_idxs], picat['SDSS_Flux_Z'][pi_idxs], picat['SDSS_Flux_Z_err'][pi_idxs], picat['SDSS_z'][pi_idxs], picat['SDSS_z_err'][pi_idxs] = check_sdss(self, ra_wise, dec_wise)
    return tpcat, picat


def combine_PI_TP(self, sourceids, tpcat, picat):
    """
    Interactive function to combine two total power counterparts into one and search for the WISE and SDSS counterpart at the new central position of the two total power counterparts
    sourceids (list of strings): Total power source ids to combine (start with "TP_")
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    com_idxs = []
    for sourceid in sourceids:
        com_idxs.append(list(np.where(picat['ID'] == sourceid)))
    idxs_arr = np.array(list(flatten(com_idxs)))
    picat['RA'][idxs_arr] = np.mean(picat['RA_Comp'][idxs_arr])
    picat['RA_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['RA_Comp_err'][idxs_arr])))
    picat['DEC'][idxs_arr] = np.mean(picat['DEC_Comp'][idxs_arr])
    picat['DEC_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['DEC_Comp_err'][idxs_arr])))
    picat['PI'][idxs_arr] = np.sum(picat['PI_Comp'][idxs_arr])
    picat['PI_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['PI_Comp_err'][idxs_arr])))
    picat['PI_rms'][idxs_arr] = np.mean(picat['PI_rms'][idxs_arr])
    src_name = misc.make_source_id(picat['RA'][idxs_arr][0], picat['DEC'][idxs_arr][0], self.prefix)
    picat['ID'][idxs_arr] = src_name
    picat['TP'][idxs_arr] = np.sum(picat['TP'][idxs_arr])
    picat['TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['TP_err'][idxs_arr])))
    picat['RA_TP'][idxs_arr] = np.mean(picat['RA_TP'][idxs_arr])
    picat['RA_TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['RA_TP_err'][idxs_arr])))
    picat['DEC_TP'][idxs_arr] = np.mean(picat['DEC_TP'][idxs_arr])
    picat['DEC_TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(picat['DEC_TP_err'][idxs_arr])))
    picat['FP'][idxs_arr] = picat['PI'][idxs_arr] / picat['TP'][idxs_arr]
    picat['FP_err'][idxs_arr] = np.sqrt((1.0/picat['TP'][idxs_arr]['TP'])**2.0 * picat['PI_err'][idxs_arr]**2.0 + (picat['PI'][idxs_arr] / (picat['TP'][idxs_arr]**2.0)) * picat['TP_err'][idxs_arr]**2.0)
    # Change the ID of the total power counterpart to the same in the PI catalogue and the same in the total power catalogue
    tp_id = picat['TP_ID'][idxs_arr][0]
    picat['TP_ID'][idxs_arr] = tp_id
    tp_ids = np.unique(tp_id)
    for id in tp_ids:
        tp_idx = np.where(tpcat['TP_ID'] == id)
        tpcat['TP_ID'][tp_idx] = tp_id
        tpcat['S_Code'][tp_idx] = 'T'
        tpcat['RA_TP'][tp_idx] = picat['RA_TP'][idxs_arr[0]]
        tpcat['RA_TP_err'][tp_idx] = picat['RA_TP_err'][idxs_arr[0]]
        tpcat['DEC_TP'][tp_idx] = picat['DEC_TP'][idxs_arr[0]]
        tpcat['DEC_TP_err'][tp_idx] = picat['DEC_TP_err'][idxs_arr[0]]
        tpcat['TP'][tp_idx] = picat['TP'][idxs_arr[0]]
        tpcat['TP_err'][tp_idx] = picat['TP_err'][idxs_arr[0]]
    # check for new wise counterpart
    picat['WISE_ID'][idxs_arr], picat['WISE_RA'][idxs_arr], picat['WISE_RA_err'][idxs_arr], picat['WISE_DEC'][idxs_arr], picat['WISE_DEC_err'][idxs_arr], picat['WISE_Flux_3.4'][idxs_arr], picat['WISE_Flux_3.4_err'][idxs_arr], picat['WISE_SNR_3.4'][idxs_arr], picat['WISE_Flux_4.6'][idxs_arr], picat['WISE_Flux_4.6_err'][idxs_arr], picat['WISE_SNR_4.6'][idxs_arr], picat['WISE_Flux_12'][idxs_arr], picat['WISE_Flux_12_err'][idxs_arr], picat['WISE_SNR_12'][idxs_arr], picat['WISE_Flux_22'][idxs_arr], picat['WISE_Flux_22_err'][idxs_arr], picat['WISE_SNR_22'][idxs_arr] = check_wise(self, ra_tp, dec_tp)
    # check the SDSS counterpart
    ra_wise, dec_wise = picat['WISE_RA'][idxs_arr][0], picat['WISE_DEC'][idxs_arr][0]
    picat['SDSS_ID'][idxs_arr], picat['SDSS_RA'][idxs_arr], picat['SDSS_DEC'][idxs_arr], picat['SDSS_Flux_U'][idxs_arr], picat['SDSS_Flux_U_err'][idxs_arr], picat['SDSS_Flux_G'][idxs_arr], picat['SDSS_Flux_G_err'][idxs_arr], picat['SDSS_Flux_R'][idxs_arr], picat['SDSS_Flux_R_err'][idxs_arr], picat['SDSS_Flux_I'][idxs_arr], picat['SDSS_Flux_I_err'][idxs_arr], picat['SDSS_Flux_Z'][idxs_arr], picat['SDSS_Flux_Z_err'][idxs_arr], picat['SDSS_z'][idxs_arr], picat['SDSS_z_err'][idxs_arr] = check_sdss(self, ra_wise, dec_wise)
    return tpcat, picat


def combine_TP_nocrossmatch(self, isl_ids, tpcat, picat):
    """
    Interactive function to combine two total power counterparts into one without any cross-matching with WISE and SDSS. Also deletes the cross-match to the polarised intensity, WISE and SDSS counterparts
    isl_ids (list of strings): Island ids in total power catalogue to combine
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    com_idxs = []
    for isl_id in isl_ids:
        com_idxs.append(list(np.where(tpcat['TP_ID'] == isl_id)))
        print(com_idxs)
    idxs_arr = np.array(list(flatten(com_idxs)))
    # Combine the source in the total power catalogue
    tpcat['RA_TP'][idxs_arr] = np.mean(tpcat['RA_Comp_TP'][idxs_arr])
    tpcat['RA_TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(tpcat['RA_Comp_TP_err'][idxs_arr])))
    tpcat['DEC_TP'][idxs_arr] = np.mean(tpcat['DEC_Comp_TP'][idxs_arr])
    tpcat['DEC_TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(tpcat['DEC_Comp_TP_err'][idxs_arr])))
    tpcat['TP'][idxs_arr] = np.sum(tpcat['TP_Comp'][idxs_arr])
    tpcat['TP_err'][idxs_arr] = np.sqrt(np.sum(np.square(tpcat['TP_Comp_err'][idxs_arr])))
    # Look for the cross-matched sources in the PI catalogue and remove their total power, WISE and SDSS counterparts
#    tp_id = picat['TP_ID'][idxs_arr][0]
#    picat['TP_ID'][idxs_arr] = tp_id
#    tp_ids = np.unique(tp_id)
    for id in isl_ids:
        tp_idx = np.where(picat['TP_ID'] == int(id))
        picat['TP'][tp_idx] = np.nan
        picat['TP_err'][tp_idx] = np.nan
        picat['RA_TP'][tp_idx] = np.nan
        picat['RA_TP_err'][tp_idx] = np.nan
        picat['DEC_TP'][tp_idx] = np.nan
        picat['DEC_TP_err'][tp_idx] = np.nan
        picat['FP'][tp_idx] = np.nan
        picat['FP_err'][tp_idx] = np.nan
        picat['WISE_ID'][tp_idx] = np.nan
        picat['WISE_RA'][tp_idx] = np.nan
        picat['WISE_RA_err'][tp_idx] = np.nan
        picat['WISE_DEC'][tp_idx] = np.nan
        picat['WISE_DEC_err'][tp_idx] = np.nan
        picat['WISE_Flux_3.4'][tp_idx] = np.nan
        picat['WISE_Flux_3.4_err'][tp_idx] = np.nan
        picat['WISE_SNR_3.4'][tp_idx] = np.nan
        picat['WISE_Flux_4.6'][tp_idx] = np.nan
        picat['WISE_Flux_4.6_err'][tp_idx] = np.nan
        picat['WISE_SNR_4.6'][tp_idx] = np.nan
        picat['WISE_Flux_12'][tp_idx] = np.nan
        picat['WISE_Flux_12_err'][tp_idx] = np.nan
        picat['WISE_SNR_12'][tp_idx] = np.nan
        picat['WISE_Flux_22'][tp_idx] = np.nan
        picat['WISE_Flux_22_err'][tp_idx] = np.nan
        picat['WISE_SNR_22'][tp_idx] = np.nan
        picat['SDSS_ID'][tp_idx] = np.nan
        picat['SDSS_RA'][tp_idx] = np.nan
        picat['SDSS_DEC'][tp_idx] = np.nan
        picat['SDSS_Flux_U'][tp_idx] = np.nan
        picat['SDSS_Flux_U_err'][tp_idx] = np.nan
        picat['SDSS_Flux_G'][tp_idx] = np.nan
        picat['SDSS_Flux_G_err'][tp_idx] = np.nan
        picat['SDSS_Flux_R'][tp_idx] = np.nan
        picat['SDSS_Flux_R_err'][tp_idx] = np.nan
        picat['SDSS_Flux_I'][tp_idx] = np.nan
        picat['SDSS_Flux_I_err'][tp_idx] = np.nan
        picat['SDSS_Flux_Z'][tp_idx] = np.nan
        picat['SDSS_Flux_Z_err'][tp_idx] = np.nan
        picat['SDSS_z'][tp_idx] = np.nan
        picat['SDSS_z_err'][tp_idx] = np.nan
    tpcat['TP_ID'][idxs_arr] = tpcat['TP_ID'][idxs_arr[0]]
    return tpcat, picat


def split_PI(self, sourceid, tpcat, picat):
    """
    Splits a polarised multi-component source into seperate sources (one for each component) and checks for total power, WISE and SDSS counterparts for each of them
    sourceids (string): Source id to split
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    split_idxs = np.where(picat['ID'] == sourceid)
    picat['RA'][split_idxs] = picat['RA_Comp'][split_idxs]
    picat['RA_err'][split_idxs] = picat['RA_Comp_err'][split_idxs]
    picat['DEC'][split_idxs] = picat['DEC_Comp'][split_idxs]
    picat['DEC_err'][split_idxs] = picat['DEC_Comp_err'][split_idxs]
    picat['PI'][split_idxs] = picat['PI_Comp'][split_idxs]
    picat['PI_err'][split_idxs] = picat['PI_Comp_err'][split_idxs]
    picat['PI_rms'][split_idxs] = picat['PI_rms'][split_idxs]
    picat['S_Code'][split_idxs] = 'S'
    print('Splitting source ' + sourceid + '!')
    for s in split_idxs[0]:
        src_name = misc.make_source_id(picat['RA'][s], picat['DEC'][s], self.prefix)
        picat['ID'][s] = src_name
        print('New source ID ' + src_name + ' generated!')
        picat['TP_ID'][s], picat['TP'][s], picat['TP_err'][s], picat['RA_TP'][s], picat['RA_TP_err'][s], picat['DEC_TP'][s], picat['DEC_TP_err'][s], picat['FP'][s], picat['FP_err'][s] = check_TP(self, tpcat, picat['RA'][s], picat['DEC'][s], picat['PI'][s], picat['PI_err'][s])
        picat['WISE_ID'][s], picat['WISE_RA'][s], picat['WISE_RA_err'][s], picat['WISE_DEC'][s], picat['WISE_DEC_err'][s], picat['WISE_Flux_3.4'][s], picat['WISE_Flux_3.4_err'][s], picat['WISE_SNR_3.4'][s], picat['WISE_Flux_4.6'][s], picat['WISE_Flux_4.6_err'][s], picat['WISE_SNR_4.6'][s], picat['WISE_Flux_12'][s], picat['WISE_Flux_12_err'][s], picat['WISE_SNR_12'][s], picat['WISE_Flux_22'][s], picat['WISE_Flux_22_err'][s], picat['WISE_SNR_22'][s] = check_wise(self, picat['RA_TP'][s], picat['DEC_TP'][s])
        picat['SDSS_ID'][s], picat['SDSS_RA'][s], picat['SDSS_DEC'][s], picat['SDSS_Flux_U'][s], picat['SDSS_Flux_U_err'][s], picat['SDSS_Flux_G'][s], picat['SDSS_Flux_G_err'][s], picat['SDSS_Flux_R'][s], picat['SDSS_Flux_R_err'][s], picat['SDSS_Flux_I'][s], picat['SDSS_Flux_I_err'][s], picat['SDSS_Flux_Z'][s], picat['SDSS_Flux_Z_err'][s], picat['SDSS_z'][s], picat['SDSS_z_err'][s] = check_sdss(self, picat['WISE_RA'][s], picat['WISE_DEC'][s])
    return tpcat, picat


def split_TP(self, isl_id, tpcat, picat):
    """
    Splits a total power multi-component source into seperate sources (one for each component) and does a new crossmatch for each component for WSIE and SDSS
    isl_id (string): Source ID in the total power catalogue to split into components (starts with "TP_")
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    # Change the entries in the total power catalogue
    split_idxs = np.where(tpcat['TP_ID'] == isl_id)
    tpcat['RA_TP'][split_idxs] = tpcat['RA_Comp_TP'][split_idxs]
    tpcat['RA_TP_err'][split_idxs] = tpcat['RA_Comp_TP_err'][split_idxs]
    tpcat['DEC_TP'][split_idxs] = tpcat['DEC_Comp_TP'][split_idxs]
    tpcat['DEC_TP_err'][split_idxs] = tpcat['DEC_Comp_TP_err'][split_idxs]
    tpcat['TP'][split_idxs] = tpcat['TP_Comp'][split_idxs]
    tpcat['TP_err'][split_idxs] = tpcat['TP_Comp_err'][split_idxs]
    print('Splitting source ' + isl_id + '!')
    for s in split_idxs[0]:
        tp_src_name = misc.make_source_id(tpcat['RA_TP'][s], tpcat['DEC_TP'][s], 'TP')
        print('New total power source ID ' + tp_src_name + ' generated!')
        tpcat['TP_ID'][s] = tp_src_name
    # Delete the entries for polarised sources using the total power id and search for the individual components for new Total power, WISE and SDSS counterparts
    chk_idxs = np.where(picat['TP_ID'] == isl_id)
    for p in chk_idxs[0]:
        # check for new total power counterpart
        picat['TP_ID'][p], picat['TP'][p], picat['TP_err'][p], picat['RA_TP'][p], picat['RA_TP_err'][p], picat['DEC_TP'][p], picat['DEC_TP_err'][p], picat['FP'][p], picat['FP_err'][p] = check_TP(self, tpcat, picat['RA'][p], picat['DEC'][p], picat['PI'][p], picat['PI_err'][p])
        # check for new WISE counterpart
        picat['WISE_ID'][p], picat['WISE_RA'][p], picat['WISE_RA_err'][p], picat['WISE_DEC'][p], picat['WISE_DEC_err'][p], picat['WISE_Flux_3.4'][p], picat['WISE_Flux_3.4_err'][p], picat['WISE_SNR_3.4'][p], picat['WISE_Flux_4.6'][p], picat['WISE_Flux_4.6_err'][p], picat['WISE_SNR_4.6'][p], picat['WISE_Flux_12'][p], picat['WISE_Flux_12_err'][p], picat['WISE_SNR_12'][p], picat['WISE_Flux_22'][p], picat['WISE_Flux_22_err'][p], picat['WISE_SNR_22'][p] = check_wise(self, picat['RA_TP'][p], picat['DEC_TP'][p])
        # check for new SDSS counterpart
        picat['SDSS_ID'][p], picat['SDSS_RA'][p], picat['SDSS_DEC'][p], picat['SDSS_Flux_U'][p], picat['SDSS_Flux_U_err'][p], picat['SDSS_Flux_G'][p], picat['SDSS_Flux_G_err'][p], picat['SDSS_Flux_R'][p], picat['SDSS_Flux_R_err'][p], picat['SDSS_Flux_I'][p], picat['SDSS_Flux_I_err'][p], picat['SDSS_Flux_Z'][p], picat['SDSS_Flux_Z_err'][p], picat['SDSS_z'][p], picat['SDSS_z_err'][p] = check_sdss(self, picat['WISE_RA'][p], picat['WISE_DEC'][p])
    return tpcat, picat


def crossmatch_TP(self, sourceid, ra_hms, dec_hms, tpcat, picat):
    """
    Search for a total power counterpart at a given position
    sourceid: Source id to cross match
    ra_hms: Right Ascension in dd:mm:ss format
    dec_hms: Declination in hh:mm:ss format
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    chk_idxs = np.where(picat['ID'] == sourceid)
    if len(chk_idxs[0]) == 0:
        print('Source with ID ' + sourceid + ' not found!')
    else:
        # check for new total power counterpart
        print('Looking for total power counterpart for source ' + sourceid + '!')
        cd = SkyCoord(ra_hms, dec_hms)
        ra_deg = cd.ra.deg
        dec_deg = cd.dec.deg
        ra, dec, pi, pi_err = ra_deg, dec_deg, picat['PI'][chk_idxs][0], picat['PI_err'][chk_idxs][0]
        # check for new total power counterpart
        picat['TP_ID'][chk_idxs], picat['TP'][chk_idxs], picat['TP_err'][chk_idxs], picat['RA_TP'][chk_idxs], picat['RA_TP_err'][chk_idxs], picat['DEC_TP'][chk_idxs], picat['DEC_TP_err'][chk_idxs], picat['FP'][chk_idxs], picat['FP_err'][chk_idxs] = check_TP(self, tpcat, ra, dec, pi, pi_err)
        ra_tp, dec_tp = picat['RA_TP'][chk_idxs][0], picat['DEC_TP'][chk_idxs][0]
        # check for new wise counterpart
        picat['WISE_ID'][chk_idxs], picat['WISE_RA'][chk_idxs], picat['WISE_RA_err'][chk_idxs], picat['WISE_DEC'][chk_idxs], picat['WISE_DEC_err'][chk_idxs], picat['WISE_Flux_3.4'][chk_idxs], picat['WISE_Flux_3.4_err'][chk_idxs], picat['WISE_SNR_3.4'][chk_idxs], picat['WISE_Flux_4.6'][chk_idxs], picat['WISE_Flux_4.6_err'][chk_idxs], picat['WISE_SNR_4.6'][chk_idxs], picat['WISE_Flux_12'][chk_idxs], picat['WISE_Flux_12_err'][chk_idxs], picat['WISE_SNR_12'][chk_idxs], picat['WISE_Flux_22'][chk_idxs], picat['WISE_Flux_22_err'][chk_idxs], picat['WISE_SNR_22'][chk_idxs] = check_wise(self, ra_tp, dec_tp)
        # check the SDSS counterpart
        ra_wise, dec_wise = picat['WISE_RA'][chk_idxs][0], picat['WISE_DEC'][chk_idxs][0]
        picat['SDSS_ID'][chk_idxs], picat['SDSS_RA'][chk_idxs], picat['SDSS_DEC'][chk_idxs], picat['SDSS_Flux_U'][chk_idxs], picat['SDSS_Flux_U_err'][chk_idxs], picat['SDSS_Flux_G'][chk_idxs], picat['SDSS_Flux_G_err'][chk_idxs], picat['SDSS_Flux_R'][chk_idxs], picat['SDSS_Flux_R_err'][chk_idxs], picat['SDSS_Flux_I'][chk_idxs], picat['SDSS_Flux_I_err'][chk_idxs], picat['SDSS_Flux_Z'][chk_idxs], picat['SDSS_Flux_Z_err'][chk_idxs], picat['SDSS_z'][chk_idxs], picat['SDSS_z_err'][chk_idxs] = check_sdss(self, ra_wise, dec_wise)
        return tpcat, picat


def crossmatch_wise(self, sourceid, ra_hms, dec_hms, tpcat, picat):
    """
    Search for a WISE counterpart at a given position
    sourceid: Source id to cross match
    ra_hms: Right Ascension in dd:mm:ss format
    dec_hms: Declination in hh:mm:ss format
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    chk_idxs = np.where(picat['ID'] == sourceid)
    if len(chk_idxs[0]) == 0:
        print('Source with ID ' + sourceid + ' not found!')
    else:
        # check for new wise counterpart
        print('Cross-matching source ' + sourceid + ' with WISE!')
        cd = SkyCoord(ra_hms, dec_hms)
        ra_deg = cd.ra.deg
        dec_deg = cd.dec.deg
        picat['WISE_ID'][chk_idxs], picat['WISE_RA'][chk_idxs], picat['WISE_RA_err'][chk_idxs], picat['WISE_DEC'][chk_idxs], picat['WISE_DEC_err'][chk_idxs], picat['WISE_Flux_3.4'][chk_idxs], picat['WISE_Flux_3.4_err'][chk_idxs], picat['WISE_SNR_3.4'][chk_idxs], picat['WISE_Flux_4.6'][chk_idxs], picat['WISE_Flux_4.6_err'][chk_idxs], picat['WISE_SNR_4.6'][chk_idxs], picat['WISE_Flux_12'][chk_idxs], picat['WISE_Flux_12_err'][chk_idxs], picat['WISE_SNR_12'][chk_idxs], picat['WISE_Flux_22'][chk_idxs], picat['WISE_Flux_22_err'][chk_idxs], picat['WISE_SNR_22'][chk_idxs] = check_wise(self, ra_deg, dec_deg)
        # check the SDSS counterpart
        ra_wise, dec_wise = picat['WISE_RA'][chk_idxs][0], picat['WISE_DEC'][chk_idxs][0]
        picat['SDSS_ID'][chk_idxs], picat['SDSS_RA'][chk_idxs], picat['SDSS_DEC'][chk_idxs], picat['SDSS_Flux_U'][chk_idxs], picat['SDSS_Flux_U_err'][chk_idxs], picat['SDSS_Flux_G'][chk_idxs], picat['SDSS_Flux_G_err'][chk_idxs], picat['SDSS_Flux_R'][chk_idxs], picat['SDSS_Flux_R_err'][chk_idxs], picat['SDSS_Flux_I'][chk_idxs], picat['SDSS_Flux_I_err'][chk_idxs], picat['SDSS_Flux_Z'][chk_idxs], picat['SDSS_Flux_Z_err'][chk_idxs], picat['SDSS_z'][chk_idxs], picat['SDSS_z_err'][chk_idxs] = check_sdss(self, ra_wise, dec_wise)
        return tpcat, picat


def crossmatch_sdss(self, sourceid, ra_hms, dec_hms, tpcat, picat):
    """
    Search for an SDSS counterpart at a given position
    sourceid: Source id to cross match
    ra_hms: Right Ascension in dd:mm:ss format
    dec_hms: Declination in hh:mm:ss format
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    chk_idxs = np.where(picat['ID'] == sourceid)
    if len(chk_idxs[0]) == 0:
        print('Source with ID ' + sourceid + ' not found!')
    else:
        # check for a new SDSS counterpart
        print('Cross-matching source ' + sourceid + ' with SDSS!')
        cd = SkyCoord(ra_hms, dec_hms)
        ra_deg = cd.ra.deg
        dec_deg = cd.dec.deg
        picat['SDSS_ID'][chk_idxs], picat['SDSS_RA'][chk_idxs], picat['SDSS_DEC'][chk_idxs], picat['SDSS_Flux_U'][chk_idxs], picat['SDSS_Flux_U_err'][chk_idxs], picat['SDSS_Flux_G'][chk_idxs], picat['SDSS_Flux_G_err'][chk_idxs], picat['SDSS_Flux_R'][chk_idxs], picat['SDSS_Flux_R_err'][chk_idxs], picat['SDSS_Flux_I'][chk_idxs], picat['SDSS_Flux_I_err'][chk_idxs], picat['SDSS_Flux_Z'][chk_idxs], picat['SDSS_Flux_Z_err'][chk_idxs], picat['SDSS_z'][chk_idxs], picat['SDSS_z_err'][chk_idxs] = check_sdss(self, ra_deg, dec_deg)
        return tpcat, picat


def remove_PI(self, sourceid, tpcat, picat):
    """
    Interactive function to remove a polarised source from the catalogue
    sourceid (string): Source id to remove
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    rem_idxs = np.where(picat['ID'] == sourceid)
    if len(rem_idxs[0]) == 0:
        print('Source with ID ' + sourceid + ' not found!')
    else:
        picat.remove_rows(rem_idxs[0])
        print('Source with ID ' + sourceid + ' and all its components are removed from the catalogue!')
    return tpcat, picat


def remove_wise(self, sourceid, tpcat, picat):
    """
    Interactive function to remove a WISE counterpart from the catalogue
    sourceid (string): Source id to remove
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    rem_idxs = np.where(picat['ID'] == sourceid)
    if len(rem_idxs[0]) == 0:
        print('Source with ID ' + sourceid + ' not found!')
    else:
        print('WISE and SDSS cross-match for source with ID ' + sourceid + ' removed from the catalogue!')
        picat['WISE_ID'][rem_idxs] = np.nan
        picat['WISE_RA'][rem_idxs] = np.nan
        picat['WISE_RA_err'][rem_idxs] = np.nan
        picat['WISE_DEC'][rem_idxs] = np.nan
        picat['WISE_DEC_err'][rem_idxs] = np.nan
        picat['WISE_Flux_3.4'][rem_idxs] = np.nan
        picat['WISE_Flux_3.4_err'][rem_idxs] = np.nan
        picat['WISE_SNR_3.4'][rem_idxs] = np.nan
        picat['WISE_Flux_4.6'][rem_idxs] = np.nan
        picat['WISE_Flux_4.6_err'][rem_idxs] = np.nan
        picat['WISE_SNR_4.6'][rem_idxs] = np.nan
        picat['WISE_Flux_12'][rem_idxs] = np.nan
        picat['WISE_Flux_12_err'][rem_idxs] = np.nan
        picat['WISE_SNR_12'][rem_idxs] = np.nan
        picat['WISE_Flux_22'][rem_idxs] = np.nan
        picat['WISE_Flux_22_err'][rem_idxs] = np.nan
        picat['WISE_SNR_22'][rem_idxs] = np.nan
        picat['SDSS_ID'][rem_idxs] = np.nan
        picat['SDSS_RA'][rem_idxs] = np.nan
        picat['SDSS_DEC'][rem_idxs] = np.nan
        picat['SDSS_Flux_U'][rem_idxs] = np.nan
        picat['SDSS_Flux_U_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_G'][rem_idxs] = np.nan
        picat['SDSS_Flux_G_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_R'][rem_idxs] = np.nan
        picat['SDSS_Flux_R_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_I'][rem_idxs] = np.nan
        picat['SDSS_Flux_I_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_Z'][rem_idxs] = np.nan
        picat['SDSS_Flux_Z_err'][rem_idxs] = np.nan
        picat['SDSS_z'][rem_idxs] = np.nan
        picat['SDSS_z_err'][rem_idxs] = np.nan
    return tpcat, picat


def remove_sdss(self, sourceid, tpcat, picat):
    """
    Interactive function to remove a SDSS counterpart from the catalogue
    sourceid (string): Source id to remove
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    rem_idxs = np.where(picat['ID'] == sourceid)
    if len(rem_idxs[0]) == 0:
        print('Source with ID ' + sourceid + ' not found!')
    else:
        print('SDSS cross-match for source with ID ' + sourceid + ' removed from the catalogue!')
        picat['SDSS_ID'][rem_idxs] = np.nan
        picat['SDSS_RA'][rem_idxs] = np.nan
        picat['SDSS_DEC'][rem_idxs] = np.nan
        picat['SDSS_Flux_U'][rem_idxs] = np.nan
        picat['SDSS_Flux_U_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_G'][rem_idxs] = np.nan
        picat['SDSS_Flux_G_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_R'][rem_idxs] = np.nan
        picat['SDSS_Flux_R_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_I'][rem_idxs] = np.nan
        picat['SDSS_Flux_I_err'][rem_idxs] = np.nan
        picat['SDSS_Flux_Z'][rem_idxs] = np.nan
        picat['SDSS_Flux_Z_err'][rem_idxs] = np.nan
        picat['SDSS_z'][rem_idxs] = np.nan
        picat['SDSS_z_err'][rem_idxs] = np.nan
    return tpcat, picat


def change_scode(self, sourceid, scode, tpcat, picat):
    """
    Interactive function to change the S-Code of a source manually
    sourceid (string): Source id to remove
    scode(string): New S-Code for source
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    spos = np.where(picat['ID'] == sourceid)
    picat['S_Code'][spos] = scode
    return tpcat, picat


def change_scode_S(self, sourceid, tpcat, picat):
    """
    Interactive function to change the S-Code of a source
    sourceid (string): Source id to edit
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    spos = np.where(picat['ID'] == sourceid)
    picat['S_Code'][spos] = 'S'
    return tpcat, picat


def change_scode_E(self, sourceid, tpcat, picat):
    """
    Interactive function to change the S-Code of a source
    sourceid (string): Source id to edit
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    spos = np.where(picat['ID'] == sourceid)
    picat['S_Code'][spos] = 'E'
    return tpcat, picat


def change_scode_C(self, sourceid, tpcat, picat):
    """
    Interactive function to change the S-Code of a source
    sourceid (string): Source id to edit
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    returns(astropy tables): Updated total power and polarised source catalogues
    """
    spos = np.where(picat['ID'] == sourceid)
    picat['S_Code'][spos] = 'C'
    return tpcat, picat


def write_interactive_catalogues(self, tpcat, picat):
    """
    Interactive function to write the new catalogues to file
    tpcat(astropy table): Total power source catalogue
    picat(astropy table): Polarised source catalogue
    """
    tpcat.write(self.polanalysisdir + '/TP_cat_final_interactive.txt', format='ascii')
    picat.write(self.polanalysisdir + '/PI_cat_final_interactive.txt', format='ascii')


def redo_single_plot(self, source_id, interactive):
    """
    Interactive function to redo a single plot given a source id
    source_id(string): Polarised source id
    interactive(Bool): Use the interactive catalogue or not
    """
    os.system('rm -rf ' + self.polanalysisplotdir + '/catalogue.pdf')
    make_plots.make_pdfs(self, id=source_id, interactive=interactive)
    make_plots.merge_pdfs(self)


def redo_plots(self):
    """
    Interactive function to redo the plots
    """
    os.system('rm -rf ' + self.polanalysisplotdir + '/*.pdf')
    make_plots.make_pdfs(self, interactive=True)
    make_plots.merge_pdfs(self)


def final_plots(self):
    """
    Interactive function to generate the final plots without all the annotations
    """
    os.system('rm -rf ' + self.polanalysisplotdir + '/*.pdf')
    make_plots.make_pdfs(self, interactive=True, final=True)
    make_plots.merge_pdfs(self)


###########################
##### Helper funtions #####
###########################


def check_TP(self, tpcat, ra, dec, pi, pi_err):
    dist_arr = np.sqrt(np.square(tpcat['RA_TP']-ra) + np.square(tpcat['DEC_TP']-dec))
    min_idx = np.argmin(np.abs(dist_arr))
    # Calculate maximum distance for a match
    bmaj, bmin = util.get_beam(self)
    dist = (np.max([bmaj*2.0, bmin*2.0]))
    if dist_arr[min_idx] <= dist:
        TP_id = tpcat[min_idx]['TP_ID']
        TP = tpcat[min_idx]['TP']
        TP_err = tpcat[min_idx]['TP_err']
        TP_ra = tpcat[min_idx]['RA_TP']
        TP_ra_err = tpcat[min_idx]['RA_TP_err']
        TP_dec = tpcat[min_idx]['DEC_TP']
        TP_dec_err = tpcat[min_idx]['DEC_TP_err']
        FP = pi/TP
        FP_err = np.sqrt((1.0/TP)**2.0 * pi_err**2.0 + (pi / (TP**2.0)) * TP_err**2.0)
        print('Found total power counterpart at RA=' + str(np.around(TP_ra, decimals=3)) + ' deg and DEC=' + str(np.around(TP_dec, decimals=3)) + ' deg!')
    else:
        TP_id = np.nan
        TP = np.nan
        TP_err = np.nan
        TP_ra = np.nan
        TP_ra_err = np.nan
        TP_dec = np.nan
        TP_dec_err = np.nan
        FP = np.nan
        FP_err = np.nan
        print('No total power counterpart found at new coordinates!')
    return TP_id, TP, TP_err, TP_ra, TP_ra_err, TP_dec, TP_dec_err, FP, FP_err


def check_wise(self, ra, dec):
    # Calculate maximum distance for a match
    bmaj, bmin = util.get_beam(self)
    dist = (np.max([bmaj, bmin]) / 2.0) * 3600.0
    # Cross match WISE sources with Apertif source catalogue
    if np.isnan(ra):
        wise_id = np.nan
        wise_ra = np.nan
        wise_ra_err = np.nan
        wise_dec = np.nan
        wise_dec_err = np.nan
        wise_3_4 = np.nan
        wise_3_4_err = np.nan
        wise_3_4_snr = np.nan
        wise_4_6 = np.nan
        wise_4_6_err = np.nan
        wise_4_6_snr = np.nan
        wise_12 = np.nan
        wise_12_err = np.nan
        wise_12_snr = np.nan
        wise_22 = np.nan
        wise_22_err = np.nan
        wise_22_snr = np.nan
    else:
        match = Irsa.query_region(coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), catalog='allwise_p3as_psd', radius=dist * u.arcsec)
        if len(match) == 0:
            wise_id = np.nan
            wise_ra = np.nan
            wise_ra_err = np.nan
            wise_dec = np.nan
            wise_dec_err = np.nan
            wise_3_4 = np.nan
            wise_3_4_err = np.nan
            wise_3_4_snr = np.nan
            wise_4_6 = np.nan
            wise_4_6_err = np.nan
            wise_4_6_snr = np.nan
            wise_12 = np.nan
            wise_12_err = np.nan
            wise_12_snr = np.nan
            wise_22 = np.nan
            wise_22_err = np.nan
            wise_22_snr = np.nan
            print('No WISE counterpart found!')
        else:
            wise_id = match['designation'][0]
            wise_ra = match['ra'][0]
            wise_ra_err = match['sigra'][0]
            wise_dec = match['dec'][0]
            wise_dec_err = match['sigdec'][0]
            wise_3_4 = match['w1mpro'][0]
            wise_3_4_err = match['w1sigmpro'][0]
            wise_3_4_snr = match['w1snr'][0]
            wise_4_6 = match['w2mpro'][0]
            wise_4_6_err = match['w2sigmpro'][0]
            wise_4_6_snr = match['w2snr'][0]
            wise_12 = match['w3mpro'][0]
            wise_12_err = match['w3sigmpro'][0]
            wise_12_snr = match['w3snr'][0]
            wise_22 = match['w4mpro'][0]
            wise_22_err = match['w4sigmpro'][0]
            wise_22_snr = match['w4snr'][0]
            print('Found WISE counterpart at RA=' + str(np.around(wise_ra, decimals=3)) + ' deg and DEC=' + str(np.around(wise_dec, decimals=3)) + ' deg!')
    return wise_id, wise_ra, wise_ra_err, wise_dec, wise_dec_err, wise_3_4, wise_3_4_err, wise_3_4_snr, wise_4_6, wise_4_6_err, wise_4_6_snr, wise_12, wise_12_err, wise_12_snr, wise_22, wise_22_err, wise_22_snr


def check_sdss(self, ra, dec):
    # Cross match SDSS sources with WISE coordinates of Apertif polarised source catalogue
    if np.isnan(ra):
        sdss_id = np.nan
        sdss_ra = np.nan
        sdss_dec = np.nan
        sdss_u = np.nan
        sdss_uerr = np.nan
        sdss_g = np.nan
        sdss_gerr = np.nan
        sdss_r = np.nan
        sdss_rerr = np.nan
        sdss_i = np.nan
        sdss_ierr = np.nan
        sdss_z = np.nan
        sdss_zerr = np.nan
        sdss_rs = np.nan
        sdss_rserr = np.nan
        print('No SDSS counterpart found!')
    else:
        match = SDSS.query_region(coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), photoobj_fields=['objid', 'ra', 'dec', 'u', 'err_u', 'g', 'err_g', 'r', 'err_r', 'i', 'err_i', 'z', 'err_z', 'type'], spectro=False, radius=3.2 * u.arcsec, data_release=16)
        if match != None:
            match_gal = match[np.where(match['type'] == 3)]
            if len(match_gal) != 0:
                dist = np.sqrt(np.square(match_gal['ra']-ra) + np.square(match_gal['dec']-dec))
                minidx = np.argmin(dist)
                sdss_src = match_gal[minidx]
                sdss_id = sdss_src['objid']
                sdss_ra = sdss_src['ra']
                sdss_dec = sdss_src['dec']
                sdss_u = sdss_src['u']
                sdss_uerr = sdss_src['err_u']
                sdss_g = sdss_src['g']
                sdss_gerr = sdss_src['err_g']
                sdss_r = sdss_src['r']
                sdss_rerr = sdss_src['err_r']
                sdss_i = sdss_src['i']
                sdss_ierr = sdss_src['err_i']
                sdss_z = sdss_src['z']
                sdss_zerr = sdss_src['err_z']
                print('Found SDSS counterpart at RA=' + str(np.around(sdss_ra, decimals=3)) + ' deg and DEC=' + str(np.around(sdss_dec, decimals=3)) + ' deg!')
                match_spec = SDSS.query_region(coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'), photoobj_fields=['objid'], specobj_fields=['ra','dec','z','zerr'], spectro=True, radius=3.2 * u.arcsec, data_release=16)
                if match_spec != None:
                    match_spec_crossid = match_spec[np.where(match_spec['objid'] == sdss_src['objid'])]
                    if len(match_spec_crossid) == 1:
                        sdss_rs = match_spec_crossid['z']
                        sdss_rserr = match_spec_crossid['zerr']
                        print('Found SDSS redshift information for counterpart with z=' + str(sdss_rs) + '!')
                    elif len(match_spec_crossid) > 1:
                        sdss_rs = match_spec_crossid['z'][0]
                        sdss_rserr = match_spec_crossid['zerr'][0]
                        print('Found SDSS redshift information for counterpart with z=' + str(sdss_rs) + '!')
                    else:
                        sdss_rs = np.nan
                        sdss_rserr = np.nan
                        print('No SDSS redshift information for counterpart available!')
                else:
                    sdss_rs = np.nan
                    sdss_rserr = np.nan
            else:
                sdss_id = np.nan
                sdss_ra = np.nan
                sdss_dec = np.nan
                sdss_u = np.nan
                sdss_uerr = np.nan
                sdss_g = np.nan
                sdss_gerr = np.nan
                sdss_r = np.nan
                sdss_rerr = np.nan
                sdss_i = np.nan
                sdss_ierr = np.nan
                sdss_z = np.nan
                sdss_zerr = np.nan
                sdss_rs = np.nan
                sdss_rserr = np.nan
        else:
            sdss_id = np.nan
            sdss_ra = np.nan
            sdss_dec = np.nan
            sdss_u = np.nan
            sdss_uerr = np.nan
            sdss_g = np.nan
            sdss_gerr = np.nan
            sdss_r = np.nan
            sdss_rerr = np.nan
            sdss_i = np.nan
            sdss_ierr = np.nan
            sdss_z = np.nan
            sdss_zerr = np.nan
            sdss_rs = np.nan
            sdss_rserr = np.nan
    return sdss_id, sdss_ra, sdss_dec, sdss_u, sdss_uerr, sdss_g, sdss_gerr, sdss_r, sdss_rerr, sdss_i, sdss_ierr, sdss_z, sdss_zerr, sdss_rs, sdss_rserr