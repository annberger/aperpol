import shutil
import os
from PyPDF2 import PdfMerger, PdfFileReader
import glob

from astropy.table import Table
import numpy as np

from kapteyn import maputils
from kapteyn import wcs
from matplotlib import pyplot as plt

import util
import misc

def make_plots(self):
#    prepare_images(self)
    make_pdfs(self)
    merge_pdfs(self)


def prepare_images(self):
    # Reproject the images to the WISE resolution
    shutil.copy(self.polanalysisdir + '/wise3.4.fits', self.polanalysisplotdir + '/wise3.4.fits')
    shutil.copy(self.polanalysisdir + '/wise4.6.fits', self.polanalysisplotdir + '/wise4.6.fits')
    shutil.copy(self.polanalysisdir + '/wise12.fits', self.polanalysisplotdir + '/wise12.fits')
    shutil.copy(self.polanalysisdir + '/wise22.fits', self.polanalysisplotdir + '/wise22.fits')

    util.remove_dims(self.polanalysisdir + '/TP.fits', self.polanalysisdir + '/TP_rd.fits')
    util.reproject_image(self.polanalysisdir + '/TP_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/TP.fits')

    util.remove_dims(self.polanalysisdir + '/TP_norm.fits', self.polanalysisdir + '/TP_norm_rd.fits')
    util.reproject_image(self.polanalysisdir + '/TP_norm_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/TP_norm.fits')

    util.reproject_image(self.polanalysisdir + '/PI.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/PI.fits')

    util.remove_dims(self.polanalysisdir + '/RM_corr_blanked.fits', self.polanalysisdir + '/RM_corr_blanked_rd.fits')
    util.reproject_image(self.polanalysisdir + '/RM_corr_blanked_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/RM.fits')

    util.remove_dims(self.polanalysisdir + '/PA_corr_blanked.fits', self.polanalysisdir + '/PA_corr_blanked_rd.fits')
    util.reproject_image(self.polanalysisdir + '/PA_corr_blanked_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/PA.fits')

    util.remove_dims(self.polanalysisdir + '/PA0_corr_blanked.fits', self.polanalysisdir + '/PA0_corr_blanked_rd.fits')
    util.reproject_image(self.polanalysisdir + '/PA0_corr_blanked_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/PA0.fits')

    util.remove_dims(self.polanalysisdir + '/PI_norm.fits', self.polanalysisdir + '/PI_norm_rd.fits')
    util.reproject_image(self.polanalysisdir + '/PI_norm_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/PI_norm.fits')

    util.remove_dims(self.polanalysisdir + '/FP_blanked.fits', self.polanalysisdir + '/FP_blanked_rd.fits')
    util.reproject_image(self.polanalysisdir + '/FP_blanked_rd.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/FP.fits')

    try:
        util.reproject_image(self.polanalysisdir + '/sdss_g.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/sdss_g.fits')
    except:
        util.blank_image(self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/sdss_g.fits')


def make_pdfs(self, interactive=False, final=False):

    # Delete the old plots
    os.system('rm -rf ' + self.polanalysisplotdir + '/*.pdf')

    # Load the source table
    if interactive:
        PI_cat = Table.read(self.polanalysisdir + '/PI_cat_final_interactive.txt', format='ascii')
        comp_cat = Table.read(self.polanalysisdir + '/PI_cat_final_interactive.txt', format='ascii')
        # Load the total power catalogue for plotting
        TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final_interactive.txt', format='ascii')

    else:
        PI_cat = Table.read(self.polanalysisdir + '/PI_cat_final.txt', format='ascii')
        comp_cat = Table.read(self.polanalysisdir + '/PI_cat_final.txt', format='ascii')
        # Load the total power catalogue for plotting
        TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final.txt', format='ascii')

#    # Load the total power catalogue for plotting
#    TP_cat = Table.read(self.polanalysisdir + '/TP_cat_final.txt', format='ascii')

    # Remove the double, triple etc. component entries
    del_entries = []
    for source in np.unique(PI_cat['ID']):
        spos = (np.where(source == PI_cat['ID']))
        if (len(spos[0])) > 1:
            del_entries.extend((spos[0][1:]))
    PI_cat.remove_rows((del_entries))

    plt.ioff()

    lastid = ''

    # Load the images
    fo_PI = maputils.FITSimage(self.polanalysisplotdir + '/PI.fits', memmap=0)

    # Get the beam parameters for PI and TP
    be_PI = maputils.FITSimage(self.polanalysisdir + '/PI.fits', memmap=0)
    be_TP = maputils.FITSimage(self.polanalysisdir + '/TP.fits', memmap=0)
    pi_bmaj = be_PI.hdr['BMAJ']
    pi_bmin = be_PI.hdr['BMIN']
    pi_bpa = be_PI.hdr['BPA']
    tp_bmaj = be_TP.hdr['BMAJ']
    tp_bmin = be_TP.hdr['BMIN']
    tp_bpa = be_TP.hdr['BPA']

    # Generate a list for sources not successfully plotted
    no_success = []

    for source in PI_cat:

        if source['ID'] == lastid:
            pass
        else:
            # try:
            # Get the catalogue entries for each component of the source
            comps = comp_cat[np.where(comp_cat['ID'] == source['ID'])]

            # Prepare the figure
            fig = plt.figure(figsize=(15, 15))
            ax1_PI = fig.add_subplot(331)
            ax2_TP = fig.add_subplot(332)
            ax3_RM = fig.add_subplot(333)
            ax4_FP = fig.add_subplot(334)
            ax5_WISE3_4 = fig.add_subplot(335)
            ax6_WISE4_6 = fig.add_subplot(336)
            ax7_SDSS = fig.add_subplot(337)
            ax8_WISE12 = fig.add_subplot(338)
            ax9_WISE22 = fig.add_subplot(339)

            # Set the positions of the individual frames
            header = fo_PI.hdr
            proj = wcs.Projection(header)

            ra = float(source['RA'])
            dec = float(source['DEC'])
            id = source['ID']
            scode = source['S_Code']
            pi = float(source['PI_Isl'])
            pi_err = float(source['PI_Isl_err'])
            tp_id = source['TP_ID']
            tp_ra = float(source['RA_TP'])
            tp_dec = float(source['DEC_TP'])
            tp = float(source['TP'])
            tp_err = float(source['TP_err'])
            rm = float(source['RM_Comp'])
            rm_err = float(source['RM_Comp_err'])
            fp = float(source['FP_Isl'])
            fp_err = float(source['FP_Isl_err'])
            nvss_ra = float(source['NVSS_RA'])
            nvss_dec = float(source['NVSS_DEC'])
            nvss_pi = float(source['NVSS_PI'])
            nvss_pi_err = float(source['NVSS_PI_err'])
            nvss_rm = float(source['NVSS_RM'])
            nvss_rm_err = float(source['NVSS_RM_err'])
            nvss_fp = float(source['NVSS_FP'])
            nvss_fp_err = float(source['NVSS_FP_err'])
            wise_id = source['WISE_ID']
            wise_ra = float(source['WISE_RA'])
            wise_dec = float(source['WISE_DEC'])
            wise_flux_3_4 = float(source['WISE_Flux_3.4'])
            wise_flux_3_4_err = float(source['WISE_Flux_3.4_err'])
            wise_flux_4_6 = float(source['WISE_Flux_4.6'])
            wise_flux_4_6_err = float(source['WISE_Flux_4.6_err'])
            wise_flux_12 = float(source['WISE_Flux_12'])
            wise_flux_12_err = float(source['WISE_Flux_12_err'])
            wise_flux_22 = float(source['WISE_Flux_22'])
            wise_flux_22_err = float(source['WISE_Flux_22_err'])
            sdss_id = source['SDSS_ID']
            sdss_ra = float(source['SDSS_RA'])
            sdss_dec = float(source['SDSS_DEC'])
            sdss_flux_u = float(source['SDSS_Flux_U'])
            sdss_flux_u_err = float(source['SDSS_Flux_U_err'])
            sdss_flux_g = float(source['SDSS_Flux_G'])
            sdss_flux_g_err = float(source['SDSS_Flux_G_err'])
            sdss_flux_r = float(source['SDSS_Flux_R'])
            sdss_flux_r_err = float(source['SDSS_Flux_R_err'])
            sdss_flux_i = float(source['SDSS_Flux_I'])
            sdss_flux_i_err = float(source['SDSS_Flux_I_err'])
            sdss_flux_z = float(source['SDSS_Flux_Z'])
            sdss_flux_z_err = float(source['SDSS_Flux_Z_err'])
            sdss_z = float(source['SDSS_z'])
            sdss_z_err = float(source['SDSS_z_err'])

            world = proj.topixel((ra, dec))

            imlist = [self.polanalysisplotdir + '/PI.fits', self.polanalysisplotdir + '/PI_norm.fits', self.polanalysisplotdir + '/TP.fits', self.polanalysisplotdir + '/TP_norm.fits', self.polanalysisplotdir + '/RM.fits', self.polanalysisplotdir + '/FP.fits', self.polanalysisplotdir + '/wise3.4.fits', self.polanalysisplotdir + '/wise4.6.fits', self.polanalysisplotdir + '/sdss_g.fits', self.polanalysisplotdir + '/wise12.fits', self.polanalysisplotdir + '/wise22.fits']

            # Generate the cutout images
            for image in imlist:
                misc.make_cutout(image, ra, dec, 300)

            # Load the cutout images
            co_PI = maputils.FITSimage(self.polanalysisplotdir + '/PI_cutout.fits', memmap=0)
            co_PInorm = maputils.FITSimage(self.polanalysisplotdir + '/PI_norm_cutout.fits', memmap=0)
            co_TP = maputils.FITSimage(self.polanalysisplotdir + '/TP_cutout.fits', memmap=0)
            co_TPnorm = maputils.FITSimage(self.polanalysisplotdir + '/TP_norm_cutout.fits', memmap=0)
            co_RM = maputils.FITSimage(self.polanalysisplotdir + '/RM_cutout.fits', memmap=0)
            co_FP = maputils.FITSimage(self.polanalysisplotdir + '/FP_cutout.fits', memmap=0)
            co_wise3_4 = maputils.FITSimage(self.polanalysisplotdir + '/wise3.4_cutout.fits', memmap=0)
            co_wise4_6 = maputils.FITSimage(self.polanalysisplotdir + '/wise4.6_cutout.fits', memmap=0)
            co_sdss = maputils.FITSimage(self.polanalysisplotdir + '/sdss_g_cutout.fits', memmap=0)
            co_wise12 = maputils.FITSimage(self.polanalysisplotdir + '/wise12_cutout.fits', memmap=0)
            co_wise22 = maputils.FITSimage(self.polanalysisplotdir + '/wise22_cutout.fits', memmap=0)

            # Define the clip levels for the images
            lclip_PI, hclip_PI = misc.calc_sigmas(co_PI.dat)
            lclip_TP, hclip_TP = misc.calc_sigmas(co_TP.dat)
            lclip_RM = np.nanmin(co_RM.dat)
            hclip_RM = np.nanmax(co_RM.dat)
            lclip_FP = 0.0
            hclip_FP = 2.0 * np.nanmedian(co_FP.dat)
            lclip_wise3_4, hclip_wise3_4 = misc.calc_sigmas(co_wise3_4.dat)
            lclip_wise4_6, hclip_wise4_6 = misc.calc_sigmas(co_wise4_6.dat)
            lclip_sdss, hclip_sdss = misc.calc_sigmas(co_sdss.dat)
            lclip_wise12, hclip_wise12 = misc.calc_sigmas(co_wise12.dat)
            lclip_wise22, hclip_wise22 = misc.calc_sigmas(co_wise22.dat)

            # Contour levels in units of local SNR
            pilevels = [6.25, 8.84, 12.5, 17.68, 25, 35.36, 50, 70.71, 100, 141.42, 200, 282.84, 400, 565.69, 800, 1131.37, 1600]
            tplevels = [7, 14, 28, 56, 112, 224, 448, 896, 1792, 3584, 7168, 14336]

            # text box properties
            tb_props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0)

            # Figure title
            fig.suptitle(id + ' ' + scode, size=20)

            # Generate a list of all sources visible in the cutout image
            co_hdr = co_PI.hdr
            co_proj = wcs.Projection(co_hdr)

            # List of polarised sources
            copi_world = co_proj.topixel((PI_cat['RA'], PI_cat['DEC']))
            copi_world_x = copi_world[0]
            copi_world_y = copi_world[1]
            copi_idxs = []
            for c, co in enumerate(copi_world_x):
                if (0.0 <= copi_world_x[c] <= 300.0) and (0.0 <= copi_world_y[c] <= 300.0):
                    copi_idxs.append(c)
            copi_cat = PI_cat[copi_idxs]

            # List of total power sources
            cotp_world = co_proj.topixel((TP_cat['RA_Comp_TP'], TP_cat['DEC_Comp_TP']))
            cotp_world_x = cotp_world[0]
            cotp_world_y = cotp_world[1]
            cotp_idxs = []
            for c, co in enumerate(cotp_world_x):
                if (0.0 <= cotp_world_x[c] <= 300.0) and (0.0 <= cotp_world_y[c] <= 300.0):
                    cotp_idxs.append(c)
            cotp_cat = TP_cat[cotp_idxs]

            # PI-image + PI-position + ID + TP_cont

            ax1_PI.set_title('Polarised Intensity + TP-contours')
            ax1_text = 'PI = ' + str(np.around(pi*1000.0, decimals=2)) + '$\pm$' + str(np.around(pi_err*1000.0, decimals=2)) + ' mJy'
            if np.isnan(nvss_pi):
                pass
            else:
                ax1_text = ax1_text + '\n PI(NVSS) = ' + str(np.around(nvss_pi*1000.0, decimals=2)) + '$\pm$' + str(np.around(nvss_pi_err*1000.0, decimals=2)) + ' mJy'
            ax1_PI.text(0.95,0.95, ax1_text, transform=ax1_PI.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            annim_pi = co_PI.Annotatedimage(ax1_PI, clipmin=lclip_PI, clipmax=hclip_PI, cmap='gray')
            annim_pi.set_blankcolor('w')
            if final==False:
                for comp in comps:
                    annim_pi.Marker(pos=str(comp['RA_Comp']) + ' deg ' + str(comp['DEC_Comp']) + ' deg', marker='x', color='lime', markersize=10, markeredgewidth=1)
                for pis in copi_cat:
                    annim_pi.Marker(pos = str(pis['RA']) + ' deg ' + str(pis['DEC']) + ' deg', marker='x', color='lime', markersize=25, markeredgewidth=1)
                    pixel_coords = co_proj.topixel((pis['RA'], pis['DEC']))
                    new_pixel_coords = misc.shift_marker(pixel_coords, 62.5)
                    new_ra_dec = co_proj.toworld(new_pixel_coords)
                    annim_pi.Marker(pos = str(new_ra_dec[0]) + ' deg ' + str(new_ra_dec[1]) + ' deg', marker='$' + str(pis['ID']).replace(self.prefix + '_','') + '$', color='lime', markersize=80, markeredgewidth=0.01)
                annim_pi.Marker(pos=str(ra) + ' deg ' + str(dec) + ' deg', marker='x', color='cyan', markersize=25, markeredgewidth=1.5)
                pixel_coords = co_proj.topixel((ra, dec))
                new_pixel_coords = misc.shift_marker(pixel_coords, 62.5)
                new_ra_dec = co_proj.toworld(new_pixel_coords)
                annim_pi.Marker(pos= str(new_ra_dec[0]) + ' deg ' + str(new_ra_dec[1]) + ' deg', marker='$' + str(id).replace(self.prefix + '_', '') + '$', color='cyan', markersize=80, markeredgewidth=0.01)
            else:
                annim_pi.Marker(pos=str(ra) + ' deg ' + str(dec) + ' deg', marker='x', color='cyan', markersize=25, markeredgewidth=1.5)
            grat_pi = annim_pi.Graticule()
            grat_pi.setp_gratline(visible=False)
            grat_pi.setp_axislabel(visible=False)
            grat_pi.setp_ticklabel(plotaxis=['left'], fontsize=9)
            grat_pi.setp_ticklabel(plotaxis=['bottom'], visible=False)
            grat_pi.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            tpim = co_TPnorm.Annotatedimage(ax1_PI, clipmin=0.0, clipmax=1.0)
            tpim.Contours(levels = tplevels, linewidths = 1.0)
            annim_pi.Beam(tp_bmaj, tp_bmin, pa=tp_bpa, pos='30,30', facecolor='aqua')
            annim_pi.Image()
            annim_pi.plot()
            tpim.plot()

            # TP-image + TP-position + NVSS + PI_cont

            ax2_TP.set_title('Total power + PI-contours')
            ax2_text = 'TP = ' + str(np.around(tp * 1000.0, decimals=2)) + '$\pm$' + str(np.around(tp_err * 1000.0, decimals=2)) + ' mJy'
            ax2_TP.text(0.95, 0.95, ax2_text, transform=ax2_TP.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            annim_tp = co_TP.Annotatedimage(ax2_TP, clipmin=lclip_TP, clipmax=hclip_TP, cmap='gray')
            annim_tp.set_blankcolor('w')
            if final==False:
                for tps in cotp_cat:
                    annim_tp.Marker(pos=str(tps['RA_TP']) + ' deg ' + str(tps['DEC_TP']) + ' deg', marker='x', color='lime', markersize=25, markeredgewidth=1)
                    annim_tp.Marker(pos=str(tps['RA_Comp_TP']) + ' deg ' + str(tps['DEC_Comp_TP']) + ' deg', marker='x', color='lime', markersize=10, markeredgewidth=1)
                    pixel_coords = co_proj.topixel((tps['RA_TP'], tps['DEC_TP']))
                    new_pixel_coords = misc.shift_marker(pixel_coords, 62.5)
                    new_ra_dec = co_proj.toworld(new_pixel_coords)
    #                    ms = misc.define_markersize(str(tps['Isl_id']))
    #                    annim_tp.Marker(pos=str(new_ra_dec[0]) + ' deg ' + str(new_ra_dec[1]) + ' deg', marker='$' + str(tps['TP_ID']) + '$', color='lime', markersize=ms, markeredgewidth=0.01)
                    annim_tp.Marker(pos=str(new_ra_dec[0]) + ' deg ' + str(new_ra_dec[1]) + ' deg', marker='$' + str(tps['TP_ID']).replace('TP_','') + '$', color='lime', markersize=80, markeredgewidth=0.01)
            else:
                pass
            if np.isnan(nvss_ra):
                pass
            else:
                annim_tp.Marker(pos=str(nvss_ra) + ' deg ' + str(nvss_dec) + ' deg', marker='x', color='r', markersize=50, markeredgewidth=2)
            if np.isnan(tp_ra):
                pass
            else:
                annim_tp.Marker(pos=str(tp_ra) + ' deg ' + str(tp_dec) + ' deg', marker='x', color='cyan', markersize=25, markeredgewidth=1.5)
                if final==False:
                    pixel_coords_main = co_proj.topixel((tp_ra, tp_dec))
                    new_pixel_coords_main = misc.shift_marker(pixel_coords_main, 62.5)
                    new_ra_dec_main = co_proj.toworld(new_pixel_coords_main)
    #                ms_main = misc.define_markersize(str(int(tp_id)))
    #                annim_tp.Marker(pos=str(new_ra_dec_main[0]) + ' deg ' + str(new_ra_dec_main[1]) + ' deg', marker='$' + str(int(tp_id)) + '$', color='cyan', markersize=ms_main, markeredgewidth=0.01)
                    annim_tp.Marker(pos=str(new_ra_dec_main[0]) + ' deg ' + str(new_ra_dec_main[1]) + ' deg', marker='$' + tp_id.replace('TP_','') + '$', color='cyan', markersize=80, markeredgewidth=0.01)
                    ax2_text = 'TP = ' + str(np.around(tp * 1000.0, decimals=2)) + '$\pm$' + str(np.around(tp_err * 1000.0, decimals=2)) + ' mJy'
                    ax2_TP.text(0.95, 0.95, ax2_text, transform=ax2_TP.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
                else:
                    pass
            grat_tp = annim_tp.Graticule()
            grat_tp.setp_gratline(visible=False)
            grat_tp.setp_axislabel(plotaxis=[0, 1], fontsize=9, visible=False)
            grat_tp.setp_ticklabel(plotaxis=['left', 'bottom'], visible=False)
            grat_tp.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            piim = co_PInorm.Annotatedimage(ax2_TP, clipmin=0.0, clipmax=1.0)
            piim.Contours(levels = pilevels, linewidths = 1.0)
            annim_tp.Beam(pi_bmaj, pi_bmin, pa=pi_bpa, pos='30,30', facecolor='aqua')
            annim_tp.Image()
            annim_tp.plot()
            piim.plot()

            # RM + PI-contours

            ax3_RM.set_title('Rotation Measure + PI-contours')
            annim_rm = co_RM.Annotatedimage(ax3_RM, clipmin=lclip_RM, clipmax=hclip_RM, cmap='winter')
            annim_rm.set_blankcolor('w')
            if len(comps) > 1:
                ax3_text = ''
                for c, co in enumerate(comps):
                    ax3_text = ax3_text + 'RM(C' + str(c) + ') = ' + str(np.around(co['RM_Comp'], decimals=1)) + '$\pm$' + str(np.around(co['RM_Comp_err'], decimals=1)) + ' rad/m$^2$\n'
            else:
                ax3_text = 'RM = ' + str(np.around(rm, decimals=1)) + '$\pm$' + str(np.around(rm_err, decimals=1)) + ' rad/m$^2$'
            if np.isnan(nvss_rm):
                pass
            else:
                ax3_text = ax3_text + '\n RM(NVSS) = '+ str(np.around(nvss_rm, decimals=1)) + '$\pm$' + str(np.around(nvss_rm_err, decimals=1)) + ' rad/m$^2$'
            ax3_RM.text(0.95, 0.95, ax3_text, transform=ax3_RM.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_rm = annim_rm.Graticule()
            grat_rm.setp_gratline(visible=False)
            grat_rm.setp_axislabel(plotaxis=[0, 1], fontsize=9, visible=False)
            grat_rm.setp_ticklabel(plotaxis=['left', 'bottom'], visible=False)
            grat_rm.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            piim = co_PInorm.Annotatedimage(ax3_RM, clipmin=0.0, clipmax=1.0)
            piim.Contours(levels=pilevels, linewidths=1.0, colors='k')
            colorbar_rmframe = fig.add_axes((0.82,0.67,0.14,0.008))
            colbar_rm = annim_rm.Colorbar(fontsize=8, orientation='horizontal', frame=colorbar_rmframe)
            units_rm = 'Rotation Measure [rad/m$^2$]'
            colbar_rm.set_label(label=units_rm, labelpad=-40)
            annim_rm.Image()
            annim_rm.plot()
            piim.plot()

            # FP + PI-contours

            ax4_FP.set_title('Fractional Polarisation + PI-contours')
            annim_fp = co_FP.Annotatedimage(ax4_FP, clipmin=lclip_FP, clipmax=hclip_FP, cmap='cool')
            annim_fp.set_blankcolor('w')
            ax4_text = 'FP = ' + str(np.around(fp*100.0, decimals=2)) + '$\pm$' + str(np.around(fp_err, decimals=2)) + '%'
            if np.isnan(nvss_fp):
                pass
            else:
                ax4_text = ax4_text + '\n FP(NVSS) = ' + str(np.around(nvss_fp*100.0, decimals=2)) + '$\pm$' + str(np.around(nvss_fp_err*100.0, decimals=2)) + ' %'
            ax4_FP.text(0.95, 0.95, ax4_text, transform=ax4_FP.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_fp = annim_fp.Graticule()
            grat_fp.setp_gratline(visible=False)
            grat_fp.setp_axislabel(plotaxis=[0], fontsize=14)
            grat_fp.setp_axislabel(plotaxis=[1], visible=False)
            grat_fp.setp_ticklabel(plotaxis=['left'] , fontsize=9)
            grat_fp.setp_ticklabel(plotaxis=['bottom'], visible=False)
            grat_fp.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            piim = co_PInorm.Annotatedimage(ax4_FP, clipmin=0.0, clipmax=1.0)
            piim.Contours(levels=pilevels, linewidths=1.0, colors='k')
            colorbar_fpframe = fig.add_axes((0.242, 0.395, 0.14, 0.008))
            colbar_fp = annim_fp.Colorbar(fontsize=8, orientation='horizontal', frame=colorbar_fpframe)
            units_fp = 'Fractional Polarisation'
            colbar_fp.set_label(label=units_fp, labelpad=-40)
            annim_fp.Image()
            annim_fp.plot()
            piim.plot()

            # Wise 3_4 + TP-contours

            ax5_WISE3_4.set_title('WISE 3.4 $\mu$m + TP-contours')
            annim_wise3_4 = co_wise3_4.Annotatedimage(ax5_WISE3_4, clipmin=lclip_wise3_4, clipmax=hclip_wise3_4, cmap='Reds')
            annim_wise3_4.set_blankcolor('w')
            if wise_id == 'nan':
                pass
            else:
                annim_wise3_4.Marker(pos=str(wise_ra) + ' deg ' + str(wise_dec) + ' deg', marker='x', color='limegreen', markersize=25, markeredgewidth=1)
                ax5_text = 'WISE(3.4$\mu$m) = ' + str(np.around(wise_flux_3_4, decimals=2)) + '$\pm$' + str(np.around(wise_flux_3_4_err, decimals=2)) + ' mag'
                ax5_WISE3_4.text(0.95, 0.95, ax5_text, transform=ax5_WISE3_4.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_wise3_4 = annim_wise3_4.Graticule()
            grat_wise3_4.setp_gratline(visible=False)
            grat_wise3_4.setp_axislabel(plotaxis=[0, 1], fontsize=9, visible=False)
            grat_wise3_4.setp_ticklabel(plotaxis=['left', 'bottom'], visible=False)
            grat_wise3_4.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            tpim = co_TPnorm.Annotatedimage(ax5_WISE3_4, clipmin=0.0, clipmax=1.0)
            tpim.Contours(levels = tplevels, linewidths = 1.0, colors='dodgerblue')
            annim_wise3_4.Beam(tp_bmaj, tp_bmin, pa=tp_bpa, pos='30,30', facecolor='aqua')
            annim_wise3_4.Image()
            annim_wise3_4.plot()
            tpim.plot()

            # Wise 4_6 + TP-contours

            ax6_WISE4_6.set_title('WISE 4.6 $\mu$m + TP-contours')
            annim_wise4_6 = co_wise4_6.Annotatedimage(ax6_WISE4_6, clipmin=lclip_wise4_6, clipmax=hclip_wise4_6, cmap='Reds')
            annim_wise4_6.set_blankcolor('w')
            if wise_id == 'nan':
                pass
            else:
                annim_wise4_6.Marker(pos=str(wise_ra) + ' deg ' + str(wise_dec) + ' deg', marker='x', color='limegreen', markersize=25, markeredgewidth=1)
                ax6_text = 'WISE(4.6$\mu$m) = ' + str(np.around(wise_flux_4_6, decimals=2)) + '$\pm$' + str(np.around(wise_flux_4_6_err, decimals=2)) + ' mag'
                ax6_WISE4_6.text(0.95, 0.95, ax6_text, transform=ax6_WISE4_6.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_wise4_6 = annim_wise4_6.Graticule()
            grat_wise4_6.setp_gratline(visible=False)
            grat_wise4_6.setp_axislabel(plotaxis=[0, 1], fontsize=9, visible=False)
            grat_wise4_6.setp_ticklabel(plotaxis=['left', 'bottom'], visible=False)
            grat_wise4_6.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            tpim = co_TPnorm.Annotatedimage(ax6_WISE4_6, clipmin=0.0, clipmax=1.0)
            tpim.Contours(levels = tplevels, linewidths = 1.0, colors='dodgerblue')
            annim_wise4_6.Beam(tp_bmaj, tp_bmin, pa=tp_bpa, pos='30,30', facecolor='aqua')
            annim_wise4_6.Image()
            annim_wise4_6.plot()
            tpim.plot()

            # SDSS + TP-contours

            ax7_SDSS.set_title('SDSS G + TP-contours')
            annim_sdss = co_sdss.Annotatedimage(ax7_SDSS, clipmin=lclip_sdss, clipmax=hclip_sdss, cmap='Blues')
            annim_sdss.set_blankcolor('w')
            if np.isnan(sdss_ra):
                pass
            else:
                annim_sdss.Marker(pos=str(sdss_ra) + ' deg ' + str(sdss_dec) + ' deg', marker='x', color='yellow', markersize=25, markeredgewidth=1)
                ax7_text = ''
                if np.isnan(sdss_flux_u):
                    pass
                else:
                    ax7_text = ax7_text + 'SDSS(U) = ' + str(np.around(sdss_flux_u, decimals=2)) + '$\pm$' + str(np.around(sdss_flux_u_err, decimals=2)) + ' mag\n'
                if np.isnan(sdss_flux_g):
                    pass
                else:
                    ax7_text = ax7_text + 'SDSS(G) = ' + str(np.around(sdss_flux_g, decimals=2)) + '$\pm$' + str(np.around(sdss_flux_g_err, decimals=2)) + ' mag\n'
                if np.isnan(sdss_flux_r):
                    pass
                else:
                    ax7_text = ax7_text + 'SDSS(R) = ' + str(np.around(sdss_flux_r, decimals=2)) + '$\pm$' + str(np.around(sdss_flux_r_err, decimals=2)) + ' mag\n'
                if np.isnan(sdss_flux_i):
                    pass
                else:
                    ax7_text = ax7_text + 'SDSS(I) = ' + str(np.around(sdss_flux_i, decimals=2)) + '$\pm$' + str(np.around(sdss_flux_i_err, decimals=2)) + ' mag\n'
                if np.isnan(sdss_flux_z):
                    pass
                else:
                    ax7_text = ax7_text + 'SDSS(Z) = ' + str(np.around(sdss_flux_z, decimals=2)) + '$\pm$' + str(np.around(sdss_flux_z_err, decimals=2)) + ' mag'
                if np.isnan(sdss_z):
                    pass
                else:
                    ax7_text = ax7_text + '\n z = ' + str(np.around(sdss_z, decimals=2)) + '$\pm$' + str(np.around(sdss_z_err, decimals=2))
                ax7_SDSS.text(0.95, 0.95, ax7_text, transform=ax7_SDSS.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_sdss = annim_sdss.Graticule()
            grat_sdss.setp_gratline(visible=False)
            grat_sdss.setp_axislabel(visible=False)
            grat_sdss.setp_ticklabel(plotaxis=['bottom', 'left'], fontsize=9, visible=True)
            grat_sdss.setp_tickmark(plotaxis=['left', 'bottom'], visible=False)
            tpim = co_TPnorm.Annotatedimage(ax7_SDSS, clipmin=0.0, clipmax=1.0)
            tpim.Contours(levels = tplevels, linewidths = 1.0)
            annim_sdss.Beam(tp_bmaj, tp_bmin, pa=tp_bpa, pos='30,30', facecolor='aqua')
            annim_sdss.Image()
            annim_sdss.plot()
            tpim.plot()

            # Wise 12 + TP-contours

            ax8_WISE12.set_title('WISE 12 $\mu$m + TP-contours')
            annim_wise12 = co_wise12.Annotatedimage(ax8_WISE12, clipmin=lclip_wise12, clipmax=hclip_wise12, cmap='Reds')
            annim_wise12.set_blankcolor('w')
            if wise_id == 'nan':
                pass
            else:
                annim_wise12.Marker(pos=str(wise_ra) + ' deg ' + str(wise_dec) + ' deg', marker='x', color='limegreen', markersize=25, markeredgewidth=1)
                ax8_text = 'WISE(12$\mu$m) = ' + str(np.around(wise_flux_12, decimals=2)) + '$\pm$' + str(np.around(wise_flux_12_err, decimals=2)) + ' mag'
                ax8_WISE12.text(0.95, 0.95, ax8_text, transform=ax8_WISE12.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_wise12 = annim_wise12.Graticule()
            grat_wise12.setp_gratline(visible=False)
            grat_wise12.setp_axislabel(plotaxis=[0], fontsize=9, visible=False)
            grat_wise12.setp_axislabel(plotaxis=[1], fontsize=14)
            grat_wise12.setp_ticklabel(plotaxis=['left'], visible=False)
            grat_wise12.setp_tickmark(plotaxis=['left','bottom'], visible=False)
            tpim = co_TPnorm.Annotatedimage(ax8_WISE12, clipmin=0.0, clipmax=1.0)
            tpim.Contours(levels = tplevels, linewidths = 1.0, colors='dodgerblue')
            annim_wise12.Beam(tp_bmaj, tp_bmin, pa=tp_bpa, pos='30,30', facecolor='aqua')
            annim_wise12.Image()
            annim_wise12.plot()
            tpim.plot()

            # Wise 22 + TP-contours

            ax9_WISE22.set_title('WISE 22 $\mu$m + TP-contours')
            annim_wise22 = co_wise22.Annotatedimage(ax9_WISE22, clipmin=lclip_wise22, clipmax=hclip_wise22, cmap='Reds')
            annim_wise22.set_blankcolor('w')
            if wise_id == 'nan':
                pass
            else:
                annim_wise22.Marker(pos=str(wise_ra) + ' deg ' + str(wise_dec) + ' deg', marker='x', color='limegreen', markersize=25, markeredgewidth=1)
                ax9_text = 'WISE(22$\mu$m) = ' + str(np.around(wise_flux_22, decimals=2)) + '$\pm$' + str(np.around(wise_flux_22_err, decimals=2)) + ' mag'
                ax9_WISE22.text(0.95, 0.95, ax9_text, transform=ax9_WISE22.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=tb_props)
            grat_wise22 = annim_wise22.Graticule()
            grat_wise22.setp_gratline(visible=False)
            grat_wise22.setp_axislabel(plotaxis=[0,1], fontsize=9, visible=False)
            grat_wise22.setp_ticklabel(plotaxis=['left'], visible=False)
            grat_wise22.setp_tickmark(plotaxis=['left','bottom'], visible=False)
            tpim = co_TPnorm.Annotatedimage(ax9_WISE22, clipmin=0.0, clipmax=1.0)
            tpim.Contours(levels = tplevels, linewidths = 1.0, colors='dodgerblue')
            annim_wise22.Beam(tp_bmaj, tp_bmin, pa=tp_bpa, pos='30,30', facecolor='aqua')
            annim_wise22.Image()
            annim_wise22.plot()
            tpim.plot()

            # Show the plots
            plt.tight_layout(rect=(0.05,0.05,1,0.95))

            fig.savefig(self.polanalysisplotdir + '/' + id + '.pdf')
            fig.clear()

            # Remove the subplots for the next iteration
            ax1_PI.clear()
            ax2_TP.clear()
            ax3_RM.clear()
            ax4_FP.clear()
            ax5_WISE3_4.clear()
            ax6_WISE4_6.clear()
            ax7_SDSS.clear()
            ax8_WISE12.clear()
            ax9_WISE22.clear()

            plt.close('all')

            lastid = source['ID']
            # except:
            #     print('Plots for source ' + source['ID'] + ' could not be generated')
            #     no_success.append(source['ID'])
    if len(no_success) != 0:
        with open(self.polanalysisplotdir + '/plots_failed.txt', 'w') as pf:
            for item in no_success:
                pf.write("%s\n" % item)

    os.system('rm -rf ' + self.polanalysisplotdir + '/*_cutout.fits')


def merge_pdfs(self):

    # Call the PdfFileMerger
    mergedObject = PdfMerger()
    files = sorted(glob.glob(self.polanalysisplotdir + '/' + self.prefix + '*.pdf'))

    for pdf in files:
        mergedObject.append(PdfFileReader(pdf, 'rb'))

    # Write all the files into a file which is named as shown below
    mergedObject.write(self.polanalysisplotdir + '/catalogue.pdf')
