import os.path
import shutil
import time

import numpy as np

import util

from MontagePy.main    import *
from MontagePy.archive import *


def make_ancillary(self):
    make_wise_mosaics(self)
    make_sdss_mosaic(self)


def make_wise_mosaics(self):
    ra_cent, dec_cent, ra_size, dec_size = util.get_imageextent(self)
    make_workenv_wise(self)
    retrieve_wise_data(self, ra_cent, dec_cent, ra_size, dec_size)
    repr_coadd_wise(self)
    matchbg_wise(self)
    correctbg_mos_wise(self)
    delete_obs_wise(self)


def make_sdss_mosaic(self):
    ra_cent, dec_cent, ra_size, dec_size = util.get_imageextent(self)
    make_workenv_sdss(self)
    retrieve_sdss_data(self, ra_cent, dec_cent, ra_size, dec_size)
    repr_coadd_sdss(self)
    matchbg_sdss(self)
    correctbg_mos_sdss(self)
    delete_obs_sdss(self)


def make_workenv_wise(self):
    for w in ['wise3.4','wise4.6','wise12','wise22']:
        if os.path.isdir(self.polanalysisdir + '/' + w):
            pass
        else:
            os.makedirs(self.polanalysisdir + '/' + w)
            for s in ['raw','projected','diffs','corrected']:
                if os.path.isdir(self.polanalysisdir + '/' + w + '/' + s):
                    pass
                else:
                    os.makedirs(self.polanalysisdir + '/' + w + '/' + s)


def retrieve_wise_data(self, ra, dec, ra_size, dec_size):
    location = str(ra) + ' ' + str(dec) + ' Equ B2000'
    size = np.max([ra_size, dec_size])
    for w in ['wise3.4', 'wise4.6', 'wise12', 'wise22']:
        os.chdir(self.polanalysisdir + '/' + w)
        if w == 'wise3.4':
            mHdr(location, ra_size, dec_size, 'region.hdr')
            mArchiveDownload('WISE 3.4 micron', location, size, 'raw')
            time.sleep(1)
            mImgtbl('raw', 'rimages.tbl')
        elif w == 'wise4.6':
            mHdr(location, ra_size, dec_size, 'region.hdr')
            mArchiveDownload('WISE 4.6 micron', location, size, 'raw')
            time.sleep(1)
            mImgtbl('raw', 'rimages.tbl')
        elif w == 'wise12':
            mHdr(location, ra_size, dec_size, 'region.hdr')
            mArchiveDownload('WISE 12 micron', location, size, 'raw')
            time.sleep(1)
            mImgtbl('raw', 'rimages.tbl')
        elif w == 'wise22':
            mHdr(location, ra_size, dec_size, 'region.hdr')
            mArchiveDownload('WISE 22 micron', location, size, 'raw')
            time.sleep(1)
            mImgtbl('raw', 'rimages.tbl')


def repr_coadd_wise(self):
    for w in ['wise3.4', 'wise4.6', 'wise12', 'wise22']:
        os.chdir(self.polanalysisdir + '/' + w)
        mProjExec('raw', 'rimages.tbl', 'region.hdr', projdir='projected', quickMode=True)
        mImgtbl('projected', 'pimages.tbl')
        mAdd('projected', 'pimages.tbl', 'region.hdr', 'uncorrected.fits')


def matchbg_wise(self):
    for w in ['wise3.4', 'wise4.6', 'wise12', 'wise22']:
        os.chdir(self.polanalysisdir + '/' + w)
        mOverlaps('pimages.tbl', 'diffs.tbl')
        mDiffFitExec('projected', 'diffs.tbl', 'region.hdr', 'diffs', 'fits.tbl')
        mBgModel('pimages.tbl', 'fits.tbl', 'corrections.tbl')


def correctbg_mos_wise(self):
    for w in ['wise3.4', 'wise4.6', 'wise12', 'wise22']:
        os.chdir(self.polanalysisdir + '/' + w)
        mBgExec('projected', 'pimages.tbl', 'corrections.tbl', 'corrected')
        mImgtbl('corrected', 'cimages.tbl')
        mAdd('corrected', 'cimages.tbl', 'region.hdr', 'mosaic.fits')


def delete_obs_wise(self):
    for w in ['wise3.4', 'wise4.6', 'wise12', 'wise22']:
        if os.path.isfile(self.polanalysisdir + '/' + w + '/mosaic.fits'):
            shutil.move(self.polanalysisdir + '/' + w + '/mosaic.fits', self.polanalysisdir + '/' + w + '.fits')
    #        shutil.rmtree(self.polanalysisdir + '/' + w)
        else:
            pass


def make_workenv_sdss(self):
    for w in ['sdss_g']:
        if os.path.isdir(self.polanalysisdir + '/' + w):
            pass
        else:
            os.makedirs(self.polanalysisdir + '/' + w)
            for s in ['raw','projected','diffs','corrected']:
                if os.path.isdir(self.polanalysisdir + '/' + w + '/' + s):
                    pass
                else:
                    os.makedirs(self.polanalysisdir + '/' + w + '/' + s)


def retrieve_sdss_data(self, ra, dec, ra_size, dec_size):
    location = str(ra) + ' ' + str(dec) + ' Equ B2000'
    size = np.max([ra_size, dec_size])
    for w in ['sdss_g']:
        os.chdir(self.polanalysisdir + '/' + w)
        if w == 'sdss_g':
            mHdr(location, ra_size, dec_size, 'region.hdr')
            mArchiveDownload('SDSS G', location, size, 'raw')
            time.sleep(1)
            mImgtbl('raw', 'rimages.tbl')


def repr_coadd_sdss(self):
    for w in ['sdss_g']:
        os.chdir(self.polanalysisdir + '/' + w)
        mProjExec('raw', 'rimages.tbl', 'region.hdr', projdir='projected', quickMode=True)
        mImgtbl('projected', 'pimages.tbl')
        mAdd('projected', 'pimages.tbl', 'region.hdr', 'uncorrected.fits')


def matchbg_sdss(self):
    for w in ['sdss_g']:
        os.chdir(self.polanalysisdir + '/' + w)
        mOverlaps('pimages.tbl', 'diffs.tbl')
        mDiffFitExec('projected', 'diffs.tbl', 'region.hdr', 'diffs', 'fits.tbl')
        mBgModel('pimages.tbl', 'fits.tbl', 'corrections.tbl')


def correctbg_mos_sdss(self):
    for w in ['sdss_g']:
        os.chdir(self.polanalysisdir + '/' + w)
        mBgExec('projected', 'pimages.tbl', 'corrections.tbl', 'corrected')
        mImgtbl('corrected', 'cimages.tbl')
        mAdd('corrected', 'cimages.tbl', 'region.hdr', 'mosaic.fits')


def delete_obs_sdss(self):
    for w in ['sdss_g']:
        if os.path.isfile(self.polanalysisdir + '/' + w + '/mosaic.fits'):
            shutil.move(self.polanalysisdir + '/' + w + '/mosaic.fits', self.polanalysisdir + '/' + w + '.fits')
    #        shutil.rmtree(self.polanalysisdir + '/' + w)
        else:
            pass
