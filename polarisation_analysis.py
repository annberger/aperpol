import util
import rmsynth
import rmanalysis
import catalogue
import cross_id
import make_ancillary
import make_plots
#import interactive
#import leakage


class polarisation_analysis:
    """
    Class to analyse polarisation mosaics.
    """
    module_name = 'POLARISATION ANALYSIS'


    def __init__(self, file_=None, **kwargs):
        self.default = util.load_config(self, file_)
        util.set_dirs(self)
        self.config_file_name = file_


    def go(self):
        """
        Function to analyse the Apertif polarisation data of a single mosaic in Stokes Q and U
        """
#        util.gen_dirs(self)
#        rmsynth.rmsynth(self)
#        rmanalysis.rmanalysis(self)
#        catalogue.catalogue(self)
#        cross_id.cross_id(self)
#        make_ancillary.make_ancillary(self)
        make_plots.make_plots(self)


    def leakage_analysis(self):
        """
        Function to derive the leakage in Stokes Q and V
        """
        leakage.leakage(self)
