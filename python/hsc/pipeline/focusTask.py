import os
import math
import traceback

from lsst.pex.config import Config, Field, DictField, ConfigField, ConfigurableField
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
from lsst.pipe.base import Struct, ArgumentParser
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.pipe.tasks.detectAndMeasure import DetectAndMeasureTask

from lsst.obs.subaru.isr import SubaruIsrTask

from lsst.ctrl.pool.pool import Pool
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.pipe.drivers.utils import getDataRef

from .focus import FocusConfig, getDistanceFromFocus, haveSimpleShape

class ProcessFocusConfig(Config):
    focus = ConfigField(dtype=FocusConfig, doc="Focus determination")
    zemax = DictField(keytype=str, itemtype=str, default={},
                      doc="Mapping from filter name to zemax configuration filename")
    isr = ConfigurableField(target=SubaruIsrTask, doc="Instrument Signature Removal")
    installPsf = ConfigurableField(target=InstallGaussianPsfTask, doc="Install a simple PSF model")
    background = ConfigField(dtype=measAlg.estimateBackground.ConfigClass, doc="Background removal")
    detectAndMeasure = ConfigurableField(target=DetectAndMeasureTask, doc="Source detection and measurement")
    starSelector = ConfigurableField(target=measAlg.ObjectSizeStarSelectorTask,
                                     doc="Star selection algorithm")
    doWrite = Field(dtype=bool, default=True, doc="Write processed image?")

    def setDefaults(self):
        """These defaults are suitable for HSC, but may be useful
        for other cameras if the focus code is employed elsewhere.
        """
        Config.setDefaults(self)
        zemaxBase = os.path.join(os.environ["OBS_SUBARU_DIR"], "hsc", "zemax_config%d_0.0.dat")
        self.zemax = dict([(f, zemaxBase % n) for f,n in [
                    ('g', 9), ('r', 1), ('i', 3), ('z', 5), ('y', 7),
                    ('N921', 5), ('N816', 3), ('N1010', 7), ('N387', 9), ('N515', 9),
                    ]])
        self.load(os.path.join(os.environ["OBS_SUBARU_DIR"], "config", "hsc", "isr.py"))
        self.installPsf.fwhm = 9 # pixels
        self.installPsf.width = 31 # pixels
        self.detectAndMeasure.detection.includeThresholdMultiplier = 3.0
        self.detectAndMeasure.measurement.algorithms.names.add("base_GaussianCentroid")
        self.detectAndMeasure.measurement.slots.centroid = "base_GaussianCentroid"
        # set up simple shape, if available (because focus calibrations are for that)
        # If it's not available, we'll crash later; but we don't want to crash here (brings everything down)!
        if haveSimpleShape:
            self.detectAndMeasure.measurement.algorithms.names.add("ext_simpleShape_SimpleShape")
            self.detectAndMeasure.measurement.algorithms["ext_simpleShape_SimpleShape"].sigma = 5.0 # pixels

        # set up background estimate
        self.background.ignoredPixelMask = ['EDGE', 'NO_DATA', 'DETECTED', 'DETECTED_NEGATIVE', 'BAD']
        self.detectAndMeasure.detection.background.algorithm='LINEAR'
        self.detectAndMeasure.doDeblend = False
        self.starSelector.badFlags = ["base_PixelFlags_flag_edge",
                                      "base_PixelFlags_flag_interpolatedCenter",
                                      "base_PixelFlags_flag_saturatedCenter",
                                      "base_PixelFlags_flag_bad",
                                      ]
        self.starSelector.sourceFluxField = "base_GaussianFlux_flux"
        self.starSelector.widthMax = 20.0
        self.starSelector.widthStdAllowed = 5.0


class ProcessFocusTask(BatchPoolTask):
    ConfigClass = ProcessFocusConfig
    _DefaultName = "processFocus"

    def __init__(self, *args, **kwargs):
        BatchPoolTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("installPsf")
        self.makeSubtask("detectAndMeasure", schema=self.schema)
        self.candidateKey = self.schema.addField(
            "hscPipeline_focus_candidate", type="Flag",
            doc=("Flag set if the source was a candidate for PSF determination, "
                 "as determined by the star selector.")
        )
        self.makeSubtask("starSelector")

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        config = parsedCmd.config
        numCcds = len(config.focus.aboveList) + len(config.focus.belowList)
        numCycles = int(math.ceil(numCcds/float(numCores)))
        numExps = len(cls.RunnerClass.getTargetList(parsedCmd))
        return time*numExps*numCycles

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="processFocus", *args, **kwargs)
        parser.add_id_argument("--id", datasetType="raw", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    def run(self, expRef):
        """Measure focus for exposure

        This method is the top-level for running the focus measurement
        as a stand-alone BatchPoolTask.

        Only the master node runs this method.
        """
        pool = Pool("processFocus")
        pool.cacheClear()
        pool.storeSet(butler=expRef.getButler())

        dataIdList = sorted([ccdRef.dataId for ccdRef in expRef.subItems("ccd") if
                             ccdRef.datasetExists("raw") and self.isFocus(ccdRef)])

        results = pool.map(self.processPool, dataIdList)

        camera = expRef.get("camera")
        plotFilename = expRef.get("focusPlot_filename")
        focus = self.measureFocus(results, camera, plotFilename)
        self.log.info("Focus result for %s: %s" % (expRef.dataId, focus))
        return focus

    def isFocus(self, dataRef):
        """Is the provided dataRef for a focus CCD?"""
        ccdId = dataRef.dataId["ccd"]
        return self.config.focus.isFocusCcd(ccdId)

    def processPool(self, cache, dataId):
        """Process focus CCD under pool

        This is a mediator for the 'process' method when running
        under the Pool.

        Only slave nodes run this method.

        @param cache: Pool cache
        @param dataId: Data identifier for CCD
        @return Processing results (from 'process' method)
        """
        try:
            return self.process(getDataRef(cache.butler, dataId))
        except Exception as e:
            self.log.warn("Failed to process %s (%s: %s):\n%s" %
                          (dataId, e.__class__.__name__, e, traceback.format_exc()))
            return None

    def process(self, dataRef):
        """Process focus CCD in preparation for focus measurement

        @param dataRef: Data reference for CCD
        @return Struct(sources: source measurements,
                       ccdId: CCD number,
                       filterName: name of filter,
                       dims: exposure dimensions
                       )
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display

        exp = self.isr.runDataRef(dataRef).exposure

        if display:
            import lsst.afw.display.ds9 as ds9
            ds9.mtv(exp, title="Post-ISR", frame=1)

        self.installPsf.run(exposure=exp)
        bg, exp = measAlg.estimateBackground(exp, self.config.background, subtract=True)

        if display:
            ds9.mtv(exp, title="Post-background", frame=2)

        dmResults = self.detectAndMeasure.run(exp, dataRef.get("expIdInfo"))
        sources = dmResults.sourceCat

        self.starSelector.run(exp, sources, isStarField="hscPipeline_focus_candidate")

        if display:
            ds9.mtv(exp, title="Post-measurement", frame=3)
            with ds9.Buffering():
                for s in sources:
                    ds9.dot("o", s.getX(), s.getY(), frame=3,
                            ctype=ds9.GREEN if s.get("calib.psf.candidate") else ds9.RED)
            import pdb;pdb.set_trace() # pause to allow inspection

        filterName = exp.getFilter().getName()

        if self.config.doWrite:
            dataRef.put(sources, "src")
            dataRef.put(exp, "visitim")

        return Struct(sources=sources, ccdId=dataRef.dataId["ccd"], filterName=filterName,
                      dims=exp.getDimensions())

    def measureFocus(self, resultsList, camera, plotFilename=None):
        """Measure focus from combining individual CCDs

        @param resultsList: Results of processing individual CCDs
        @param camera: Camera object
        @param plotFilename: Name of file for plot
        @return tuple(corrected distance from focus,
                      error in corrected distance from focus,
                      uncorrected distance from focus,
                      error in uncorrected distance from focus,
                      )
        """
        resultsList = [res for res in resultsList if res is not None]
        if not resultsList:
            raise RuntimeError("No results from which to measure focus")
        sources = dict((res.ccdId, res.sources) for res in resultsList)
        ccds = dict((ccd.getId(), ccd) for ccd in camera)

        dims = dict((res.ccdId, res.dims) for res in resultsList)
        filterSet = set([res.filterName for res in resultsList])
        assert len(filterSet) == 1
        filterName = filterSet.pop()
        zemax = self.config.zemax[filterName]

        return getDistanceFromFocus(sources, ccds, dims, zemax, self.config.focus, plotFilename=plotFilename)
