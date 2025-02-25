import os
import subprocess

from config import config
from prototype.module import Module
from moduleResult.minimapResult import MinimapResult
from module.minimap import Minimap
from moduleResult.alignment import Alignment

from utils import IOUtils

class MinimapThresholdModule(Minimap):
    def __init__(self, reference, mode='ont', threads=12, skipComments=True, factors=['most']):
        super().__init__(reference, mode, threads, skipComments)
        self.factors = factors if isinstance(factors, list) else [factors]
        self.moduleName = f'minimapThresh-ref={self.reference};mode={self.mode};thresh-{"_".join(self.factors)}'

    def run(self, samples):
        results = super().run(samples)
        return [self.extractResult(sample, result) for sample, result in zip(samples, results)]
    
    
    def extractResult(self, sample, result):
        if (result is not None):
            if ("positive" in self.factors and result.bestAlignment.quality == 0):
                result = None
            elif ("60" in self.factors and result.bestAlignment.quality < 60):
                result = None
            elif ("completeMatch" in self.factors and result.bestAlignment.queryCoverLength < sample.length):
                result = None
            elif ("singleAlignment" in self.factors and len(result.alignments) > 1):
                result = None
        
        return result