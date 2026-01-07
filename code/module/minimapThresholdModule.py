import os
import subprocess
import multiprocessing

from config import config
from prototype.module import Module
from moduleResult.minimapResult import MinimapResult
from module.minimap import Minimap
from moduleResult.alignment import Alignment

from utils import IOUtils

class MinimapThresholdModule(Minimap):
    def __init__(self, reference, mode='ont', threads=multiprocessing.cpu_count(), skipComments=True, factors=['most']):
        super().__init__(reference, mode, threads, skipComments)
        self.factors = factors if isinstance(factors, list) else [factors]
        self.moduleName = f'minimapThresh-ref={self.reference};mode={self.mode};thresh-{"_".join(self.factors)}'

    def run(self, samples):
        results = super().run(samples)
        return [self.extractResult(sample, result) for sample, result in zip(samples, results)]
    
    
    def extractResult(self, sample, result):
        if (result is not None):
            if ("singleAlignment" in self.factors and len(result) > 1):
                return None
            else:
                results = []
                for r in result:
                    if ("positive" in self.factors and r.bestAlignment.quality == 0):
                        continue
                    elif ("60" in self.factors and r.bestAlignment.quality < 60):
                        continue
                    elif ("completeMatch" in self.factors and r.bestAlignment.queryCoverLength < sample.length):
                        continue
                    else:
                        results.append(r)
                return results if len(results) > 0 else None
        else:
            return None