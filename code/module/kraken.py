# reconstructed
import os
import sys
import json
import multiprocessing
import subprocess
from tqdm import tqdm

from prototype.module import Module
from config import config
from utils import IOUtils
from entity.sample import Sample
from moduleResult.krakenResult import KrakenResult


class Kraken(Module):
    def __init__(self):
        super().__init__("kraken-old")
        self.cacheFile = f"{config.cacheResultFolder}/{self.moduleName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, int] = dict()

    def kraken(self, samples:list[Sample])->None:
        queryFile = f"{config.cacheFolder}/kraken.fasta"
        resultFile = f"{config.cacheFolder}/kraken.tsv"
        IOUtils.writeSampleFasta(samples, queryFile)
        IOUtils.showInfo(f"Begin kraken on {len(samples)} samples")

        command = self.getKrakenCommand(queryFile)
        with open(resultFile, 'wt') as fp:
            subprocess.run(command, shell=True, stdout=fp, stderr=subprocess.DEVNULL)
        targetFP = open(self.cacheFile, 'at')

        nextOffset = self.cachedSamples["nextOffset"] if "nextOffset" in self.cachedSamples else 0

        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                sampleName = terms[1]
                self.cachedSamples[sampleName] = nextOffset

                nextOffset += len(line)
                targetFP.write(line)
            
            self.cachedSamples["nextOffset"] = nextOffset

        targetFP.close()
        os.remove(resultFile)
        os.remove(queryFile)

    def run(self, samples):

        samplesToRun:list[Sample] = list()

        if (os.path.exists(self.cacheIndex)):
            with open(self.cacheIndex) as fp:
                self.cachedSamples = json.load(fp)  # id: [offset, alignmentCount]

        
            for sample in samples:
                if (sample.id not in self.cachedSamples):
                    samplesToRun.append(sample)
        else:
            samplesToRun = samples
        
        if (len(samplesToRun) > 0):
            self.kraken(samplesToRun)
        
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->KrakenResult:
        offset = self.cachedSamples[sample.id]
        cachedResultFP.seek(offset)
        result = KrakenResult(cachedResultFP.readline())

        if (result.finalSpecies is None):
            result = None
        return result
    
    def getKrakenCommand(self, queryFile):
        return f"kraken2 --db {config.modelRoot}/kraken2 {queryFile}"