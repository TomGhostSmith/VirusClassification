# reconstructed
import os
import sys
import json
import shutil
import multiprocessing
import subprocess
from tqdm import tqdm

from prototype.module import Module
from config import config
from utils import IOUtils
from entity.sample import Sample
from moduleResult.genomadResult import GenomadResult


class Genomad(Module):
    def __init__(self, threads=12):
        super().__init__("genomad-1.9")
        self.threads = threads
        self.cacheFile = f"{config.cacheResultFolder}/{self.moduleName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.database = f"{config.modelRoot}/genomad"
        self.cachedSamples:dict[str, int] = dict()
    
    def genomad(self, samples:list[Sample])->None:
        queryFile = f"{config.cacheFolder}/genomad.fasta"
        outputFolder = f"{config.cacheFolder}/genomad_output"
        resultFile = f"{outputFolder}/genomad_summary/genomad_virus_summary.tsv"
        IOUtils.writeSampleFasta(samples, queryFile)
        IOUtils.showInfo(f"Begin genomad on {len(samples)} samples")

        command = f"conda run -n genomad genomad end-to-end --splits {self.threads} {queryFile} {outputFolder} {self.database}"
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(command, shell=True)

        targetFP = open(self.cacheFile, 'at')

        nextOffset = self.cachedSamples["nextOffset"] if "nextOffset" in self.cachedSamples else 0

        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                sampleName = terms[0]
                self.cachedSamples[sampleName] = nextOffset

                nextOffset += len(line)
                targetFP.write(line)
            
            self.cachedSamples["nextOffset"] = nextOffset

        targetFP.close()

        for sample in samples:
            if sample.id not in self.cachedSamples:
                self.cachedSamples[sample.id] = -1

        shutil.rmtree(outputFolder)  # remove the output tree in case the genomad tool always use cached sequence
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
            self.genomad(samplesToRun)
        
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->GenomadResult:
        offset = self.cachedSamples[sample.id]
        if offset != -1:
            cachedResultFP.seek(offset)
            result = GenomadResult(cachedResultFP.readline())
        else:
            result = None
        return result