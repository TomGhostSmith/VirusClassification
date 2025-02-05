# reconstructed
import os
import json
import subprocess
from Bio import SeqIO

from config import config
from prototype.module import Module
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment
from entity.sample import Sample

from utils import IOUtils

class Minimap(Module):
    def __init__(self, reference, mode='ont', threads=12, skipComments=True):
        self.reference=reference
        self.mode = mode
        self.threads = threads
        self.skipComments = skipComments
        super().__init__(f'minimap-ref={self.reference};mode={self.mode}')
        self.baseName = self.moduleName  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()

    def minimap(self, samples):
        queryFile = f"{config.cacheFolder}/minimap.fasta"
        resultFile = f"{config.cacheFolder}/alignment.sam"
        IOUtils.writeSampleFasta(samples, queryFile)
        IOUtils.showInfo(f"Begin minimap on {len(samples)} samples")

        command = self.getMinimapCommand(queryFile)
        with open(resultFile, 'wt') as fp:
            subprocess.run(command, shell=True, stdout=fp, stderr=subprocess.DEVNULL)
        targetFP = open(self.cacheFile, 'at')

        thisName = None
        thisOffset = self.cachedSamples["nextOffset"] if "nextOffset" in self.cachedSamples else 0
        nextOffset = thisOffset
        alignmentCount = 0


        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                terms[9] = '*'  # omit the query sequence to reduce storage space usage
                sampleName = terms[0]
                if (sampleName != thisName and thisName is not None):
                    # update last sample
                    self.cachedSamples[thisName] = [thisOffset, alignmentCount]
                    thisOffset = nextOffset
                    alignmentCount = 0

                content = "\t".join(terms) + "\n"
                thisName = sampleName
                nextOffset += len(content)
                alignmentCount += 1
                targetFP.write(content)

            
            if (thisName is not None):
                # update last sample
                self.cachedSamples[thisName] = [thisOffset, alignmentCount]
                self.cachedSamples["nextOffset"] = nextOffset

        targetFP.close()
        os.remove(resultFile)
        os.remove(queryFile)


    def run(self, samples:list[Sample]):
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
            self.minimap(samplesToRun)
        
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->MinimapResult:
        # use baseName to cache the result in the results dict
        if (self.baseName in sample.results):
            return sample.results[self.baseName]
        
        offset, alignmentCount = self.cachedSamples[sample.id]
        cachedResultFP.seek(offset)
        alignments:list[Alignment] = [Alignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
        
        result = MinimapResult()
        for alignment in alignments:
            if (alignment.ref is not None):
                result.addAlignment(alignment)

        if (result.bestAlignment is None):
            result = None
        sample.results[self.baseName] = result
        return result
    
    def getMinimapCommand(self, queryFile):
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        minimapBase = "minimap2"   # if you cannot call minimap2 directly, use its path here
        if self.mode == 'ont':
            mode = "-ax map-ont"
        else:
            mode = "-a"
        thread = f"-t {self.threads}"
        if self.skipComments:
            postProcess = ' | grep -v "^@"'
        else:
            postProcess = ""
        command = f"{minimapBase} {mode} {referenceFasta} {queryFile} {thread} {postProcess}"
        return command