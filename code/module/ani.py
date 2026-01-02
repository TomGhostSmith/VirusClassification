# reconstructed
import os
import json
import pyfastani
import multiprocessing

from config import config
from prototype.module import Module
from moduleResult.ANIResult import ANIResult
from moduleResult.ANIAlignment import ANIAlignment
from entity.sample import Sample

from utils import IOUtils

class ANI(Module):
    def __init__(self, reference, threads=multiprocessing.cpu_count()):
        self.reference=reference
        self.threads = threads
        super().__init__(f'ANI-ref={self.reference}')
        self.baseName = self.moduleName  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()

    def ani(self, samples:list[Sample]):
        IOUtils.showInfo(f"Begin ANI on {len(samples)} samples")
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        refSamples = IOUtils.loadSamples(referenceFasta)
        sketch = pyfastani.Sketch()
        resultFile = f"{config.cacheFolder}/alignment.sam"

        for r in refSamples:
            sketch.add_draft(r.id, [bytes(r.seq.seq)])

        mapper = sketch.index()

        resFP = open(resultFile, 'wt')  # first write to a result file, then collect it, to avoid data collapse due to interruption
        for query in samples:
            hits = mapper.query_genome(bytes(query.seq.seq), threads=self.threads)
            for hit in hits:
                resFP.write(f"{query.id}\t{hit.name}\t{hit.identity}\t{hit.matches}\t{hit.fragments}\n")
        resFP.close()

        targetFP = open(self.cacheFile, 'at')

        thisName = None
        thisOffset = self.cachedSamples["nextOffset"] if "nextOffset" in self.cachedSamples else 0
        nextOffset = thisOffset
        alignmentCount = 0


        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                sampleName = terms[0]
                if (sampleName != thisName and thisName is not None):
                    # update last sample
                    self.cachedSamples[thisName] = [thisOffset, alignmentCount]
                    thisOffset = nextOffset
                    alignmentCount = 0

                thisName = sampleName
                nextOffset += len(line)
                alignmentCount += 1
                targetFP.write(line)

            
            if (thisName is not None):
                # update last sample
                self.cachedSamples[thisName] = [thisOffset, alignmentCount]
                self.cachedSamples["nextOffset"] = nextOffset
        
        for sample in samples:
            if (sample.id not in self.cachedSamples):
                self.cachedSamples[sample.id] = "N/A"

        targetFP.close()
        os.remove(resultFile)


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
            self.ani(samplesToRun)
        
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->ANIResult:
        # use baseName to cache the result in the results dict
        if (self.baseName in sample.results):
            return sample.results[self.baseName]
        
        # result = ANIResult()
        results:list[ANIResult] = []
        resultIndex = self.cachedSamples[sample.id]
        if (resultIndex != "N/A"):
            offset, alignmentCount = resultIndex
            cachedResultFP.seek(offset)
            alignments:list[ANIAlignment] = [ANIAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
            alignments = sorted(alignments, key=lambda x:x.overallIdentity, reverse=True)
            
            for alignment in alignments:
                if (alignment.ref is not None):
                    r = ANIResult()
                    r.addAlignment(alignment)
                    results.append(r)
                    # result.addAlignment(alignment)

        if (len(results) == 0):
            results = None
            sample.info["ANI"] = 0
            sample.info["Overall ANI"] = 0
        else:
            sample.info["ANI"] = results[0].bestAlignment.identity
            sample.info["Overall ANI"] = results[0].bestAlignment.overallIdentity
        sample.results[self.baseName] = results
        return results