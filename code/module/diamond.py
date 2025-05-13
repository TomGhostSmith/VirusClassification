# reconstructing
import os
import json
import math
import subprocess
import multiprocessing

from config import config
from prototype.module import Module
from moduleResult.blastResult import BlastResult
from moduleResult.blastAlignment import BlastAlignment
from entity.sample import Sample
from entity.proteinSample import ProteinSample

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class Diamond(Module):
    def __init__(self, reference, threads=12):
        self.reference=reference
        self.threads = threads
        super().__init__(f'diamond-ref={self.reference}')
        self.baseName = self.moduleName  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()  # note: the cached samples are protein-level results

        self.referenceDB = f"{config.cacheResultFolder}/{self.reference}_diamonddb"
    
    def buildDB(self):
        IOUtils.showInfo(f"Making diamond database for {self.reference}")
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        referenceProteinFasta = f"{config.cacheResultFolder}/{self.reference}.faa"
        refSamples = IOUtils.loadSamples(referenceFasta)
        NucleotideUtils.extractProtein(refSamples)
        IOUtils.writeSampleProteinFasta(refSamples, referenceProteinFasta)
        subprocess.run(f"diamond makedb --in {referenceProteinFasta} -d {self.referenceDB}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open(self.referenceMapping, 'wt') as fp:
            json.dump(self.refP2C, fp, indent=2)


    def diamond(self, samples:list[Sample]):
        if not os.path.exists(self.referenceDB):
            self.buildDB()

        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"

        IOUtils.showInfo(f"Begin blast on {len(samples)} samples")
        os.remove(queryFile)

        NucleotideUtils.extractProtein(samples)
        IOUtils.writeSampleProteinFasta(samples, queryFile)

        command = self.getBlastCommand(queryFile, resultFile)
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        
        # if there is no alignment, then the query won't show up in the output file
        for sample in samples:
            for protein in sample.proteins:
                if protein.id not in self.cachedSamples:
                    self.cachedSamples[protein.id] = [0, 0]

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
            self.diamond(samplesToRun)
                    
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getProteinResult(self, sample:ProteinSample, cachedResultFP)->BlastResult:
        # use baseName to cache the result in the results dict
        if (self.baseName in sample.results):
            return sample.results[self.baseName]
        
        offset, alignmentCount = self.cachedSamples[sample.id]
        cachedResultFP.seek(offset)
        alignments:list[BlastAlignment] = [BlastAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
        
        result = BlastResult()
        for alignment in alignments:
            if (alignment.ref is not None):
                result.addAlignment(alignment)

        if (result.bestAlignment is None):
            result = None
        sample.results[self.baseName] = result
        return result
    
    def getBlastCommand(self, queryFile, resultFile):
        # first check if the reference fasta is made a database

        # cline = NcbiblastnCommandline(query=queryFile, db=referenceDB, evalue=1e-3, outfmt=5, out=resultFile)
        # stdout, stderr = cline()
        command = f"diamond blastp -q {queryFile} -d {self.referenceDB} -o {resultFile} -f 6 -k 0 -p {self.threads}"
        return command