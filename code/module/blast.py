# reconstructing
import os
import json
import subprocess
from Bio.Blast import NCBIXML

from config import config
from prototype.module import Module
from moduleResult.blastResult import BlastResult
from moduleResult.blastAlignment import BlastAlignment
from entity.sample import Sample

from utils import IOUtils

class Blast(Module):
    def __init__(self, reference, threads=12):
        self.reference=reference
        self.threads = threads
        super().__init__(f'blast-ref={self.reference}')
        self.baseName = self.moduleName  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()

    def blast(self, samples:list[Sample]):
        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"
        IOUtils.writeSampleFasta(samples, queryFile)
        IOUtils.showInfo(f"Begin blast on {len(samples)} samples")

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
            if sample.id not in self.cachedSamples:
                self.cachedSamples[sample.id] = [0, 0]

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
            self.blast(samplesToRun)
                    
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->BlastResult:
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
        referenceDB = f"{config.cacheResultFolder}/{self.reference}_blastdb"
        dbFiles = [f"{referenceDB}.nhr", f"{referenceDB}.nin", f"{referenceDB}.nsq"]
        dbvalid = True
        for dbFile in dbFiles:
            if not os.path.exists(dbFile):
                dbvalid = False
                break
        if not dbvalid:
            referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
            IOUtils.showInfo(f"Making blast database for {self.reference}")
            subprocess.run(f"makeblastdb -in {referenceFasta} -dbtype nucl -out {referenceDB}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # cline = NcbiblastnCommandline(query=queryFile, db=referenceDB, evalue=1e-3, outfmt=5, out=resultFile)
        # stdout, stderr = cline()
        command = f"blastn -query {queryFile} -db {referenceDB} -evalue 0.001 -outfmt 6 -out {resultFile} -num_threads {self.threads}"
        return command