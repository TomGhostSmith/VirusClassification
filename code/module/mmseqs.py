# reconstructed
import os
import json
import subprocess
from Bio import SeqIO
import multiprocessing

from config import config
from prototype.module import Module
from moduleResult.blastResult import BlastResult
from moduleResult.blastAlignment import BlastAlignment
from entity.sample import Sample

from utils import IOUtils

class MMseqs(Module):
    def __init__(self, reference, sensitivity, coverage, identity, threads=multiprocessing.cpu_count()):
        self.reference=reference
        self.sensitivity = sensitivity
        self.coverage = coverage
        self.identity = identity
        self.threads = threads
        super().__init__(f'mmseqs-ref={self.reference};s{sensitivity};c{coverage};i{identity}')
        self.baseName = f'mmseqs-ref={self.reference};s{sensitivity}'  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()

    def mmseq(self, samples):
        queryFile = f"{config.cacheFolder}/mmseq.fasta"
        resultFile = f"{config.cacheFolder}/alignment.tsv"
        IOUtils.writeSampleFasta(samples, queryFile)
        IOUtils.showInfo(f"Begin mmseqs on {len(samples)} samples")

        command = self.getMMseqsCommand(queryFile, resultFile)
        subprocess.run(command, shell=True)

        targetFP = open(self.cacheFile, 'at')

        thisName = None
        thisOffset = self.cachedSamples["nextOffset"] if "nextOffset" in self.cachedSamples else 0
        nextOffset = thisOffset
        alignmentCount = 0


        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip('\n').split('\t')
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
            self.mmseq(samplesToRun)
        
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
        targetLen = sample.length * self.coverage / 100
        for alignment in alignments:
            if (alignment.ref is not None and alignment.queryCoverLength >= targetLen and alignment.similarity >= self.identity/100):
                result.addAlignment(alignment)

        if (result.bestAlignment is None):
            result = None
            sample.info["mmseq"] = 0
        else:
            sample.info["mmseq"] = result.bestAlignment.similarity
        sample.results[self.baseName] = result
        return result
    
    def getMMseqsCommand(self, queryFile, resultFile):
        # first check if the reference fasta is made a database
        referenceDB = f"{config.cacheResultFolder}/{self.reference}_mmseqdb"
        dbFiles = [referenceDB, f"{referenceDB}_h", f"{referenceDB}_h.dbtype", f"{referenceDB}_h.index", f"{referenceDB}.dbtype", f"{referenceDB}.index", f"{referenceDB}.lookup", f"{referenceDB}.source"]
        dbvalid = True
        for dbFile in dbFiles:
            if not os.path.exists(dbFile):
                dbvalid = False
                break
        if not dbvalid:
            referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
            IOUtils.showInfo(f"Making mmseq database for {self.reference}")
            cmd = f"mmseqs createdb {referenceFasta} {referenceDB}"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # cline = NcbiblastnCommandline(query=queryFile, db=referenceDB, evalue=1e-3, outfmt=5, out=resultFile)
        # stdout, stderr = cline()
        command = f"mmseqs easy-search {queryFile} {referenceDB} {resultFile} /tmp --threads {self.threads} -s {self.sensitivity} --search-type 3"
        return command