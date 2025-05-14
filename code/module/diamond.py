# reconstructing
import os
import json
import math
import subprocess
import multiprocessing

from config import config
from prototype.module import Module
from moduleResult.plainResult import PlainResult
from moduleResult.diamondAlignment import DiamondAlignment
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from entity.taxoTree import taxoTree

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class Diamond(Module):
    def __init__(self, reference, method, threads=multiprocessing.cpu_count()):
        if (method not in ["sum", "vote"]):
            raise ValueError("Unsupported pooling method")
        self.method = method
        self.reference=reference
        self.threads = threads
        super().__init__(f'diamond-ref={self.reference};method={self.method}')
        self.baseName = f'diamond-ref={self.reference}'  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()  # note: the cached samples are protein-level results
        self.cachedSampleNameFile = f"{config.cacheResultFolder}/{self.baseName}.names"
        self.cachedSampleNames = set()

        self.referenceDB = f"{config.cacheResultFolder}/{self.reference}_diamonddb.dmnd"
    
    def buildDB(self):
        IOUtils.showInfo(f"Making diamond database for {self.reference}")
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        referenceProteinFasta = f"{config.cacheResultFolder}/{self.reference}.faa"
        refSamples = IOUtils.loadSamples(referenceFasta)
        NucleotideUtils.extractProtein(refSamples)
        IOUtils.writeSampleProteinFasta(refSamples, referenceProteinFasta)
        subprocess.run(f"diamond makedb --in {referenceProteinFasta} -d {self.referenceDB}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        


    def diamond(self, samples:list[Sample]):
        if not os.path.exists(self.referenceDB):
            self.buildDB()

        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"

        IOUtils.showInfo(f"Begin diamond on {len(samples)} samples")

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

            self.cachedSampleNames.add(sample.id)

        targetFP.close()
        os.remove(resultFile)
        os.remove(queryFile)


    def run(self, samples:list[Sample]):
        samplesToRun:list[Sample] = list()

        if (os.path.exists(self.cacheIndex) and os.path.exists(self.cachedSampleNameFile)):
            with open(self.cacheIndex) as fp:
                self.cachedSamples = json.load(fp)  # id: [offset, alignmentCount]
            
            with open(self.cachedSampleNameFile) as fp:
                self.cachedSampleNames = set(json.load(fp))
        
            for sample in samples:
                if (sample.id not in self.cachedSampleNames):
                    samplesToRun.append(sample)
        else:
            samplesToRun = samples
        
        if (len(samplesToRun) > 0):
            self.diamond(samplesToRun)
                    
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)
            
            with open(self.cachedSampleNameFile, 'wt') as fp:
                json.dump(list(self.cachedSampleNames), fp, indent=2)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->PlainResult:
        # note: result of basename is not available
        result = None
        
        if (self.method == "vote"):
            votes:dict[str, int] = dict()
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                if (len(alignments) > 0):
                    bestAlignment = alignments[0]
                    for alignment in alignments[1:]:
                        if alignment.betterThan(bestAlignment):
                            bestAlignment = alignment
                    ICTVID = taxoTree.ICTVTree.accession2ID[bestAlignment.refContig]
                    if ICTVID in votes:
                        votes[ICTVID] += 1
                    else:
                        votes[ICTVID] = 1
            
        elif (self.method == "sum"):
            votes:dict[str, int] = dict()
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                for alignment in alignments:
                    ICTVID = taxoTree.ICTVTree.accession2ID[alignment.refContig]
                    if ICTVID in votes:
                        votes[ICTVID] += alignment.similarity
                    else:
                        votes[ICTVID] = alignment.similarity
            
        if len(votes) > 0:
            totalVotes = sum(votes.values())
            winner, maxVotes = max(votes.items(), key=lambda x: x[1])
            result = PlainResult(taxoTree.ICTVTree.ID2name[winner], score=maxVotes/totalVotes)
        
        return result
    
    def getBlastCommand(self, queryFile, resultFile):
        # first check if the reference fasta is made a database

        # cline = NcbiblastnCommandline(query=queryFile, db=referenceDB, evalue=1e-3, outfmt=5, out=resultFile)
        # stdout, stderr = cline()
        command = f"diamond blastp -q {queryFile} -d {self.referenceDB} -o {resultFile} -f 6 -k 0 -p {self.threads}"
        return command