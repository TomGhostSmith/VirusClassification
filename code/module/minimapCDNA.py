# reconstructing
import os
import json
import math
import subprocess
from functools import cmp_to_key
import multiprocessing

from config import config
from prototype.module import Module
from moduleResult.plainResult import PlainResult
from moduleResult.cDNAAlignment import CDNAAlignment
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from entity.taxoTree import taxoTree

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class MinimapCDNA(Module):
    def __init__(self, reference, method,  mode='simple', threads=multiprocessing.cpu_count(), threshRank='species', skipComments=True):
        if (method not in ["sum", "vote", "coverage"] and not method.startswith("top")):
            raise ValueError("Unsupported pooling method")
        self.skipComments = skipComments
        self.method = method
        self.mode = mode
        self.reference=reference
        self.threshRank = threshRank
        self.threads = threads
        super().__init__(f'cDNAminimap-ref={self.reference};method={self.method};thresh={self.threshRank}')
        self.baseName = f'cDNAminimap-ref={self.reference}'  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()  # note: the cached samples are protein-level results
        self.cachedSampleNameFile = f"{config.cacheResultFolder}/{self.baseName}.names"
        self.cachedSampleNames = set()

        self.db = f"{config.cacheResultFolder}/{self.baseName}.db.fasta"

    def buildDB(self):
        IOUtils.showInfo(f"Making cDNA minimap database for {self.reference}")
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        refSamples = IOUtils.loadSamples(referenceFasta)
        NucleotideUtils.extractProtein(refSamples)
        IOUtils.writeSampleCDNAFasta(refSamples, self.db)


    def minimap(self, samples:list[Sample]):
        if (not os.path.exists(self.db)):
            self.buildDB()
        queryFile = f"{config.cacheFolder}/minimap.fasta"
        resultFile = f"{config.cacheFolder}/alignment.sam"

        IOUtils.showInfo(f"Begin cDNA minimap on {len(samples)} samples")

        IOUtils.writeSampleCDNAFasta(samples, queryFile)

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
                if (line.startswith("[")):
                    continue
                terms = line.strip().split('\t')
                if (len(terms) < 10):
                    IOUtils.showInfo(f"The output of the minimap is not standard. Expect at least 10 columns, got {len(terms)}", "ERROR")
                    IOUtils.showInfo(f"The error line is '{line}'")
                    exit(-1)
                terms[9] = '*'
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
        
        # if there is no alignment, then the query won't show up in the output file
        for sample in samples:
            for protein in sample.cDNAs:
                if protein.id not in self.cachedSamples:
                    self.cachedSamples[protein.id] = [0, 0]

            self.cachedSampleNames.add(sample.id)

        targetFP.close()
        os.remove(resultFile)
        os.remove(queryFile)


    def run(self, samples:list[Sample], **kwargs):
        raise NotImplementedError("not reconstruct for multi-result")
        samplesToRun:list[Sample] = list()
        NucleotideUtils.extractProtein(samples)

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
            self.minimap(samplesToRun)
                    
            with open(self.cacheIndex, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)
            
            with open(self.cachedSampleNameFile, 'wt') as fp:
                json.dump(list(self.cachedSampleNames), fp, indent=2)

        if (self.method == "coverage"):
            if (not os.path.exists(f"{config.cacheResultFolder}/c2p.json")):
                self.buildDB()
            with open(f"{config.cacheResultFolder}/c2p.json") as fp:
                self.c2p = json.load(fp)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP)->PlainResult:
        # note: result of basename is not available
        result = None
        
        votes:dict[str, int] = dict()
        if (self.method == "vote"):
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[CDNAAlignment] = [CDNAAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = [a for a in alignments if a.ref is not None]
                if (len(alignments) > 0):
                    bestAlignment = alignments[0]
                    for alignment in alignments[1:]:
                        if alignment.betterThan(bestAlignment):
                            bestAlignment = alignment
                    ICTVName = taxoTree.getTaxoNodeFromAccession(bestAlignment.ref).ICTVName
                    if ICTVName in votes:
                        votes[ICTVName] += 1
                    else:
                        votes[ICTVName] = 1
        elif (self.method == "coverage"):
            accessionVotes = dict()
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[CDNAAlignment] = [CDNAAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = [a for a in alignments if a.ref is not None]
                for alignment in alignments:
                    accession = alignment.ref
                    if (accession in accessionVotes):
                        accessionVotes[accession] += 1/len(self.c2p[accession])
                    else:
                        accessionVotes[accession] = 1/len(self.c2p[accession])
            for accession, coverage in accessionVotes.items():
                ICTVName = taxoTree.getTaxoNodeFromAccession(accession).ICTVName
                if ICTVName in votes:
                    votes[ICTVName] = max(coverage, votes[ICTVName])  # the vote is the maximum coverage for that species
                else:
                    votes[ICTVName] = coverage
        elif (self.method == "sum"):
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[CDNAAlignment] = [CDNAAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = [a for a in alignments if a.ref is not None]
                for alignment in alignments:
                    ICTVName = taxoTree.getTaxoNodeFromAccession(alignment.ref).ICTVName
                    if ICTVName in votes:
                        votes[ICTVName] += alignment.quality/60
                    else:
                        votes[ICTVName] = alignment.quality/60
        elif (self.method.startswith("top")):
            thresh = int(self.method[3:])
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[CDNAAlignment] = [CDNAAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = [a for a in alignments if a.ref is not None]
                alignments = sorted(alignments, key=cmp_to_key(lambda a, b: -1 if a.betterThan(b) else (1 if b.betterThan(a) else 0)))
                for alignment in alignments[:thresh]:
                    if (alignment.ref is None):
                        continue
                    ICTVName = taxoTree.getTaxoNodeFromAccession(alignment.ref).ICTVName
                    if ICTVName in votes:
                        votes[ICTVName] += alignment.quality/60
                    else:
                        votes[ICTVName] = alignment.quality/60
        if len(votes) > 0:
            totalVotes = sum(votes.values())
            winner, maxVotes = max(votes.items(), key=lambda x: x[1])
            winnerNode = taxoTree.ICTVTree.nodes[winner]
            for n in reversed(winnerNode.path):
                if (config.rankLevels[n.rank] <= config.rankLevels[self.threshRank] ):
                    result = PlainResult(n.name, score=maxVotes/totalVotes)
                    break
            if result is None:
                result = PlainResult(winner, score=maxVotes/totalVotes)

        
        return result
    
    def getMinimapCommand(self, queryFile):
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
        command = f"{minimapBase} {mode} {self.db} {queryFile} {thread} {postProcess}"
        return command