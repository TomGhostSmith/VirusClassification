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
from moduleResult.diamondAlignment import DiamondAlignment
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from entity.taxoTree import taxoTree

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class Diamond(Module):
    def __init__(self, reference, method, threads=multiprocessing.cpu_count(), threshRank='species', tool="diamond"):
        if (method not in ["sum", "vote", "coverage"] and not method.startswith("top")):
            raise ValueError("Unsupported pooling method")
        if (tool not in ["diamond", "mmseqs"]):
            raise ValueError("Unsupported tool")
        self.tool = tool
        self.method = method
        self.reference=reference
        self.threshRank = threshRank
        self.threads = threads
        super().__init__(f'diamond-ref={self.reference};tool={self.tool};method={self.method};thresh={self.threshRank}')
        self.baseName = f'diamond-ref={self.reference};tool={self.tool}'  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()  # note: the cached samples are protein-level results
        self.cachedSampleNameFile = f"{config.cacheResultFolder}/{self.baseName}.names"
        self.cachedSampleNames = set()

        if (self.tool == "diamond"):
            self.referenceDB = f"{config.cacheResultFolder}/{self.reference}_diamonddb.dmnd"
        elif (self.tool == "mmseqs"):
            self.referenceDB = f"{config.cacheResultFolder}/{self.reference}_prot_mmseqdb"
    
    def buildDB(self):
        IOUtils.showInfo(f"Making diamond database for {self.reference}")
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        referenceProteinFasta = f"{config.cacheResultFolder}/{self.reference}.faa"
        refSamples = IOUtils.loadSamples(referenceFasta)
        NucleotideUtils.extractProtein(refSamples)
        IOUtils.writeSampleProteinFasta(refSamples, referenceProteinFasta)
        if (self.tool == "diamond"):
            cmd = f"diamond makedb --in {referenceProteinFasta} -d {self.referenceDB}"
        else:
            cmd = f"mmseqs createdb {referenceProteinFasta} {self.referenceDB}"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def diamond(self, samples:list[Sample]):
        if not os.path.exists(self.referenceDB):
            self.buildDB()

        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"

        IOUtils.showInfo(f"Begin diamond on {len(samples)} samples")

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
            self.diamond(samplesToRun)
                    
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
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                if (len(alignments) > 0):
                    bestAlignment = alignments[0]
                    for alignment in alignments[1:]:
                        if alignment.betterThan(bestAlignment):
                            bestAlignment = alignment
                    ICTVName = taxoTree.getTaxoNodeFromAccession(bestAlignment.refContig).ICTVName
                    if ICTVName in votes:
                        votes[ICTVName] += 1
                    else:
                        votes[ICTVName] = 1
        elif (self.method == "coverage"):
            accessionVotes = dict()
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                for alignment in alignments:
                    accession = alignment.refContig
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
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                for alignment in alignments:
                    ICTVName = taxoTree.getTaxoNodeFromAccession(alignment.refContig).ICTVName
                    if ICTVName in votes:
                        votes[ICTVName] += alignment.similarity
                    else:
                        votes[ICTVName] = alignment.similarity
        elif (self.method.startswith("top")):
            thresh = int(self.method[3:])
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = sorted(alignments, key=cmp_to_key(lambda a, b: -1 if a.betterThan(b) else (1 if b.betterThan(a) else 0)))
                for alignment in alignments[:thresh]:
                    ICTVName = taxoTree.getTaxoNodeFromAccession(alignment.refContig).ICTVName
                    if ICTVName in votes:
                        votes[ICTVName] += alignment.similarity
                    else:
                        votes[ICTVName] = alignment.similarity
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
        else:
            maxVotes = 0


        proteinCount = len(sample.proteins)
        if (proteinCount == 1 and result is None):
            proteinCount = 0.5
            
        sample.info["protein_count_match"] = proteinCount
        sample.info["protein_count"] = len(sample.proteins)
        sample.info["protein_match_ratio"] = maxVotes / len(sample.proteins) if len(sample.proteins) > 0 else 0

        
        return result
    
    def getBlastCommand(self, queryFile, resultFile):
        # first check if the reference fasta is made a database

        # cline = NcbiblastnCommandline(query=queryFile, db=referenceDB, evalue=1e-3, outfmt=5, out=resultFile)
        # stdout, stderr = cline()
        if (self.tool == "diamond"):
            command = f"diamond blastp -q {queryFile} -d {self.referenceDB} -o {resultFile} -f 6 -k 0 -p {self.threads} --block-size 20"
        elif (self.tool == "mmseqs"):
            command = f"mmseqs easy-search {queryFile} {self.referenceDB} {resultFile} /tmp --threads {self.threads} -s 10 --search-type 0"
        return command