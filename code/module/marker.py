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
from moduleResult.markerResult import MarkerResult
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from entity.taxoTree import taxoTree

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class Marker(Module):
    def __init__(self, reference, method, threads=multiprocessing.cpu_count(), threshRank='species', coverage=None, identity=50, thresh=0.5, queryMethod="sensitive"):
        if (method not in ["sum", "vote"] and not method.startswith("top")):
            raise ValueError("Unsupported pooling method")
        if (queryMethod not in ["sensitive", "coverage_identity"]):
            raise ValueError("Unsupported query method")
        self.method = method
        self.reference=reference
        self.threshRank = threshRank
        self.identity = identity
        self.coverage = coverage
        self.threads = threads
        self.thresh = thresh
        self.queryMethod = queryMethod
        super().__init__(f'marker-ref={self.reference};coverage={coverage};identity={identity};queryMethod={queryMethod};method={self.method};thresh={self.threshRank}_{self.thresh}')
        self.baseName = f'marker-ref={self.reference};coverage={coverage};identity={identity};queryMethod={queryMethod}'  # do not use 'self.moduleName' in code directly, in case of subClass!

        self.cacheFile = f"{config.cacheResultFolder}/{self.baseName}.tmp"
        self.cacheIndex = f"{config.cacheResultFolder}/{self.baseName}.json"

        self.cachedSamples:dict[str, tuple[int, int]] = dict()  # note: the cached samples are protein-level results
        self.cachedSampleNameFile = f"{config.cacheResultFolder}/{self.baseName}.names"
        self.cachedSampleNames = set()

        self.referenceDB = f"{config.cacheResultFolder}/{self.reference}_c{coverage}_i{identity}_marker.dmnd"
        self.markerDB = f"{config.cacheResultFolder}/{self.reference}_c{coverage}_i{identity}_marker.json"

        self.LCAs:dict[str, str] = {}
    
    def buildDB(self):
        # build diamond DB
        IOUtils.showInfo(f"Making diamond database for {self.reference}")
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        referenceProteinFasta = f"{config.cacheResultFolder}/{self.reference}.faa"
        refSamples = IOUtils.loadSamples(referenceFasta)
        NucleotideUtils.extractProtein(refSamples)
        IOUtils.writeSampleProteinFasta(refSamples, referenceProteinFasta)
        subprocess.run(f"diamond makedb --in {referenceProteinFasta} -d {self.referenceDB}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)        

        # build marker DB
        # key: protein id
        # value: a list of GT species contains this protein ID
        occurSet:dict[str, set[str]] = dict()

        # step 1: assume all genes are unique
        for sample in refSamples:
            speciesNode = taxoTree.getTaxoNodeFromAccession(sample.id).ICTVNode
            for protein in sample.proteins:
                occurSet[protein.id] = {speciesNode}

        # step 2: perform ref-ref alignment by diamond
        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"
        IOUtils.showInfo(f"Begin self-diamond on {self.reference}")
        IOUtils.writeSampleProteinFasta(refSamples, queryFile)
        command = self.getBlastCommandForMarker(queryFile, resultFile)
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # step 3: collect results, merge similar genes
        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                queryID = terms[0]   # protein ID
                refID = terms[1]     # protein ID
                score = float(terms[2])
                if (score > self.identity/100):
                    occurSet[queryID] |= occurSet[refID]
                    occurSet[refID] = occurSet[queryID]  # now they share the same addr in mem
        
        # step 4: calculate LCA for each protein
        self.LCAs = {}
        for proteinID, nodeset in occurSet.items():
            LCANode = taxoTree.ICTVTree.findLCA(nodeset)
            self.LCAs[proteinID] = LCANode.name
        
        with open(self.markerDB, 'wt') as fp:
            json.dump(self.LCAs, fp, indent=2)

                

    def diamond(self, samples:list[Sample]):
        if (not os.path.exists(self.referenceDB)) or (not os.path.exists(self.markerDB)):
            self.buildDB()

        queryFile = f"{config.cacheFolder}/blast.fasta"
        resultFile = f"{config.cacheFolder}/blast.tsv"

        IOUtils.showInfo(f"Begin diamond on {len(samples)} samples")

        IOUtils.writeSampleProteinFasta(samples, queryFile)

        command = self.getBlastCommandForQuery(queryFile, resultFile)
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


    def run(self, samples:list[Sample], keepVotes=False, keepProteinRes=False, withProteinMeta=False, withProteinCandidateMeta=False, **kwargs):
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
        
        with open(self.markerDB) as fp:
            self.LCAs = json.load(fp)

        cachedResultFP = open(self.cacheFile)
        results = [self.getResult(sample, cachedResultFP, keepVotes, keepProteinRes, withProteinMeta, withProteinCandidateMeta) for sample in samples]
        cachedResultFP.close()

        return results
    
    def getResult(self, sample:Sample, cachedResultFP, keepVotes, keepProteinRes, withProteinMeta, withProteinCandidateMeta)->PlainResult:
        # note: result of basename is not available
        result = None

        keepProteinRes = True
        
        votes:dict[str, int] = {}
        if (self.method == "vote"):
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = sorted(alignments, key=cmp_to_key(lambda a, b: -1 if a.betterThan(b) else (1 if b.betterThan(a) else 0)))
                if (withProteinMeta):
                    if (len(alignments) > 0):
                        markerName = self.LCAs[alignments[0].ref]
                        markerRank = taxoTree.ICTVTree.nodes[markerName].rank
                        protein.info["bestMarker"] = config.rankLevels[markerRank]
                        markerNodes = [taxoTree.ICTVTree.nodes[self.LCAs[a.ref]] for a in alignments]
                        rankLCA = taxoTree.ICTVTree.findLCA(markerNodes).rank
                        if (rankLCA.startswith("sub")):
                            normalizedRankLCA = rankLCA[3:]
                        else:
                            normalizedRankLCA = rankLCA
                            
                        protein.info["markerRank"] = config.rankLevels[rankLCA]
                        protein.info["normalizedMarkerRank"] = config.rankLevels[normalizedRankLCA]
                        protein.info["markerRankBin"] = getRankBin(normalizedRankLCA)
                        protein.info["bin"] = ""
                if (keepProteinRes):
                    res = []
                    for a in alignments:
                        markerName = self.LCAs[a.ref]
                        r = MarkerResult(a, markerName)
                        if (withProteinCandidateMeta):
                            markerRank = taxoTree.ICTVTree.nodes[markerName].rank
                            r.info["rank"] = config.rankLevels[markerRank]
                        res.append(r)
                    if (len(res) == 0):
                        res = None
                    protein.addResult(self.moduleName, res)
                if (len(alignments) > 0):
                    bestAlignment = alignments[0]
                    ICTVName = self.LCAs[bestAlignment.ref]
                    if ICTVName in votes:
                        votes[ICTVName] += 1
                    else:
                        votes[ICTVName] = 1
            
        elif (self.method == "sum" or self.method.startswith("top")):
            if (self.method.startswith("top")):
                thresh = int(self.method[3:])
            else:
                thresh = None
            for protein in sample.proteins:
                offset, alignmentCount = self.cachedSamples[protein.id]
                cachedResultFP.seek(offset)
                alignments:list[DiamondAlignment] = [DiamondAlignment(cachedResultFP.readline()) for _ in range(alignmentCount)]
                alignments = sorted(alignments, key=cmp_to_key(lambda a, b: -1 if a.betterThan(b) else (1 if b.betterThan(a) else 0)))
                if (withProteinMeta):
                    if (len(alignments) > 0):
                        markerName = self.LCAs[alignments[0].ref]
                        markerRank = taxoTree.ICTVTree.nodes[markerName].rank
                        protein.info["bestMarker"] = config.rankLevels[markerRank]
                        markerNodes = [taxoTree.ICTVTree.nodes[self.LCAs[a.ref]] for a in alignments]
                        rankLCA = taxoTree.ICTVTree.findLCA(markerNodes).rank
                        if (rankLCA.startswith("sub")):
                            normalizedRankLCA = rankLCA[3:]
                        else:
                            normalizedRankLCA = rankLCA
                    protein.info["markerRank"] = config.rankLevels[rankLCA]
                    protein.info["normalizedMarkerRank"] = config.rankLevels[normalizedRankLCA]
                    protein.info["markerRankBin"] = getRankBin(normalizedRankLCA)
                if (keepProteinRes):
                    res = []
                    for a in alignments:
                        markerName = self.LCAs[a.ref]
                        r = MarkerResult(a, markerName)
                        if (withProteinCandidateMeta):
                            markerRank = taxoTree.ICTVTree.nodes[markerName].rank
                            r.info["rank"] = config.rankLevels[markerRank]
                        res.append(r)
                    if (len(res) == 0):
                        res = None
                    protein.addResult(self.moduleName, res)
                for alignment in alignments[:thresh]:
                    ICTVName = self.LCAs[alignment.ref]
                    if ICTVName in votes:
                        votes[ICTVName] += alignment.similarity
                    else:
                        votes[ICTVName] = alignment.similarity
        if len(votes) > 0:
            if (self.threshRank == 'vitax'):
                threshold = self.thresh
                scores = {taxoTree.ICTVTree.nodes[k]: v for k, v in votes.items()}
                highestScore = 0
                highestNode = None
                for th in reversed(range(config.rankLevels["superkingdom"], config.rankLevels["species"] + 1)):
                    for node, s in scores.items():
                        totalScore = 0
                        for n in node.path:  # include itself
                            if (n in scores):
                                totalScore += scores[n]
                        thisScore = totalScore / len(sample.proteins)
                        if (thisScore >= threshold and thisScore > highestScore):
                            highestScore = thisScore
                            highestNode = node
                            
                    if (highestNode is not None):  # if there is already some node above thresh, then return
                        result = [PlainResult(highestNode.name, score=highestScore)]
                        return result


                    originScores = scores
                    scores = dict()
                    for node, s in originScores.items():
                        if (config.rankLevels[node.rank] > th):  # merge current score to its parent
                            target = node.parent
                        else:
                            target = node
                        if (target in scores):
                            scores[target] += s
                        else:
                            scores[target] = s
            elif (self.threshRank == "bottomup"):
                newVotes = {r: {} for r in reversed(config.resultRanks)}
                for name, vote in votes.items():
                    for n in taxoTree.ICTVTree.nodes[name].path:
                        if n.rank in newVotes:
                            if (n.name not in newVotes[n.rank]):
                                newVotes[n.rank][n.name] = vote
                            else:
                                newVotes[n.rank][n.name] += vote
                for rank, rankVotes in newVotes.items():
                    if (len(rankVotes) > 0):
                        winner, maxVotes = max(rankVotes.items(), key=lambda x: x[1])
                        if (maxVotes > self.thresh * len(sample.proteins)):
                            result = [PlainResult(winner, score=maxVotes / (len(sample.proteins)))]
                            break
                # else, if in no rank maxVote is higher than thresh, then keep result=None
            elif (self.threshRank == "bottomup_ratio"):
                newVotes = {r: {} for r in reversed(config.resultRanks)}
                for name, vote in votes.items():
                    for n in taxoTree.ICTVTree.nodes[name].path:
                        if n.rank in newVotes:
                            if (n.name not in newVotes[n.rank]):
                                newVotes[n.rank][n.name] = vote
                            else:
                                newVotes[n.rank][n.name] += vote
                for rank, rankVotes in newVotes.items():
                    if (len(rankVotes) > 1):
                        (top1Name, top1Votes), (top2Name, top2Votes) = sorted(list(rankVotes.items()), key=lambda x:x[1], reverse=True)[:2]
                        if (top1Votes > top2Votes * self.thresh):
                            result = [PlainResult(top1Name, score=top1Votes / len(sample.proteins))]
                            break
            else:
                totalVotes = sum(votes.values())
                pairs = sorted(votes.items(), key=lambda x:x[1], reverse=True)
                result = []
                for name, v in pairs:
                    winnerNode = taxoTree.ICTVTree.nodes[name]
                    for n in reversed(winnerNode.path):
                        if (config.rankLevels[n.rank] <= config.rankLevels[self.threshRank] ):
                            result.append(PlainResult(n.name, score=v/totalVotes))
                            break
                if (len(result) == 0):
                    result = None

        if (keepVotes):
            sample.info[f"{self.moduleName}_votes"] = votes
        
        return result
    
    def getBlastCommandForMarker(self, queryFile, resultFile):
        identityTh = f" --id {self.identity}"
        coverageTh = f" --query-cover {self.coverage} --subject-cover {self.coverage}" if self.coverage else ""
        command = f"diamond blastp -q {queryFile} -d {self.referenceDB} -o {resultFile} -f 6 -k 0 -p {self.threads} --block-size 20{identityTh}{coverageTh}"
        return command
    
    def getBlastCommandForQuery(self, queryFile, resultFile):
        if (self.queryMethod == "sensitive"):
            command = f"diamond blastp -q {queryFile} -d {self.referenceDB} -o {resultFile} -f 6 -k 0 -p {self.threads} --block-size 20 --more-sensitive --evalue 1e-3"
        else:
            identityTh = f" --id {self.identity}"
            coverageTh = f" --query-cover {self.coverage} --subject-cover {self.coverage}" if self.coverage else ""
            command = f"diamond blastp -q {queryFile} -d {self.referenceDB} -o {resultFile} -f 6 -k 0 -p {self.threads} --block-size 20 {identityTh}{coverageTh}"
        return command
    
    def getLCAs(self):
        if len(self.LCAs) == 0:
            with open(self.markerDB) as fp:
                self.LCAs = json.load(fp)
        return self.LCAs

def getRankBin(rank):
    if rank in ["species", "genus"]:
        return "specific"
    if rank in ["family", "order"]:
        return "medium"
    return "coarse"