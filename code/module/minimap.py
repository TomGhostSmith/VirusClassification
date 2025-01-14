import os
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


    def minimap(self, samples, resultFolder):
        queryFile = f"{config.tempFolder}/minimap.fasta"
        with open(queryFile, 'wt') as fp:
            for seq in samples:
                SeqIO.write(seq.seq, fp, 'fasta')

        IOUtils.showInfo(f"Begin minimap on {config.datasetName}")
        command = self.getMinimapCommand(queryFile)
        with open(f"{config.tempFolder}/alignment.sam", 'wt') as fp:
            subprocess.run(command, shell=True, stdout=fp, stderr=subprocess.DEVNULL)
        with open(f"{config.tempFolder}/alignment.sam") as fp:
            for line in fp:
                sampleName = line.split('\t')[0]
                with open(f"{resultFolder}/{sampleName}.sam", 'at') as f:
                    f.write(line)
        os.remove(f"{config.tempFolder}/alignment.sam")
        os.remove(queryFile)


    def run(self, samples):
        resultFolder = f"{config.tempFolder}/minimapResult-{self.baseName}"

        samplesToRun = list()

        if (os.path.exists(resultFolder)):
            for sample in samples:
                if (not os.path.exists(f"{resultFolder}/{sample.id}.sam")):
                    samplesToRun.append(sample)
        else:
            samplesToRun = samples
            os.makedirs(resultFolder)
        
        if (len(samplesToRun) > 0):
            self.minimap(samplesToRun, resultFolder)
                    
        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample):
        if self.baseName in sample.results:
            return sample.results[self.baseName]
        resultFolder = f"{config.resultBase}/minimapResult-{self.baseName}"
        
        result = MinimapResult()
        with open(f"{resultFolder}/{sample.id}.sam") as fp:
            reads = fp.readlines()
        for read in reads:
            terms = read.strip().split('\t')
            referenceName = terms[2]
            CIGAR = terms[5]
            mappingQuality = int(terms[4])
            if (referenceName != '*'):
                result.addAlignment(Alignment(referenceName, mappingQuality, CIGAR))
        if (result.bestAlignment is None):
            result = None
        sample.results[self.baseName] = result
        return result
    
    def getMinimapCommand(self, queryFile):
        referenceFasta = f"/Data/ICTVData/reference/{self.reference}/{self.reference}.fasta"
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