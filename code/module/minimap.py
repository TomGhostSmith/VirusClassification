import os
import subprocess

from config import config
from prototype.module import Module
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment

from utils import IOUtils

class Minimap(Module):
    def __init__(self, reference, mode='ont', threads=12, skipComments=True):
        self.reference=reference
        self.mode = mode
        self.threads = threads
        self.skipComments = skipComments
        super().__init__(f'minimap-ref={self.reference};mode={self.mode}')
        self.baseName = self.moduleName  # do not use 'self.moduleName' in code directly, in case of subClass!


    def run(self):
        queryFile = f"{config.datasetBase}/{config.datasetName}.fasta"
        resultFolder = f"{config.resultBase}/minimapResult-{self.baseName}"

        if (os.path.exists(resultFolder)):
            # IOUtils.showInfo(f'Skipped minimap on {config.datasetName}')
            return
        
        os.makedirs(resultFolder)
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
    
    def getResult(self, sample):
        resultFolder = f"{config.resultBase}/minimapResult-{self.baseName}"

        if (self.baseName in sample.results):
            return (sample.results[self.baseName])
        
        result = MinimapResult()
        if (os.path.exists(f"{resultFolder}/{sample.query}.sam")):
            with open(f"{resultFolder}/{sample.query}.sam") as fp:
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