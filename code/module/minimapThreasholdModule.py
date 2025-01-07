import os
import subprocess

from config import config
from prototype.module import Module
from moduleConfig.minimapThreasholdConfig import MinimapThresholdConfig
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment

from utils import IOUtils

class MinimapThreasholdModule(Module):
    def __init__(self, config:MinimapThresholdConfig):
        super().__init__(config)
        self.minimapThConfig = config

    def run(self):
        queryFile = f"{config.datasetBase}/{config.datasetName}.fasta"
        resultFolder = f"{config.resultBase}/minimapResult-{self.minimapThConfig.baseName}"

        if (os.path.exists(resultFolder)):
            IOUtils.showInfo(f'Skipped minimap on {config.datasetName}')
            return
        
        os.makedirs(resultFolder)
        IOUtils.showInfo(f"Begin minimap on {config.datasetName}")
        command = self.minimapThConfig.getMinimapCommand(queryFile)
        with open(f"{config.tempFolder}/alignment.sam", 'wt') as fp:
            subprocess.run(command, shell=True, stdout=fp, stderr=subprocess.DEVNULL)
        with open(f"{config.tempFolder}/alignment.sam") as fp:
            for line in fp:
                sampleName = line.split('\t')[0]
                with open(f"{resultFolder}/{sampleName}.sam", 'at') as f:
                    f.write(line)
        os.remove(f"{config.tempFolder}/alignment.sam")
    
    def getResult(self, sample):
        resultFolder = f"{config.resultBase}/minimapResult-{self.minimapThConfig.baseName}"
        
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
        else:
            if ("positive" in self.minimapThConfig.factors and result.bestAlignment.quality == 0):
                result = None
            elif ("60" in self.minimapThConfig.factors and result.bestAlignment.quality < 60):
                result = None
            elif ("completeMatch" in self.minimapThConfig.factors and result.bestAlignment.queryCoverLength < sample.length):
                result = None
            elif ("singleAlignment" in self.minimapThConfig.factors and len(result.alignments) > 1):
                result = None
        
        return result