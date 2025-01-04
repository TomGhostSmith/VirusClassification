import os
import subprocess

from config import config
from prototype.module import Module
from moduleConfig.minimapConfig import minimapConfig
from moduleResult.minimapResult import MinimapResult
from moduleResult.alignment import Alignment

class Minimap(Module):
    def __init__(self):
        super().__init__()

    def run(self):
        queryFile = f"{config.datasetBase}/{config.datasetName}.fasta"
        resultFolder = f"{config.resultBase}/minimapResult-{minimapConfig.name}"

        os.makedirs(resultFolder)
        command = minimapConfig.getMinimapCommand(queryFile)
        with open(f"{config.tempFolder}/alignment.sam", 'wt') as fp:
            subprocess.run(command, shell=True, stdout=fp, stderr=subprocess.DEVNULL)
        with open(f"{config.tempFolder}/alignment.sam") as fp:
            for line in fp:
                sampleName = line.split('\t')[0]
                with open(f"{resultFolder}/{sampleName}.sam", 'at') as f:
                    f.write(line)
        os.remove(f"{config.tempFolder}/alignment.sam")
    
    def getResults(self, sampleList):
        resultFolder = f"{config.resultBase}/minimapResult-{minimapConfig.name}"
        if (not os.path.exists(resultFolder)):
            self.run()
        
        for sample in sampleList:
            if (os.path.exists(f"{resultFolder}/{sample.query}.sam")):
                result = MinimapResult()
                with open(f"{resultFolder}/{sample.query}.sam") as fp:
                    reads = fp.readlines()
                for read in reads:
                    terms = read.strip().split('\t')
                    referenceName = terms[2]
                    CIGAR = terms[5]
                    mappingQuality = int(terms[4])
                    if (referenceName != '*'):
                        result.addAlignment(Alignment(referenceName, mappingQuality, CIGAR))
                if (result.bestAlignment is not None):
                    sample.addResult("minimap", result)

minimap = Minimap()