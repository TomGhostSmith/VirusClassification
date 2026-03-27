# reconstructed
import os
import json
import time
import shutil
import subprocess

from config import config
from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module
from moduleResult.plainResult import PlainResult


class VCAT(Module):
    def __init__(self):
        super().__init__(f"VCAT_VMRv4_MSL39")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()
    

    def vcat(self, samples:list[Sample])->None:

        cacheFolder = f"{config.cacheFolder}/vcat"
        inputFile = f"{cacheFolder}/input.fasta"
        outputFolder = f"{cacheFolder}/output"
        resultFile = f"{outputFolder}/results/input_fasta.tsv"

        IOUtils.showInfo(f"Begin Vcat on {len(samples)} samples")

        os.makedirs(cacheFolder, exist_ok=True)
        IOUtils.writeSampleFasta(samples, inputFile)
        time.sleep(1)
        command = f"conda run -n vcat --no-capture-output vcat contigs -i {inputFile} -o {outputFolder}"

        subprocess.run(command, shell=True, cwd=cacheFolder)

        with open(resultFile) as fp:
            fp.readline()
            for line in fp:
                terms = line.strip('\n').split('\t')
                id = terms[0]
                self.cachedSamples[id] = line.strip('\n')
        
        shutil.rmtree(cacheFolder)

        for sample in samples:
            if sample.id not in self.cachedSamples:
                self.cachedSamples[sample.id] = "N/A"

    def run(self, samples, **kwargs):

        samplesToRun:list[Sample] = list()

        if (os.path.exists(self.cacheResult)):
            with open(self.cacheResult) as fp:
                self.cachedSamples = json.load(fp)  # id: [offset, alignmentCount]

            for sample in samples:
                if (sample.id not in self.cachedSamples):
                    samplesToRun.append(sample)
        else:
            samplesToRun = samples
        
        if (len(samplesToRun) > 0):
            self.vcat(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->PlainResult:
        res = self.cachedSamples[sample.id]
        result = None
        if (res != "N/A"):
            terms = res.strip('\n').split("\t")
            taxo = None
            score = float(terms[2])
            method = terms[3]
            for i in reversed(range(4, len(terms))):
                if (terms[i]):
                    taxo = terms[i]
                    result = [PlainResult(taxo, score)]
                    break

        return result