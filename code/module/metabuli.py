# reconstructed
import os
import sys
import json
import shutil
import subprocess

from prototype.module import Module
from config import config
from utils import IOUtils
from entity.sample import Sample
from moduleResult.plainResult import PlainResult


class Metabuli(Module):
    def __init__(self, trainset):
        super().__init__(f"metabuli-v1.0.9.2-{trainset}")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.trainset = trainset
        self.cachedSamples:dict[str, str] = dict()
    

    def metabuli(self, samples:list[Sample])->None:

        cacheFolder = f"{config.cacheFolder}/metabuli"
        inputFile = f"{config.cacheFolder}/metabuli.fasta"
        resultFile = f"{cacheFolder}/0_classifications.tsv"

        IOUtils.showInfo(f"Begin Metabuli on {len(samples)} samples")


        # re-collect former results:
        if (os.path.exists(resultFile)):
            modifiedResult = 0
            with open(resultFile) as fp:
                for line in fp:
                    terms = line.strip().split('\t')
                    id = terms[1]
                    self.cachedSamples[id] = line
                    modifiedResult += 1
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)
            IOUtils.showInfo(f"Found and saved previous {modifiedResult} results. Please re-run the command")
            shutil.rmtree(cacheFolder)
            exit(0)


        IOUtils.writeSampleFasta(samples, inputFile)
        cwd = "/Software/Metabuli/working/Metabuli-ICTV-challenge"
        command = f"../../build/src/metabuli classify --seq-mode 1 {inputFile} db_{self.trainset} {cacheFolder} 0 --lineage 1"

        subprocess.run(command, shell=True, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)

        with open(resultFile) as fp:
            for line in fp:
                terms = line.strip().split('\t')
                id = terms[1]
                self.cachedSamples[id] = line
        
        os.remove(inputFile)
        shutil.rmtree(cacheFolder)


        for sample in samples:
            if sample.id not in self.cachedSamples:
                self.cachedSamples[sample.id] = "N/A"

    def run(self, samples):

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
            self.metabuli(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->PlainResult:
        res = self.cachedSamples[sample.id]
        if (res != "N/A"):
            terms = res.split("\t")
            if (terms[0] == "0"):
                result = None
            else:
                taxo = terms[6].split(';')
                term = taxo[-1].split('_')[1]
                result = [PlainResult(term)]
        else:
            result = None

        return result