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


class Virgo(Module):
    def __init__(self, trainset):
        super().__init__(f"Virgo-{trainset}")
        if (trainset not in ["VMRv4", "VMRv4_ML_train"]):
            raise ValueError("Unsupported Virgo training set")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.trainset = trainset
        self.cachedSamples:dict[str, str] = dict()
    

    def virgo(self, samples:list[Sample])->None:

        cacheFolder = f"{config.cacheFolder}/virgo"
        inputFolder = f"{cacheFolder}/input/"
        outputFolder = f"{cacheFolder}/output/"
        resultFile = f"{outputFolder}/results.csv"

        IOUtils.showInfo(f"Begin Virgo on {len(samples)} samples")


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

        os.makedirs(inputFolder)
        for sample in samples:
            IOUtils.writeSampleFasta([sample], f"{inputFolder}/{sample.id}.fasta")
        time.sleep(1)
        cwd = "/Software/Virgo"
        command = f"conda run -n virgo --no-capture-output python src/virgo.py -i {inputFolder} -o {outputFolder} -d database/{self.trainset}/"
        print(command)
        print(len(os.listdir(inputFolder)))

        subprocess.run(command, shell=True, cwd=cwd)

        with open(resultFile) as fp:
            fp.readline()
            for line in fp:
                terms = line.strip().split(',')
                id = terms[0]
                self.cachedSamples[id] = line
        
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
            self.virgo(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->PlainResult:
        res = self.cachedSamples[sample.id]
        result = None
        if (res != "N/A"):
            terms = res.split(",")
            taxo = None
            score = float(terms[8])
            for i in reversed(range(1, 7)):
                if (terms[i]):
                    taxo = terms[i]
                    result = [PlainResult(taxo, score)]
                    break

        return result