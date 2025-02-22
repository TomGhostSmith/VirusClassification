# reconstructed
import os
import re
import sys
import json
import math
import shutil
import multiprocessing
import subprocess
from tqdm import tqdm
from Bio import SeqIO

from prototype.module import Module
from config import config
from utils import IOUtils
from entity.sample import Sample
from moduleResult.phagcnResult import PhaGCNResult


class Vcontact(Module):
    def __init__(self, threads=12):
        self.threads = threads
        super().__init__("vContact")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()

    
    def runOneVcontact(self, outputFolder, idx):
        cwd = "/Software/vcontact2"
        command = f"conda run -n vContact2 python main.py --input {outputFolder}/dna.fasta --output {outputFolder}"
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=cwd, env=env)
        IOUtils.showInfo(f"Working on fragment {idx}")
        IOUtils.showInfo(f"Finished on fragment {idx}")
        subprocess.run(command, shell=True, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)
        return idx
    

    def vcontact(self, samples:list[Sample])->None:

        IOUtils.showInfo(f"Begin vConTACT on {len(samples)} samples")


        # divide sequences into groups of 1000

        params = list()

        samplePerGroup = 300

        commonFiles = math.ceil(len(samples)/samplePerGroup)
        for i in range(commonFiles):
            outputFolder = f"{config.cacheFolder}/vcontact_output_{i}"
            queryFile = f"{outputFolder}/dna.fasta"
            # os.makedirs(outputFolder)
            os.makedirs(outputFolder, exist_ok=True)
            IOUtils.writeSampleFasta(samples[i*samplePerGroup : (i+1)*samplePerGroup], queryFile)
            params.append([outputFolder, str(i)])
        

        # for param in params:
        #     res = self.runOnePhagcn(*param)


        with multiprocessing.Pool(self.threads) as pool:
            asyncResults = [pool.apply_async(self.runOneVcontact, param) for param in params]
            for asyncResult in asyncResults:
                idx = asyncResult.get()
                outputFolder = f"{config.cacheFolder}/vcontact_output_{idx}"
                resultFile = f"{outputFolder}/summary.json"
                with open(resultFile) as fp:
                    self.cachedSamples.update(json.load(fp))
            pool.close()
            pool.join()
        
        # for i in range(commonFiles):
        #     shutil.rmtree(f"{config.cacheFolder}/vcontact_output_{i}")

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
            self.vcontact(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->PhaGCNResult:
        res = self.cachedSamples[sample.id]
        if (res != "N/A"):
            result = PhaGCNResult(res)
        else:
            result = None

        return result
    
    def getKrakenCommand(self, queryFile):
        return f"kraken2 --db {config.modelRoot}/kraken2 {queryFile}"
