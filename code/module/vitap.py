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
from moduleResult.vitapResult import VitapResult


class VITAP(Module):
    def __init__(self, threads=12):
        self.threads = threads
        super().__init__("VITAP-VMRv4")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()

    
    def runOneVitap(self, outputFolder, idx):
        env = os.environ.copy()
        # env['CUDA_VISIBLE_DEVICES'] = "1"
        # env["CNN_DEVICE"] = GPUDevice
        
        cwd = "/Software/VITAP"
        inputFasta = f"{outputFolder}/query.fasta"
        databasePath = "DB_MSL"
        command = f"conda run -n vitap bash scripts/VITAP assignment -i {inputFasta} -d {databasePath} -o {outputFolder}"
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=cwd, env=env)
        subprocess.run(command, shell=True, cwd=cwd, env=env, stdout=sys.stdout, stderr=sys.stderr)


        return idx
    

    def vitap(self, samples:list[Sample])->None:

        IOUtils.showInfo(f"Begin VITAP on {len(samples)} samples")


        # divide sequences into groups of 1000
        params = list()

        # commonFiles = math.ceil(len(samples)/1000)
        if (len(samples) > self.threads * 100):
            threads = self.threads
        else:
            threads = math.ceil(len(samples) / 100)
        filePerThread = math.ceil(len(samples) / threads)
        for i in range(threads):
            outputFolder = f"{config.cacheFolder}/vitap_output_{i}"
            queryFile = f"{outputFolder}/query.fasta"
            IOUtils.writeSampleFasta(samples[i*filePerThread:(i+1)*filePerThread], queryFile)
            params.append([outputFolder, str(i)])

        # for param in params:
        #     res = self.runOnePhagcn(*param)


        with multiprocessing.Pool(threads) as pool:
            asyncResults = [pool.apply_async(self.runOnePhagcn, param) for param in params]
            for asyncResult in asyncResults:
                idx = asyncResult.get()
                outputFolder = f"{config.cacheFolder}/vitap_output_{idx}"
                resultFile = f"{outputFolder}/best_determined_lineages.tsv"
                with open(resultFile) as fp:
                    fp.readline()
                    for line in fp:
                        terms = line.strip().split('\t')
                        sampleName = terms[0]
                        sampleResult = "\t".join(terms[1:])
                        self.cachedSamples[sampleName] = sampleResult
            pool.close()
            pool.join()
        
        for i in range(threads):
            shutil.rmtree(f"{config.cacheFolder}/vitap_output_{i}")

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
            self.vitap(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->VitapResult:
        res = self.cachedSamples[sample.id]
        if (res != "N/A"):
            result = VitapResult(res)
        else:
            result = None

        return result