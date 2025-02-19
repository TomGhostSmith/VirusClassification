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


class PhaGCN(Module):
    def __init__(self, threads=12, lenThresh=8000):
        self.threads = threads
        super().__init__("PhaGCN2.0")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, int] = dict()
        self.lenThresh = lenThresh   # PhaGCN recommend 8000, and the minimum is 1700

    
    def runOnePhagcn(self, outputFolder, idx):
        env = {
            'CUDA_VISIBLE_DEVICES': "1",
            'MKL_SERVICE_FORCE_INTEL': "1"
        }
        env.update(os.environ)
        cwd = "/Software/PhaGCN2.0"
        command = f"conda run -n phagcn python run_Speed_up.py --len {self.lenThresh} --outpath {outputFolder}"
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=cwd, env=env)
        subprocess.run(command, shell=True, cwd=cwd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        
        # process = subprocess.Popen(command, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # while True:
        #     output = process.stdout.readline()
        #     if len(output.strip()) == 0 and process.poll() is not None:
        #         break
        #     if output:
        #         print(f"{output.strip()}")


        return idx
    

    def phagcn(self, samples:list[Sample])->None:

        IOUtils.showInfo(f"Begin PhaGCN on {len(samples)} samples")


        # divide sequences into groups of 1000

        params = list()

        file_id = 0
        records = []
        for sample in samples:
            if len(records) == 1000:
                outputFolder = f"{config.cacheFolder}/phagcn_output_{file_id}"
                queryFile = f"{outputFolder}/input/contig_0.fasta"
                # IOUtils.checkAndEmptyFolder(f"{outputFolder}")
                os.makedirs(f"{outputFolder}/input")
                IOUtils.writeSampleFasta(records, queryFile)
                params.append([outputFolder, file_id])
                records = []
                file_id += 1
            if not bool(re.compile(r'[^ATCG]').search(str(sample.seq.seq).upper())):
                if len(sample.seq.seq) > self.lenThresh:
                    records.append(sample)
        if len(records) != 0:
            outputFolder = f"{config.cacheFolder}/phagcn_output_{file_id}"
            queryFile = f"{outputFolder}/input/contig_0.fasta"
            # IOUtils.checkAndEmptyFolder(f"{outputFolder}")
            os.makedirs(f"{outputFolder}/input")
            IOUtils.writeSampleFasta(records, queryFile)
            params.append([outputFolder, file_id])
            file_id += 1

        # for param in params:
        #     res = self.runOnePhagcn(*param)


        with multiprocessing.Pool(self.threads) as pool:
            asyncResults = [pool.apply_async(self.runOnePhagcn, param) for param in params]
            for asyncResult in asyncResults:
                idx = asyncResult.get()
                outputFolder = f"{config.cacheFolder}/phagcn_output_{idx}"
                resultFile = f"{outputFolder}/pred/contig_0.csv"
                with open(resultFile) as fp:
                    fp.readline()  # skip the title
                    for line in fp:
                        terms = line.strip().split(',')
                        sampleName = terms[0]
                        sampleResult = ",".join(terms[2:])
                        self.cachedSamples[sampleName] = sampleResult
            pool.close()
            pool.join()
        
        for i in range(file_id):
            shutil.rmtree(f"{config.cacheFolder}/phagcn_output_{i}")

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
            self.phagcn(samplesToRun)
        
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