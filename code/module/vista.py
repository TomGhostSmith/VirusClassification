import os
import json
import math
import subprocess
import multiprocessing
from tqdm import tqdm

from config import config
from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module
from moduleResult.plainResult import PlainResult


class VISTA(Module):
    def __init__(self, batchSize=100, threads=multiprocessing.cpu_count(), batchThreads=4, cacheBase="."):
        super().__init__(f"VISTA")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()
        self.batchSize = batchSize
        self.threads = threads
        self.batchThreads = batchThreads
        self.cacheBase=cacheBase

    def vista(self, samples:list[Sample], **kwargs)->None:

        IOUtils.showInfo(f"Begin VISTA on {len(samples)} samples")

        batchCount = math.ceil(len(samples) / self.batchSize)
        for i in range (batchCount):
            input_fasta = f"{config.cacheFolder}/vista_{i}.fasta"
            IOUtils.writeSampleFasta(samples[i * self.batchSize : (i+1)*self.batchSize], input_fasta)

        jobs = list(range(batchCount))
        procs = list()
        pbar = tqdm(total=batchCount, desc="VISTA")
        cwd = "/Software/VISTA"
        while jobs or procs:
            for proc, idx in procs:
                output_txt = f"{config.cacheFolder}/vista_{idx}.tsv"
                if (proc.poll() is not None):
                    procs.remove((proc, idx))
                    with open(output_txt) as fp:
                        for line in fp:
                            terms = line.split('\t')
                            self.cachedSamples[terms[1]] = line.strip()
                    pbar.update(1)

            while (len(procs) < self.threads and jobs):
                idx = jobs.pop(0)
                input_fasta = f"{config.cacheFolder}/vista_{idx}.fasta"
                output_txt = f"{config.cacheFolder}/vista_{idx}.tsv"
                command = f"conda run -n vista --no-capture-output python Scripts/base.py {input_fasta} {output_txt} {self.cacheBase}/VISTA_{idx} {self.batchThreads}"
                proc = subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # proc = subprocess.Popen(command, shell=True, cwd=cwd)
                procs.append((proc, idx))


        for i in range(batchCount):
            input_fasta = f"{config.cacheFolder}/vista_{i}.fasta"
            output_txt = f"{config.cacheFolder}/vista_{i}.tsv"
            os.remove(input_fasta)
            os.remove(output_txt)

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
            self.vista(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->PlainResult:
        res = self.cachedSamples[sample.id]
        if (res != "N/A"):
            terms = res.split("\t")
            confidence = terms[9]
            if (confidence == "Known Species"):
                ans = terms[7]
            elif (confidence == "Novel Species" or confidence == "Known Genus"):  # use genus
                ans = terms[6]

            elif (confidence == "Novel Genus"):  # use genus
                ans = terms[5]
            else:  # use family
                ans = terms[4]
            result = [PlainResult(ans)]
        else:
            result = None

        return result