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
from moduleResult.plainResult import PlainResult


class VISTA(Module):
    def __init__(self):
        super().__init__(f"VISTA")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()
    

    def vista(self, samples:list[Sample])->None:

        IOUtils.showInfo(f"Begin VISTA on {len(samples)} samples")

        input_fasta = f"{config.cacheFolder}/vitax.fasta"
        output_txt = f"{config.cacheFolder}/vitax.txt"
        IOUtils.writeSampleFasta(samples, input_fasta)
        cwd = "/Software/ViTax"
        command = f"conda run -n vista Scripts/VISTA.sh -i {input_fasta} -o {output_txt}"
        subprocess.run(command, shell=True, cwd=cwd)

        with open(output_txt) as fp:
            for line in fp:
                terms = line.split('\t')
                if (terms[1] != 'unclassified'):
                    self.cachedSamples[terms[0]] = line.strip()
                else:
                    self.cachedSamples[terms[0]] = "N/A"
        
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
            ans = terms[1].split("_")[0]
            score = float(terms[2])
            result = PlainResult(ans, score)
        else:
            result = None

        return result