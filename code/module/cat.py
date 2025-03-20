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
from moduleResult.catResult import CatResult


class CAT(Module):
    def __init__(self):
        super().__init__("CAT-NCBI")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()
        self.blockSize = 10
        self.threads = 2

    
    def runOneCAT(self, outputFolder, idx):
        env = os.environ.copy()
        
        cwd = "/Software/CAT_pack"
        inputFile = f"{outputFolder}/query.fasta"
        command = f"conda run -n CAT CAT_pack/CAT_pack contigs -c {inputFile} -d /Software/CAT_pack/model/20241212_CAT_nr_website/db -t /Software/CAT_pack/model/20241212_CAT_nr_website/tax --path_to_diamond /Software/CAT_pack/model/20241212_CAT_nr_website/diamond -o {outputFolder}/CAT --block_size {self.blockSize:.1f}"
        # python run_Speed_up.py --len {self.lenThresh} --outpath {outputFolder}"
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
    

    def cat(self, samples:list[Sample])->None:

        IOUtils.showInfo(f"Begin CAT on {len(samples)} samples")


        # divide sequences into groups of 1000

        # re-collect former results:
        modifiedResult = 0
        hasFormerResult = False
        for i in range(2):
            outputFolder = f"{config.cacheFolder}/CAT_output_{i}"
            resultFile = f"{outputFolder}/CAT.contig2classification.txt"
            if (os.path.exists(outputFolder)):
                hasFormerResult = True
                if (os.path.exists(resultFile)):
                    with open(resultFile) as fp:
                        fp.readline()  # skip the title
                        for line in fp:
                            terms = line.strip().split('\t')
                            sampleName = terms[0]
                            sampleResult = "\t".join(terms[1:])
                            self.cachedSamples[sampleName] = sampleResult
                            modifiedResult += 1
                shutil.rmtree(f"{config.cacheFolder}/CAT_output_{i}")
        if (hasFormerResult):
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)
            IOUtils.showInfo(f"Found and saved previous {modifiedResult} results. Please re-run the command")
            exit(-1)

        params = list()

        maxSamplePerThread = 1200
        if (len(samples) > 2 * maxSamplePerThread):
            processes = math.ceil(len(samples) / maxSamplePerThread / 2) * 2   # we want to avoid an "odd" number of threads
        else:
            processes = 2
        filePerThread = math.ceil(len(samples) / processes)
        # memory: 20GB + fpt/100 + blocksize + fpt * blocksize / 675 < 50
        self.blockSize = (30 - filePerThread/100) / (filePerThread/675 + 1)
        for i in range(processes):
            outputFolder = f"{config.cacheFolder}/CAT_output_{i}"
            queryFile = f"{outputFolder}/query.fasta"
            os.makedirs(outputFolder)
            IOUtils.writeSampleFasta(samples[i*filePerThread:(i+1)*filePerThread], queryFile)
            params.append([outputFolder, str(i)])

        with multiprocessing.Pool(self.threads) as pool:
            asyncResults = [pool.apply_async(self.runOneCAT, param) for param in params]
            for asyncResult in asyncResults:
                idx = asyncResult.get()
                outputFolder = f"{config.cacheFolder}/CAT_output_{idx}"
                resultFile = f"{outputFolder}/CAT.contig2classification.txt"
                with open(resultFile) as fp:
                    fp.readline()  # skip the title
                    for line in fp:
                        terms = line.strip().split('\t')
                        sampleName = terms[0]
                        sampleResult = "\t".join(terms[1:])
                        self.cachedSamples[sampleName] = sampleResult
            pool.close()
            pool.join()
        
        for i in range(processes):
            shutil.rmtree(f"{config.cacheFolder}/CAT_output_{i}")

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
            self.cat(samplesToRun)
        
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)

        results = [self.getResult(sample) for sample in samples]

        return results
    
    def getResult(self, sample:Sample)->CatResult:
        res = self.cachedSamples[sample.id]
        result = None
        if (res != "N/A"):
            terms = res.split('\t')
            if (len(terms) > 3):
                taxos = terms[2].split(';')
                scores = terms[3].split(';')
                scores = [float(score) for score in scores]
                taxoRes = list(zip(taxos, scores))
                result = CatResult(taxoRes)
                
        return result