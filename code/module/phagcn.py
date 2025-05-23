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


class PhaGCN(Module):
    def __init__(self, version, threads=multiprocessing.cpu_count(), lenThresh=8000, n=10000):
        self.threads = threads
        self.version = version
        super().__init__(f"PhaGCN{self.version}_n={n}-VMRv1")
        self.cacheResult = f"{config.cacheResultFolder}/{self.moduleName}.json"
        self.cachedSamples:dict[str, str] = dict()
        self.lenThresh = lenThresh   # PhaGCN recommend 8000, and the minimum is 1700
        self.oomThresh = 500000  # sequence longer than this should be run with CPU to avoid cuda OOM

        self.commonFileSize = n
        self.longseqFileSize = n // 10

    
    def runOnePhagcn(self, outputFolder, idx, GPUDevice="1"):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = "1"
        env['MKL_SERVICE_FORCE_INTEL'] = "1"
        env["CNN_DEVICE"] = GPUDevice
        
        if (self.version == "2"):
            cwd = "/Software/PhaGCN2.0"
            command = f"conda run -n phagcn python run_Speed_up.py --len {self.lenThresh} --outpath {outputFolder}"

        elif (self.version.startswith("3")):
            cwd = "/Software/PhaGCN3"
            command = f"conda run -n PhaGCN3 python run_Speed_up.py --len {self.lenThresh} --outpath {outputFolder}"
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

        IOUtils.showInfo(f"Begin PhaGCN{self.version} on {len(samples)} samples")


        # re-collect former results:
        modifiedResult = 0
        hasFormerResult = False
        folders = os.listdir(config.cacheFolder)
        for folder in folders:
            folderPath = f"{config.cacheFolder}/{folder}"
            if (folder.startswith("phagcn_output_") and os.path.isdir(folderPath)):

                if (self.version in ["2", "3"]):
                    resultFile = f"{folderPath}/final_prediction.csv"
                elif (self.version in ["3-merged"]):
                    resultFile = f"{folderPath}/merged_prediction.csv"
                hasFormerResult = True
                if (os.path.exists(resultFile)):
                    with open(resultFile) as fp:
                        fp.readline()  # skip the title
                        for line in fp:
                            terms = line.strip().split(',')
                            sampleName = terms[0]
                            sampleResult = "\t".join(terms[2:])
                            self.cachedSamples[sampleName] = sampleResult
                            modifiedResult += 1
                shutil.rmtree(folderPath)
        if (hasFormerResult):
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)
            IOUtils.showInfo(f"Found and saved previous {modifiedResult} results. Please re-run the command")
            exit(0)


        # divide sequences into groups of 1000

        params = list()

        records = []
        longRecords = []
        for sample in samples:
            if not bool(re.compile(r'[^ATCG]').search(str(sample.seq.seq).upper())):
                if len(sample.seq.seq) > self.lenThresh:
                    if (len(sample.seq.seq) < self.oomThresh):
                        records.append(sample)
                    else:
                        longRecords.append(sample)

        commonFiles = math.ceil(len(records)/self.commonFileSize)
        for i in range(commonFiles):
            outputFolder = f"{config.cacheFolder}/phagcn_output_{i}"
            queryFile = f"{outputFolder}/input/contig_0.fasta"
            os.makedirs(f"{outputFolder}/input")
            IOUtils.writeSampleFasta(records[i*self.commonFileSize:(i+1)*self.commonFileSize], queryFile)
            params.append([outputFolder, str(i), "0"])
        
        longSeqFiles = math.ceil(len(longRecords)/self.longseqFileSize)
        for i in range(longSeqFiles):
            outputFolder = f"{config.cacheFolder}/phagcn_output_long_{i}"
            queryFile = f"{outputFolder}/input/contig_0.fasta"
            os.makedirs(f"{outputFolder}/input")
            IOUtils.writeSampleFasta(longRecords[i*self.longseqFileSize:(i+1)*self.longseqFileSize], queryFile)
            params.append([outputFolder, f"long_{i}", ""])
        

        # for param in params:
        #     res = self.runOnePhagcn(*param)


        with multiprocessing.Pool(self.threads) as pool:
            asyncResults = [pool.apply_async(self.runOnePhagcn, param) for param in params]
            for asyncResult in asyncResults:
                idx = asyncResult.get()
                outputFolder = f"{config.cacheFolder}/phagcn_output_{idx}"
                if (self.version in ["2", "3"]):
                    resultFile = f"{outputFolder}/final_prediction.csv"
                elif (self.version in ["3-merged"]):
                    resultFile = f"{outputFolder}/merged_prediction.csv"
                with open(resultFile) as fp:
                    fp.readline()  # skip the title
                    for line in fp:
                        terms = line.strip().split(',')
                        sampleName = terms[0]
                        sampleResult = "\t".join(terms[2:])
                        self.cachedSamples[sampleName] = sampleResult
            pool.close()
            pool.join()
        

        for i in range(commonFiles):
            shutil.rmtree(f"{config.cacheFolder}/phagcn_output_{i}")
        for i in range(longSeqFiles):
            shutil.rmtree(f"{config.cacheFolder}/phagcn_output_long_{i}")

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
    
    def getResult(self, sample:Sample)->PlainResult:
        res = self.cachedSamples[sample.id]
        if (res != "N/A"):
            term = res.split("\t")[0]
            result = PlainResult(term)
        else:
            result = None

        return result