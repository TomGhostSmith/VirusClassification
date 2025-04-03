# reconstructed
import os
import re
import sys
import json
import math
import time
import psutil
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
        self.threads = 2

    
    def runOneCAT(self, outputFolder, idx, blockSize, querySize):
        IOUtils.showInfo(f"run cat for subset {idx}")
        env = os.environ.copy()
        
        cwd = "/Software/CAT_pack"
        inputFile = f"{outputFolder}/query.fasta"
        command = f"conda run -n CAT CAT_pack/CAT_pack contigs -c {inputFile} -d /Software/CAT_pack/model/20241212_CAT_nr_website/db -t /Software/CAT_pack/model/20241212_CAT_nr_website/tax --path_to_diamond /Software/CAT_pack/model/20241212_CAT_nr_website/diamond -o {outputFolder}/CAT --block_size {blockSize:.1f}"
        # python run_Speed_up.py --len {self.lenThresh} --outpath {outputFolder}"
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=cwd, env=env)
        process = subprocess.Popen(command, shell=True, cwd=cwd, env=env, stdout=sys.stdout, stderr=sys.stderr)

        monitored_time = 0
        cat_process = psutil.Process(process.pid)
        wait_diamond_timeout = 300
        monitor_timeout = 600
        diamondPID = None
        diamondProcess = None
        mem_thresh = 60
        # step 1: find diamond
        while monitored_time < wait_diamond_timeout:
            for child in cat_process.children(recursive=True):
                if ("diamond" in child.name().lower()):
                    diamondPID = child.pid
                    diamondProcess = psutil.Process(child.pid)
                    break
            if (diamondPID is not None):
                break
            time.sleep(5)
            monitored_time += 5

        # handle error if timeout and not found diamond
        if (diamondProcess is None):
            IOUtils.showInfo("timeout for finding DIAMOND process")
            return idx, blockSize, querySize, True
        IOUtils.showInfo(f"found DIAMOND process: PID={diamondPID}")

        while monitored_time < monitor_timeout:
            if not diamondProcess.is_running():
                IOUtils.showInfo("subprocess 'diamond' is finished", 'WARN')
                return idx, blockSize, querySize, True
            try:
                mem_usage = diamondProcess.memory_percent()
                totalUsed = psutil.virtual_memory().used/1024/1024/1024
                totalMem = psutil.virtual_memory().total/1024/1024/1024
            except psutil.NoSuchProcess:
                IOUtils.showInfo("subprocess 'diamond' is lost", 'WARN')
                return idx, blockSize, querySize, True
            if (mem_usage > mem_thresh and totalMem - totalUsed < 10):
                print(f"kill pid={diamondPID}, who uses memory {mem_usage}%, which is above the threshold {mem_thresh}%")
                subprocess.run(f"kill {diamondPID}", shell=True)
                return idx, blockSize, querySize, False

            monitored_time += 5
            time.sleep(5)

        IOUtils.showInfo(f'monitor for PID={diamondPID} timeout. Diamond will keep running')
        
        process.wait()
        IOUtils.showInfo("one CAT finished")

        return idx, blockSize, querySize, True

        
        # process = subprocess.Popen(command, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # while True:
        #     output = process.stdout.readline()
        #     if len(output.strip()) == 0 and process.poll() is not None:
        #         break
        #     if output:
        #         print(f"{output.strip()}")

    

    def cat(self, samples:list[Sample])->None:

        IOUtils.showInfo(f"Begin CAT on {len(samples)} samples")

        # re-collect former results:
        modifiedResult = 0
        hasFormerResult = False
        potentialIndexes = list()
        potentialIndexes += list(range(100))
        potentialIndexes += [f"spec_{idx}" for idx in range(100)]
        folders = os.listdir(config.cacheFolder)
        for folder in folders:
            folderPath = f"{config.cacheFolder}/{folder}"
            if (folder.startswith("CAT_output_") and os.path.isdir(folderPath)):
                resultFile = f"{folderPath}/CAT.contig2classification.txt"
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
                shutil.rmtree(folderPath)
        if (hasFormerResult):
            with open(self.cacheResult, 'wt') as fp:
                json.dump(self.cachedSamples, fp, indent=2)
            IOUtils.showInfo(f"Found and saved previous {modifiedResult} results. Please re-run the command")
            exit(-1)

        params = list()

        # pickout too short samples (who may have too many alignments)
        basicSamples = list()
        remainedSamples = list()
        for sample in samples:
            # if (sample.length < 200):
            #     specSamples.append(sample)
            # else:
            basicSamples.append(sample)
        
        indexes = list()

        downGrades = [
50, 40, 30, 20, 10
        ]
        # downGrades = [
        #     2400,
        #     1800,
        #     1200,
        #     800,
        #     500,
        #     300,
        #     200,
        #     100
        # ]
        # extCount = 0

        # maxSamplePerThread = 1800
        # maxSamplePerThread = 60
        maxSamplePerThread = 45
        # if (len(basicSamples) > 2 * maxSamplePerThread):
        #     processes = math.ceil(len(basicSamples) / maxSamplePerThread / 2) * 2   # we want to avoid an "odd" number of threads
        # else:
        #     processes = 2
        totalBP = 0
        # processes = 0
        # filePerThread = math.ceil(len(basicSamples) / processes)
        # memory: 20GB + fpt/100 + blocksize + fpt * blocksize / 675 < 50
        # self.blockSize = (28 - filePerThread/100) / (filePerThread/675 + 1)
        thisCollection = list()
        thisIdx = 0
        for sample in samples:
            totalBP += sample.length
            thisCollection.append(sample)
            if totalBP > maxSamplePerThread * 1024 * 1024:
                outputFolder = f"{config.cacheFolder}/CAT_output_{thisIdx}"
                queryFile = f"{outputFolder}/query.fasta"
                os.makedirs(outputFolder)
                # IOUtils.writeSampleFasta(basicSamples[i*filePerThread:(i+1)*filePerThread], queryFile)
                IOUtils.writeSampleFasta(thisCollection, queryFile)
                params.append([outputFolder, str(thisIdx), 10, totalBP/1024/1024])
                indexes.append(str(thisIdx))
                thisCollection = list()
                totalBP = 0
                thisIdx += 1
        
        if totalBP > 0:
            outputFolder = f"{config.cacheFolder}/CAT_output_{thisIdx}"
            queryFile = f"{outputFolder}/query.fasta"
            os.makedirs(outputFolder)
            # IOUtils.writeSampleFasta(basicSamples[i*filePerThread:(i+1)*filePerThread], queryFile)
            IOUtils.writeSampleFasta(thisCollection, queryFile)
            params.append([outputFolder, str(thisIdx), 10, totalBP/1024/1024])
            indexes.append(str(thisIdx))
            thisCollection = list()
            totalBP = 0
            thisIdx += 1

        with multiprocessing.Pool(self.threads) as pool:
            asyncResults = [pool.apply_async(self.runOneCAT, param) for param in params]
            while (len(asyncResults) > 0):
                time.sleep(5)
                for asyncResult in asyncResults[:]:
                    if (asyncResult.ready()):
                        idx, blkSize, querySize, success = asyncResult.get()
                        asyncResults.remove(asyncResult)
                        if (success):
                            outputFolder = f"{config.cacheFolder}/CAT_output_{idx}"
                            resultFile = f"{outputFolder}/CAT.contig2classification.txt"
                            with open(resultFile) as fp:
                                fp.readline()  # skip the title
                                for line in fp:
                                    terms = line.strip().split('\t')
                                    sampleName = terms[0]
                                    sampleResult = "\t".join(terms[1:])
                                    self.cachedSamples[sampleName] = sampleResult
                        else:  # OOM happened
                            originFile = f"{config.cacheFolder}/CAT_output_{idx}/query.fasta"
                            originSamples = IOUtils.loadSamples(originFile)
                            remainedSamples += originSamples


                            nextSampleSize = 50
                            for k in downGrades:
                                if (k < querySize):
                                    nextSampleSize = k
                                    break

                            thisCollection = list()
                            for sample in remainedSamples:
                                totalBP += sample.length
                                thisCollection.append(sample)
                                if totalBP > maxSamplePerThread * 1024 * 1024:
                                    outputFolder = f"{config.cacheFolder}/CAT_output_{thisIdx}"
                                    queryFile = f"{outputFolder}/query.fasta"
                                    os.makedirs(outputFolder)
                                    # IOUtils.writeSampleFasta(basicSamples[i*filePerThread:(i+1)*filePerThread], queryFile)
                                    IOUtils.writeSampleFasta(thisCollection, queryFile)
                                    asyncResults.append(pool.apply_async(self.runOneCAT, [outputFolder, str(thisIdx), blkSize - 1, nextSampleSize]))
                                    indexes.append(str(thisIdx))
                                    thisIdx += 1
                                    thisCollection = list()
                                    totalBP = 0

                        if (len(asyncResults) <= 1 and len(remainedSamples) > 0):
                            outputFolder = f"{config.cacheFolder}/CAT_output_{thisIdx}"
                            queryFile = f"{outputFolder}/query.fasta"
                            os.makedirs(outputFolder)
                            # IOUtils.writeSampleFasta(basicSamples[i*filePerThread:(i+1)*filePerThread], queryFile)
                            IOUtils.writeSampleFasta(remainedSamples, queryFile)
                            asyncResults.append(pool.apply_async(self.runOneCAT, [outputFolder, str(thisIdx), blkSize - 1, nextSampleSize]))
                            indexes.append(str(thisIdx))
                            thisIdx += 1
                            thisCollection = list()
                            totalBP = 0


            pool.close()
            pool.join()
        
        for i in indexes:
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
