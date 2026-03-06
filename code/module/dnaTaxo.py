import os
import json
import math
import time
import torch
import numpy
import shutil
import subprocess

from prototype.module import Module
from entity.sample import Sample
from moduleResult.plainResult import PlainResult
from config import config
from utils import IOUtils
from entity.taxoTree import taxoTree

class DNATaxo(Module):
    def __init__(self, modelName, rank):
        super().__init__(f"DNA_taxo_{modelName}")
        self.name = modelName
        self.rank = rank

        self.class_names = taxoTree.taxaNames[rank]

        self.cacheProbFile = f"{config.cacheResultFolder}/DNA_taxo_{modelName}_prob.tmp"
        self.cacheProbIndex = f"{config.cacheResultFolder}/DNA_taxo_{modelName}_prob.json"

        self.cachedSamples_prob = {"nextOffset": 0}
        self.nextOffset_prob = 0

    def dnaTaxo(self, samples:list[Sample]):
        IOUtils.showInfo(f"Run {len(samples)} on {self.moduleName}")
        modelPath = f"{config.modelRoot}/dnamodel/{self.name}"
        GPUs = torch.cuda.device_count()
        samplesPerGPU = math.ceil(len(samples) / GPUs)

        processes = []
        scriptPath = "/Software/VirusClassification/working/dnaTaxoTrain.py"
        for i in range(GPUs):
            cwd = f"{config.cacheFolder}/DNATaxo-{i}"
            fastaPath = f"{cwd}/input.fasta"
            os.makedirs(cwd, exist_ok=True)
            IOUtils.writeSampleFasta(samples[i * samplesPerGPU : (i+1) * samplesPerGPU], fastaPath)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(i)

            cmd = f"python {scriptPath} --num-labels {len(self.class_names)} --max-length 32768 -b {config.DNABatchSize} --model-path {modelPath} predict --test-data {fastaPath}"
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env)
            processes.append((i, proc))

        
        lines = []
        while processes:
            for i, p in processes[:]:
                if p.poll() is not None:
                    cwd = f"{config.cacheFolder}/DNATaxo-{i}"
                    matrix = numpy.load(f"{cwd}/test_scores.npy")
                    matrix = torch.nn.functional.softmax(torch.tensor(matrix), dim=1).numpy().astype(numpy.float16)

                    for sample, line in zip(samples[i * samplesPerGPU : (i+1) * samplesPerGPU], matrix):
                        line = line.astype(numpy.float16)
                        probText = f"{sample.id}\t{IOUtils.encodeBase64(line)}\n"
                        lines.append(probText)
                        self.cachedSamples_prob[sample.id] = self.nextOffset_prob
                        self.nextOffset_prob += len(probText)
                    
                    shutil.rmtree(cwd)
                    processes.remove((i, p))
            time.sleep(1)

        for sample in samples:
            if sample.id not in self.cachedSamples_prob:
                self.cachedSamples_prob[sample.id] = -1


        self.cachedSamples_prob["nextOffset"] = self.nextOffset_prob

        with open(self.cacheProbFile, 'at') as fp:
            fp.writelines(lines)

    def run(self, samples:list[Sample], keepVotes=False, **kwargs):
        if (os.path.exists(self.cacheProbIndex)):
            with open(self.cacheProbIndex) as fp:
                self.cachedSamples_prob = json.load(fp)
                self.nextOffset_prob = self.cachedSamples_prob["nextOffset"]

        samplesToRun = []
        for sample in samples:
            if (sample.id not in self.cachedSamples_prob):
                samplesToRun.append(sample)
        
        if (len(samplesToRun) > 0):
            self.dnaTaxo(samplesToRun)

            with open(self.cacheProbIndex, 'wt') as fp:
                json.dump(self.cachedSamples_prob, fp, indent=2)


        fp = open(self.cacheProbFile)
        results = [self.getResult(sample, fp, keepVotes) for sample in samples]
        fp.close()

        return results

    def getResult(self, sample:Sample, fp, keepVotes):
        offset = self.cachedSamples_prob[sample.id]
        if (offset == -1):
            return None
        fp.seek(offset)
        line = fp.readline().strip('\n')
        probText = line[line.find('\t')+1:]
        probs = IOUtils.decodeBase64(probText)
        predictions = [PlainResult(pred, prob) for pred, prob in sorted(zip(self.class_names, probs), key=lambda x:x[1], reverse=True) if prob > 0 and "Unknown" not in pred]
        if (keepVotes):
            votes = {pred: prob for pred, prob in zip(self.class_names, probs)}
            sample.info[f"{self.moduleName}_votes"] = votes
        return predictions
