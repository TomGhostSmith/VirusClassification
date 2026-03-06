import os
import h5py
import math
import time
import torch
import shutil
import subprocess


from entity.sample import Sample
from config import config
from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

from module.dnaLMRunner import DNALMRunner


class PSTRunner(DNALMRunner):
    def __init__(self, modelName):
        super().__init__(modelName)
        self.model = modelName
     
    def run(self, samples:list[Sample], **kwargs):  # only work for DNA whole sequence
        embeddings = {}
        NucleotideUtils.extractProtein(samples)

        samplesToRun = []
        for sample in samples:
            if (len(sample.proteins) > 1):
                samplesToRun.append(sample)
        GPUs = torch.cuda.device_count()
        samplesPerGPU = math.ceil(len(samplesToRun) / GPUs)

        processes:list[subprocess.Popen] = []

        for i in range(GPUs):
            tmpFolder = f"{config.cacheFolder}/PST_{i}"
            inputFile = f"{tmpFolder}/input.fasta"
            os.makedirs(tmpFolder, exist_ok=True)
            IOUtils.writeSampleProteinFasta(samplesToRun[i * samplesPerGPU : (i+1) * samplesPerGPU], inputFile, withHead=True)

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(i)
            cmd = f"conda run -n pst --no-capture-output python main.py {inputFile} {tmpFolder} {self.model}"
            cwd = "/Software/protein_set_transformer"
            processes.append((i, subprocess.Popen(cmd, cwd=cwd, shell=True, env=env)))
        # subprocess.run(cmd, cwd=cwd, shell=True)

        while processes:
            for i, p in processes[:]:
                if p.poll() is not None:
                    tmpFolder = f"{config.cacheFolder}/PST_{i}"
                    outputFile = f"{tmpFolder}/embeddings.h5"
                    file = h5py.File(outputFile, "r")

                    emb = file["genome"][:]  # Load into numpy array
                    for idx, s in enumerate(samplesToRun[i * samplesPerGPU : (i+1) * samplesPerGPU]):
                        embeddings[s.id] = emb[idx, :]

                    shutil.rmtree(tmpFolder)

                    processes.remove((i, p))
            time.sleep(1)

        return embeddings
    
    def clean(self):
        pass