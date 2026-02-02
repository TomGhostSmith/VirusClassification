import subprocess
import os
import shutil
import h5py


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
        tmpFolder = f"{config.cacheFolder}/PST"
        inputFile = f"{tmpFolder}/input.fasta"
        outputFile = f"{tmpFolder}/embeddings.h5"
        os.makedirs(tmpFolder, exist_ok=True)
        IOUtils.writeSampleProteinFasta(samplesToRun, inputFile, withHead=True)

        cmd = f"conda run -n pst --no-capture-output python main.py {inputFile} {tmpFolder} {self.model}"
        cwd = "/Software/protein_set_transformer"
        subprocess.run(cmd, cwd=cwd, shell=True)


        file = h5py.File(outputFile, "r")

        emb = file["genome"][:]  # Load into numpy array
        for idx, s in enumerate(samplesToRun):
            embeddings[s.id] = emb[idx, :]

        shutil.rmtree(tmpFolder)
        return embeddings
    
    def clean(self):
        pass