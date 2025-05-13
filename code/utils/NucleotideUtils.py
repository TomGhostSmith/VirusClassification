import os
import math
import json
from tqdm import tqdm
import subprocess
from Bio import SeqIO
import multiprocessing
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from entity.sample import Sample
from entity.proteinSample import ProteinSample

from config import config
from utils import IOUtils

class NucleotideUtil:
    def __init__(self):
        self.proteinIndex = f"{config.cacheFolder}/proteins.json"
        self.proteinFasta = f"{config.cacheFolder}/proteins.fasta"
        self.c2pCache = f"{config.cacheResultFolder}/c2p.json"

        self.c2p:dict[str, list] = dict()
        self.cachedProteins:dict[str, int] = dict()
        if (os.path.exists(self.c2pCache) and os.path.exists(self.proteinIndex)):
            with open(self.c2pCache) as fp:
                self.c2p = json.load(fp)
            with open(self.proteinIndex) as fp:
                self.cachedProteins = json.load(fp)
        
        self.thisOffset = self.cachedProteins["nextOffset"] if "nextOffset" in self.cachedProteins else 0
        self.nextOffset = self.thisOffset

    def runSingleProdigal(self, dnaFasta, outputFasta, idx=None):
        subprocess.run(f"prodigal-gv -i {dnaFasta} -a {outputFasta} -p meta", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return idx

    # extract protein with prodigal-gv, store the ids of protein to the sample
    def extractProtein(self, samples:list[Sample], samplePerThread=100, threads=multiprocessing.cpu_count())-> None:
        outputFile = f"{config.cacheFolder}/tmp.faa"
        # step 1: get samples to run, load cached samples
        samplesToRun:list[Sample] = list()


        for sample in samples:
            cached = True
            if (sample.proteins is not None):
                continue  # skip the sample that already has protein annotation
            if sample.id in self.c2p:
                    proteins = self.c2p[sample.id]
                    for protein in proteins:
                        if (protein not in self.cachedProteins):
                            cached = False
            else:
                cached = False
            if (not cached):
                samplesToRun.append(sample)
        

        # step 2: run prodigal with multi-threads
        splitCount = math.ceil(len(samplesToRun)/samplePerThread)
        tempFileName = f"{outputFile}.DNA"

        targetFP = open(outputFile)

        with multiprocessing.Pool(threads) as pool:
            asyncResults = list()
            for i in range(splitCount):
                IOUtils.writeSampleFasta(samplesToRun[samplePerThread * i : samplePerThread * (i+1)], f"{tempFileName}.{i}")
        
                asyncResults.append(pool.apply_async(self.runSingleProdigal, 
                                                [f"{tempFileName}.{i}", f"{outputFile}.{i}", 1]))
                
            for asyncResult in tqdm(asyncResults):
                idx = asyncResult.get()
                proteinSamlpes = IOUtils.loadProteinSamples(f"{outputFile}.{idx}")
                for protein in proteinSamlpes:
                    if (protein.contigID in self.c2p):
                        self.c2p[protein.contigID].append(protein.id)
                    else:
                        self.c2p[protein.contigID] = [protein.id]

                    head = f">{protein.id}\n"
                    seq = f"{protein.seq.seq}\n"
                    
                    targetFP.write(head)
                    targetFP.write(seq)

                    self.cachedProteins[protein.id] = self.thisOffset + len(head)
                    self.thisOffset = self.nextOffset
                    self.nextOffset = self.nextOffset + len(head) + len(seq)


        targetFP.close()

        cachedProteinFP = open(self.proteinFasta)
        for sample in samples:
            self.loadProteinSample(sample, cachedProteinFP)
        cachedProteinFP.close()

                
        for i in range(splitCount):
            os.remove(f"{tempFileName}.{i}")
            os.remove(f"{outputFile}.{i}")
        os.remove(outputFile)

        with open(self.c2pCache, 'wt') as fp:
            json.dump(self.c2p, fp, indent=2)
        with open(self.proteinIndex, 'wt') as fp:
            json.dump(self.cachedProteins, fp, indent=2)


    def loadProteinSample(self, sample:Sample, cachedProteinFP) -> None:
        proteins = self.c2p[sample.id] if sample.id in self.c2p else list()
        sample.proteins = list()
        for protein in proteins:
            offset = self.cachedProteins[protein]
            cachedProteinFP.seek(offset)
            sample.proteins.append(ProteinSample(SeqRecord(
                Seq(cachedProteinFP.readline().strip()),
                id=protein,
                description=protein)))
        
NucleotideUtils = NucleotideUtil()  # this is a singleton