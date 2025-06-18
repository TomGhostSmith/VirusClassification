import os
import math
import json
import time
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

import fcntl

class NucleotideUtil:
    def __init__(self):
        self.proteinIndex = f"{config.cacheResultFolder}/proteins.json"
        self.proteinFasta = f"{config.cacheResultFolder}/proteins.fasta"
        self.cDNAIndex = f"{config.cacheResultFolder}/cDNAs.json"
        self.cDNAFasta = f"{config.cacheResultFolder}/cDNAs.fasta"
        self.c2pCache = f"{config.cacheResultFolder}/c2p.json"

        self.c2p:dict[str, list] = dict()
        self.cachedProteins:dict[str, int] = dict()
        self.cachedCDNAs:dict[str, int] = dict()
        if (os.path.exists(self.c2pCache) and os.path.exists(self.proteinIndex)):
            with open(self.c2pCache) as fp:
                self.c2p = json.load(fp)
            with open(self.proteinIndex) as fp:
                self.cachedProteins = json.load(fp)
            with open(self.cDNAIndex) as fp:
                self.cachedCDNAs = json.load(fp)
        
        self.proteinOffset = self.cachedProteins["nextOffset"] if "nextOffset" in self.cachedProteins else 0
        self.cDNAOffset = self.cachedCDNAs["nextOffset"] if "nextOffset" in self.cachedCDNAs else 0

    # extract protein with prodigal-gv, store the ids of protein to the sample
    def extractProtein(self, samples:list[Sample], samplePerThread=100, threads=multiprocessing.cpu_count())-> None:
        outputPrefix = f"{config.cacheFolder}/tmp.faa"
        cDNAoutputPrefix = f"{config.cacheFolder}/tmp.fna"
        # step 1: get samples to run, load cached samples
        samplesToRun:list[Sample] = list()
        samplesNotLoaded:list[Sample] = list()

        for sample in samples:
            if (sample.proteins is not None): # skip the sample that already has protein annotation
                continue
            samplesNotLoaded.append(sample)

        for sample in samplesNotLoaded:
            cached = True
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
        if (len(samplesToRun) > 0):
            splitCount = math.ceil(len(samplesToRun)/samplePerThread)
            tempFileName = f"{outputPrefix}.DNA"

            proteinFP = open(self.proteinFasta, 'at')
            cDNAFP = open(self.cDNAFasta, 'at')

            procs:list[tuple] = list()
            jobs = list(range(splitCount))
            pbar = tqdm(total=len(jobs), desc="protein translation")

            while procs or jobs:
                for proc, idx in procs:
                    if (proc.poll()) is not None:
                        procs.remove((proc, idx))
                        pbar.update(1)
                
                # if the pool has some room, apply more tasks
                # do not use fork to avoid large memory copy problem
                while len(procs) < threads and jobs:
                    idx = jobs.pop(0)
                    IOUtils.writeSampleFasta(samplesToRun[samplePerThread * idx : samplePerThread * (idx+1)], f"{tempFileName}.{idx}")
                    proc = subprocess.Popen(f"prodigal-gv -i {tempFileName}.{idx} -a {outputPrefix}.{idx} -d {cDNAoutputPrefix}.{idx} -p meta", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    procs.append((proc, idx))
                
                time.sleep(1)
            
            pbar.close()

            # collect result when all done
            for idx in tqdm(range(splitCount), desc="indexing result"):
                proteinSamlpes = IOUtils.loadProteinSamples(f"{outputPrefix}.{idx}")
                for protein in proteinSamlpes:
                    if (protein.contigID in self.c2p):
                        self.c2p[protein.contigID].append(protein.id)
                    else:
                        self.c2p[protein.contigID] = [protein.id]

                    head = f">{protein.id}\n"
                    seq = f"{protein.seq.seq}\n"
                    
                    proteinFP.write(head)
                    proteinFP.write(seq)

                    self.cachedProteins[protein.id] = self.proteinOffset + len(head)
                    self.proteinOffset = self.proteinOffset + len(head) + len(seq)

                cDNASamples = IOUtils.loadProteinSamples(f"{cDNAoutputPrefix}.{idx}")
                for protein in cDNASamples:
                    head = f">{protein.id}\n"
                    seq = f"{protein.seq.seq}\n"
                    
                    cDNAFP.write(head)
                    cDNAFP.write(seq)

                    self.cachedCDNAs[protein.id] = self.cDNAOffset + len(head)
                    self.cDNAOffset = self.cDNAOffset + len(head) + len(seq)

            proteinFP.close()
            cDNAFP.close()
            self.cachedProteins['nextOffset'] = self.proteinOffset
            self.cachedCDNAs['nextOffset'] = self.cDNAOffset

            for sample in samples:
                if sample.id not in self.c2p:
                    self.c2p[sample.id] = list()

            with open(self.c2pCache, 'wt') as fp:
                json.dump(self.c2p, fp, indent=2)
            with open(self.proteinIndex, 'wt') as fp:
                json.dump(self.cachedProteins, fp, indent=2)
            with open(self.cDNAIndex, 'wt') as fp:
                json.dump(self.cachedCDNAs, fp, indent=2)
                
            for i in range(splitCount):
                os.remove(f"{tempFileName}.{i}")
                if (os.path.exists(f"{outputPrefix}.{i}")):
                    os.remove(f"{outputPrefix}.{i}")
                if (os.path.exists(f"{cDNAoutputPrefix}.{i}")):
                    os.remove(f"{cDNAoutputPrefix}.{i}")

        cachedProteinFP = open(self.proteinFasta)
        cachedCDNAFP = open(self.proteinFasta)
        for sample in samplesNotLoaded:
            self.loadProteinSample(sample, cachedProteinFP, cachedCDNAFP)
        cachedProteinFP.close()
        cachedCDNAFP.close()

    def loadProteinSample(self, sample:Sample, cachedProteinFP, cachedCDNAFP) -> None:
        proteins = self.c2p[sample.id] if sample.id in self.c2p else list()
        sample.proteins = list()
        sample.cDNAs = list()
        for protein in proteins:
            offset = self.cachedProteins[protein]
            cachedProteinFP.seek(offset)
            sample.proteins.append(ProteinSample(SeqRecord(
                Seq(cachedProteinFP.readline().strip()),
                id=protein,
                description=protein)))
            
            offset = self.cachedCDNAs[protein]
            cachedCDNAFP.seek(offset)
            sample.cDNAs.append(ProteinSample(SeqRecord(
                Seq(cachedProteinFP.readline().strip()),
                id=protein,
                description=protein)))
        
NucleotideUtils = NucleotideUtil()  # this is a singleton