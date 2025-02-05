import datetime
import os
import sys
from Bio import SeqIO
from entity.sample import Sample

def showInfo(message, typ='INFO'):
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    msg = f"{currentTime} ({os.getpid()}) [{typ}] {message}\n"
    if (typ == 'WARN' or typ == 'PROC'):
        sys.stderr.write(msg)
    else:
        sys.stdout.write(msg)

def writeSampleFasta(samples:list[Sample], targetFile:str):
    with open(targetFile, 'wt') as fp:
        for sample in samples:
            SeqIO.write(sample.seq, fp, 'fasta')

# note: here we only consider the scenario that there is only one subset file
def loadSamples(fastaFile:str, subsetFile:str=None)->list[Sample]:
    interestedSampleIDs = None
    if (subsetFile is not None):
        with open(subsetFile) as fp:
            interestedSampleIDs = {line.strip() for line in fp.readlines()}
    samples:list[Sample] = list()
    for record in SeqIO.parse(fastaFile, 'fasta'):
        if interestedSampleIDs is None or record.id in interestedSampleIDs:
            samples.append(Sample(seq=record))
    
    return samples

def findSample(samples:list[Sample], sampleID:str):
    for sample in samples:
        if sample.id == sampleID:
            return sample
    return None