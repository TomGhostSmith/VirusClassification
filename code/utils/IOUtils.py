import os
import sys
import math
import gzip
import shutil
import datetime
import subprocess
import multiprocessing
from tqdm import tqdm
from Bio import SeqIO
from entity.sample import Sample
from entity.proteinSample import ProteinSample

def showInfo(message, typ='INFO'):
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    msg = f"{currentTime} ({os.getpid()}) [{typ}] {message}\n"
    if (typ == 'WARN' or typ == 'PROC'):
        sys.stderr.write(msg)
    else:
        sys.stdout.write(msg)

def writeSampleFasta(samples:list[Sample]|list[ProteinSample], targetFile:str, append=False):
    mode = 'at' if append else 'wt'
    with open(targetFile, mode) as fp:
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

def loadProteinSamples(fastaFile:str, subsetFile:str=None)->list[ProteinSample]:
    interestedSampleIDs = None
    if (subsetFile is not None):
        with open(subsetFile) as fp:
            interestedSampleIDs = {line.strip() for line in fp.readlines()}
    samples:list[ProteinSample] = list()
    for record in SeqIO.parse(fastaFile, 'fasta'):
        if interestedSampleIDs is None or record.id in interestedSampleIDs:
            samples.append(ProteinSample(seq=record))
    
    return samples

def findSample(samples:list[Sample], sampleID:str):
    for sample in samples:
        if sample.id == sampleID:
            return sample
    return None

def checkAndEmptyFolder(folder):
    if os.path.exists(folder) and os.path.isdir(folder):
        showInfo(f"Emptying {folder}", "WARN")
        shutil.rmtree(folder)
    os.makedirs(folder)

def appendFile(source, dest, buffer_size=1024*1024):
    with open(source, 'r') as src, open(dest, 'a') as dst:
        while True:
            chunk = src.read(buffer_size)
            if not chunk:
                break
            dst.write(chunk)

def compress_to_gz(input_path, output_path=None):
    if output_path is None:
        output_path = input_path + '.gz'
    
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_path

def runSingleProdigal(dnaFasta, outputFasta, idx=None):
    subprocess.run(f"prodigal-gv -i {dnaFasta} -a {outputFasta} -p meta", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return idx

# extract protein with prodigal-gv. Return tuples (pID, cID, index)
def extractProtein(samples, outputFile, samplePerThread=100, threads=multiprocessing.cpu_count())-> list[tuple[str, str, int]]:
    if (os.path.exists(outputFile)):
        showInfo(f"File {outputFile} exists, will be overwritten")
        os.remove(outputFile)

    splitCount = math.ceil(len(samples)/samplePerThread)
    tempFileName = f"{outputFile}.DNA"

    contigIDs:list[tuple[str, str, int]] = list()

    with multiprocessing.Pool(threads) as pool:
        asyncResults = list()
        for i in range(splitCount):
            writeSampleFasta(samples[samplePerThread * i : samplePerThread * (i+1)], f"{tempFileName}.{i}")
    
            asyncResults.append(pool.apply_async(runSingleProdigal, 
                                            [f"{tempFileName}.{i}", f"{outputFile}.{i}", 1]))
            
        for asyncResult in tqdm(asyncResults):
            idx = asyncResult.get()
            proteinSamlpes = loadProteinSamples(f"{outputFile}.{idx}")
            for protein in proteinSamlpes:
                # contigIDs[protein.id] = protein.contigID
                contigIDs.append((protein.id, protein.contigID, f"segment{protein.index}"))
            appendFile(f"{outputFile}.{idx}", outputFile, buffer_size=16*1024*1024)
            
            
    # subprocess.run(f"prodigal-gv -i {self.tempDNAFasta} -a {self.tempProFasta} -p meta", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(splitCount):
        os.remove(f"{tempFileName}.{i}")
        os.remove(f"{outputFile}.{i}")

    return contigIDs