import multiprocessing.shared_memory
import os
import sys
import math
import gzip
import numpy
import base64
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
    # with open("nohup.txt", 'at') as fp:
    #     fp.write(msg)
    if (typ == 'WARN' or typ == 'PROC'):
        sys.stderr.write(msg)
    else:
        sys.stdout.write(msg)

def writeSampleFasta(samples:list[Sample], targetFile:str, append=False):
    mode = 'at' if append else 'wt'
    with open(targetFile, mode) as fp:
        for sample in samples:
            SeqIO.write(sample.seq, fp, 'fasta')

def writeSampleProteinFasta(samples:list[Sample], targetFile:str, append=False, withHead=False):
    mode = 'at' if append else 'wt'
    with open(targetFile, mode) as fp:
        for sample in samples:
            for protein in sample.proteins:
                if (withHead):
                    fp.write(f">{protein.head}\n{protein.seq.seq}\n")
                else:
                    SeqIO.write(protein.seq, fp, 'fasta')

def writeSampleCDNAFasta(samples:list[Sample], targetFile:str, append=False, withHead=False):
    mode = 'at' if append else 'wt'
    with open(targetFile, mode) as fp:
        for sample in samples:
            for protein in sample.cDNAs:
                if (withHead):
                    fp.write(f">{protein.head}\n{protein.seq.seq}\n")
                else:
                    SeqIO.write(protein.seq, fp, 'fasta')

# note: here we only consider the scenario that there is only one subset file
def loadSamples(fastaFile:str, subsetFile:str=None, subset:list=None)->list[Sample]:
    interestedSampleIDs = None
    if (subsetFile is not None):
        with open(subsetFile) as fp:
            interestedSampleIDs = {line.strip() for line in fp.readlines()}
        if (subset is not None):
            interestedSampleIDs = interestedSampleIDs & set(subset)
    elif (subset is not None):
        interestedSampleIDs = set(subsetFile)
    samples:list[Sample] = list()
    for record in SeqIO.parse(fastaFile, 'fasta'):
        if interestedSampleIDs is None or record.id in interestedSampleIDs:
            samples.append(Sample(seq=record))
    
    return samples

def loadProteinSamples(fastaFile:str, subsetFile:str=None, subset:list=None)->list[ProteinSample]:
    if (not os.path.exists(fastaFile)):
        return list()
    interestedSampleIDs = None
    if (subsetFile is not None):
        with open(subsetFile) as fp:
            interestedSampleIDs = {line.strip() for line in fp.readlines()}
        if (subset is not None):
            interestedSampleIDs = interestedSampleIDs & set(subset)
    elif (subset is not None):
        interestedSampleIDs = set(subsetFile)
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

def encodeBase64(array):
    return base64.b64encode(array.tobytes()).decode('ascii')

def decodeBase64(text, dtype=numpy.float16):
    return numpy.frombuffer(base64.b64decode(text), dtype=dtype)

def progressListener(queue, pbars):
    while True:
        msg = queue.get()
        if msg == "Done":
            for pbar in pbars:
                pbar.close()
            break
        idx, increment = msg
        pbars[idx].update(increment)

def getProgressListener(pbars):
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listener = multiprocessing.Process(target=progressListener, args=(queue, pbars))
    listener.start()
    return listener, queue

def stopProgressListener(listener, queue):
    queue.put("Done")
    listener.join()

def createSharedMemory(array: numpy.ndarray):
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=array.nbytes)
    sharedArray = numpy.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    sharedArray[:] = array[:]
    return (shm.name, array.shape, array.dtype), shm

def unlinkSharedMemory(shm):
    shm.close()
    shm.unlink()

def loadSharedMemory(name, shape, dtype):
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    return shm, numpy.ndarray(shape, dtype=dtype, buffer=shm.buf)

def closeSharedMemory(shm):
    shm.close()
