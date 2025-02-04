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

def writeSampleFasta(samples:list[Sample], targetFile):
    with open(targetFile, 'wt') as fp:
        for sample in samples:
            SeqIO.write(sample.seq, fp, 'fasta')