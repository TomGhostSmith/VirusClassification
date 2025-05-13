import os
import sys
import json
from tqdm import tqdm
import pandas
import subprocess
import multiprocessing
sys.path.append("code")

from config import config
from utils import IOUtils

modelRoot = "/Data/VirusClassification/model"
outputRoot = "/Data/VirusClassification"
queryFilePath = ""
querySubsetFilePath = None
config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)


# total line in AllNucleotide.fa: 4743378784
# total line in metadata.csv: 13804005 (including title)

# class NucleotideUtils():
#     def __init__(self) -> None:
#         pass
nucleotideLineCount = 4743378784
metadataLineCount = 13804005

def downloadOne(accession, targetFile):
    with open(targetFile, 'wt') as fp:
        for _ in range(5):
            result = subprocess.run(f"efetch -db nuccore -id {accession} -format fasta", shell=True, stdout=fp).returncode
            if (result == 0):
                return None
    
    IOUtils.showInfo(f"fail to download {accession}")
    return accession

def createFolders(version):
    fnaFolder = f"{config.modelRoot}/NCBI/Nucleotide/{version}/fna"
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/AllNuclMetadata.csv") as fp:
    # with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/2025SpringNuclMetadata.csv") as fp:
        fp.readline()
        for line in tqdm(fp, total=metadataLineCount - 1):
            name = line[:line.index(".")]
            chunks = [name[i:i+3] for i in range(0, len(name), 3)]
            folderPath = os.path.join(fnaFolder, *chunks[:-1])
            os.makedirs(folderPath, exist_ok=True)

def downloadAccessions(accessions, version):
    params = list()
    fnaFolder = f"{config.modelRoot}/NCBI/Nucleotide/{version}/fna"
    for accession in accessions:
            name = accession[:accession.index(".")]
            chunks = [name[i:i+3] for i in range(0, len(name), 3)]
            folderPath = os.path.join(fnaFolder, *chunks)
            params.append((accession, f"{folderPath}.fasta"))
    
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/failDownload.txt", 'wt') as fp:
        bar = tqdm(total=len(params))
        with multiprocessing.Pool(16) as pool:
            asyncResults = [pool.apply_async(downloadOne, param) for param in params]
            while asyncResults:
                for asyncResult in asyncResults[:]:
                    if (asyncResult.ready()):
                        result = asyncResult.get()
                        bar.update(1)
                        asyncResults.remove(asyncResult)
                        if (result is not None):
                            fp.write(f"{result}\n")
            pool.close()
            pool.join()
        bar.close()


def downloadFile(version, fileName):
    IOUtils.showInfo("Due to API rate limit of NCBI, you cannot download multiple accessions at the same time")
    raise Exception("Method abandoned")
    accessions = list()
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/{fileName}") as fp:
        fp.readline()
        for line in fp:
            accessions.append(line[:line.index(",")])
    downloadAccessions(accessions, version)

        

# problem: PQ802350 is not in the metadata, but in the fasta file
def splitFile(version):
    targetFP = None
    fnaFolder = f"{config.modelRoot}/NCBI/Nucleotide/{version}/fna"
    nameFP = open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/names.txt", 'wt')
    missingFolderFP = open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/missingFolder.txt", 'wt')
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/AllNucleotide.fa") as fp:
        for line in tqdm(fp, total=nucleotideLineCount):
            if line[0] == ">":
                if (targetFP is not None):
                    targetFP.close()
                nameFP.write(line[1:])
                name = line[1:line.index(".")]
                chunks = [name[i:i+3] for i in range(0, len(name), 3)]
                fileName = f"{chunks[-1]}.fasta"
                folderPath = os.path.join(fnaFolder, *chunks[:-1])
                if (not os.path.exists(folderPath)):
                    os.makedirs(folderPath, exist_ok=True)
                    missingFolderFP.write(f"{name}\n")
                filePath = os.path.join(fnaFolder, *chunks[:-1], fileName)
                targetFP = open(filePath, 'wt')
            targetFP.write(line)
        
        if (targetFP is not None):
            targetFP.close()
    
    nameFP.close()
    missingFolderFP.close()


def loadMetadata(version):
    metaFile = f"{config.modelRoot}/NCBI/Nucleotide/{version}/AllNuclMetadata.csv"
    # with tqdm(total=metadataLineCount - 1, desc="loading meta data") as pbar:
    #     chunks = list()
    #     for chunk in pandas.read_csv(metaFile, quotechar='"', chunksize=65536, low_memory=False):
    #         chunks.append(chunk)
    #         pbar.update(len(chunk))
    
    # dataFrame = pandas.concat(chunks, ignore_index=True)

    dataFrame = pandas.read_csv(metaFile, low_memory=False)
    dataFrame.columns = [col.lstrip("#") for col in dataFrame.columns]

    return dataFrame

def checkMetaSeqAlignment(version):
    dataFrame = loadMetadata(version)
    metaNames = dataFrame["Accession"]
    metaNameset = set(metaNames)
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/names.txt") as fp:
        seqNames = [name[:-1] for name in fp.readlines()]
    seqNameset = set(seqNames)
    
    metaUnique = metaNameset - seqNameset
    seqUnique = seqNameset - metaNameset

    print(f"meta unique: {len(metaUnique)}")
    print(f"seq unique: {len(seqUnique)}")

    with open(f'{config.modelRoot}/NCBI/Nucleotide/{version}/metaUniqueNames.txt', 'wt') as fp:
        for name in metaUnique:
            fp.write(f"{name}\n")

    with open(f'{config.modelRoot}/NCBI/Nucleotide/{version}/seqUniqueNames.txt', 'wt') as fp:
        for name in seqUnique:
            fp.write(f"{name}\n")

    fnaFolder = f"{config.modelRoot}/NCBI/Nucleotide/{version}/fna"
    count = 0
    for name in metaUnique:
        name = name[:name.index(".")]
        chunks = [name[i:i+3] for i in range(0, len(name), 3)]
        path = f"{os.path.join(fnaFolder, *chunks)}.fasta"
        if (os.path.exists(path)):
            if (os.path.getsize(path) == 0):
                os.remove(path)
                count += 1
            else:
                with open(path) as fp:
                    line = fp.readline()
                if not line.startswith(">"):
                    os.remove(path)
                    count += 1
    IOUtils.showInfo(f"successfully removed {count} empty files")


def getHost(version):
    metaData = loadMetadata(version)
    host = dict()
    for row in tqdm(metaData.itertuples(), total=metadataLineCount-1):
        if ((not pandas.isnull(row.Species)) and (not pandas.isnull(row.Host))):
            if row.Species in host:
                host[row.Species].add(row.Host)
            else:
                host[row.Species] = {row.Host}
    
    for k, v in host.items():
        host[k] = list(v)
    
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/rawHosts.json", 'wt') as fp:
        json.dump(host,  fp, indent=2)

def getGenBank(version):
    metaData = loadMetadata(version)
    tarFP = open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/genbank.accession", 'wt')
    tarFP.write("Accession,Species\n")
    for row in tqdm(metaData.itertuples(), total=metadataLineCount-1):
        if ((not pandas.isnull(row.Species)) and (not pandas.isnull(row.Accession))):
            tarFP.write(f"{row.Accession},{row.Species}\n")
    
    tarFP.close()

def getNewReleaseGenBank(version):
    metaData = loadMetadata(version)
    tarFP = open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/genbank_2025Spring.accession", 'wt')
    tarFP.write("Accession,Species\n")
    for row in tqdm(metaData.itertuples(), total=metadataLineCount-1):
        if ((not pandas.isnull(row.Species)) and (not pandas.isnull(row.Accession)) and row.Release_Date.startswith("2025")):
            tarFP.write(f"{row.Accession},{row.Species}\n")
    
    tarFP.close()


# note: there are some sequences in the fasta but has no metadata in the csv file. Please check missingFolder.txt
def splitAllFasta(version):
    # createFolders(version)        # takes several minutes
    splitFile(version)            # takes about an hour

def main():
    version = "20250505"
    # createFolders(version)
    # downloadFile(version, "2025SpringNuclMetadata.csv")
    # splitAllFasta(version)
    checkMetaSeqAlignment(version)
    # getHost(version)
    # getGenBank(version)
    # getNewReleaseGenBank(version)

if (__name__ == '__main__'):
    main()