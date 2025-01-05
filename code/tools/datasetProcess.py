import os
import re
from tqdm import tqdm
import json
import random

from config import config
from entity.taxoTree import taxoTree
from entity.taxoNode import TaxoNode
from entity.dataset import Dataset

from utils import IterUtils

# combine dataset if needed
def combineChallenge(challengeFolder, datasetName):
    config.majorDataset = datasetName
    config.minorDataset = None
    config.updatePath()

    files = sorted(os.listdir(challengeFolder))
    # random.shuffle(files)
    targetFp = open(f"{config.datasetBase}/{config.datasetName}.fasta", 'wt')
    progress = tqdm(total=len(files), desc='Merge files')
    for file in files:
        if (not file.endswith('.fasta')):
            progress.update(1)
            continue
        with open(f"{challengeFolder}/{file}") as fp:
            line = fp.readline()
            if (not line.startswith(">")):
                print(f"{file} not start with >")
            targetFp.write(line)

            lines = fp.readlines()
            for line in lines:
                content = line.strip()
                if (not re.match(r'^[ATCGN]*', content)):
                    print(f'{file} not align ATCGN')
                targetFp.write(line)
        progress.update(1)
    targetFp.close()
    progress.close()

def splitByRank(datasetName, getAnswerTaxoNodeFunc):
    config.majorDataset = datasetName
    config.minorDataset = None
    config.updatePath()

    for n in config.minorDatasetRanks:
        os.makedirs(f"{config.datasetBase}/{n}")

    fps = {
        n: open(f"{config.datasetBase}/{n}/{datasetName}-{n}.fasta", 'wt') for n in config.minorDatasetRanks
    }
    targetFP = {
        n: fps[n] for n in config.minorDatasetRanks
    }
    targetFP["subphylum"] = fps["phylum"]
    targetFP["subclass"] = fps["class"]
    targetFP["suborder"] = fps["order"]
    targetFP["subfamily"] = fps["family"]
    targetFP["subgenus"] = fps["genus"]

    title = None
    rank = None
    rankDict = dict()
    with open(f"{config.datasetBase}/{config.datasetName}.fasta") as fp:
        for line in fp:
            if (line.startswith(">")):
                title = line.strip()[1:]
                taxoNode:TaxoNode = getAnswerTaxoNodeFunc(title)
                if (taxoNode is not None):
                    rank = taxoNode.ICTVNode.rank
                else:
                    rank = "nonVirus"
                targetFP[rank].write(line)
                rankDict[title] = rank
            else:
                content = line.strip()
                if (not re.match(r'^[ATCGN]*', content)):
                    print(f'{title} not align ATCGN')
                targetFP[rank].write(line)

    with open(f"{config.datasetBase}/rank.json", 'wt') as fp:
        json.dump(rankDict, fp, indent=2, sort_keys=True)

    for fp in fps.values():
        fp.close()


# generate answer, length, isATCG, etc.
def analyseDataset(getAnswerTaxoNodeFunc):
    lengthDict = dict()
    isATCGDict = dict()
    answerDict = dict()

    length = 0
    isATCG = True
    title = None

    with open(f"{config.datasetBase}/{config.datasetName}.fasta") as fp:
        for line in fp:
            if (line.startswith(">")):
                if (title is not None):
                    lengthDict[title] = length
                    isATCGDict[title] = isATCG

                title = line.strip()[1:]
                answerNode:TaxoNode = getAnswerTaxoNodeFunc(title)
                if (answerNode is not None):
                    answerDict[title] = answerNode.ICTVName
                else:
                    answerDict[title] = 'no answer'
                length = 0
                isATCG = True
            else:
                content = line.strip()
                length += len(content)
                if (not re.match(r'^[ATCG]*', content)):
                    isATCG = False
    
    # update for the last title
    # use this if-statement in case there is no sample at all
    if (title is not None):
        lengthDict[title] = length
        isATCGDict[title] = isATCG

    with open(f"{config.datasetBase}/length.json", 'wt') as fp:
        json.dump(lengthDict, fp, indent=2, sort_keys=True)
    with open(f"{config.datasetBase}/isATCG.json", 'wt') as fp:
        json.dump(isATCGDict, fp, indent=2, sort_keys=True)
    with open(f"{config.datasetBase}/answer.json", 'wt') as fp:
        json.dump(answerDict, fp, indent=2, sort_keys=True)

def getRank(title):
    terms = title.split("|")
    superkingdom = terms[0]
    if (superkingdom == "Viruses"):
        id = terms[2]
        taxoNode = taxoTree.getTaxoNodeFromNCBI(NCBIID=id)
        return taxoNode
    else:
        return None
    
def analyseNCBIDatasets():
    analyseDataset(getRank)

def main():
    # combineChallenge("/Data/ICTVData/dataset/Challenge/fasta", "Challenge")
    # analyseDataset(lambda x:None)

    # combineChallenge("/Data/ICTVData/dataset/refseq_2024_test/fasta", "refseq_2024_test")
    # splitByRank('refseq_2024_test', getRank)
    # IterUtils.iterDatasets(Dataset('refseq_2024_test', config.minorDatasetRanks), analyseNCBIDatasets)
    
    combineChallenge("/Data/ICTVData/dataset/genbank_2024_test/fasta", "genbank_2024_test")
    splitByRank('genbank_2024_test', getRank)
    IterUtils.iterDatasets(Dataset('genbank_2024_test', config.minorDatasetRanks), analyseNCBIDatasets)


if (__name__ == '__main__'):
    main()