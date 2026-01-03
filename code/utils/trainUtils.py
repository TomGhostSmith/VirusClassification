import json

from config import config
from utils import IOUtils
from entity.sample import Sample
from entity.taxoTree import taxoTree

datasetRoot = "/Data/VirusClassification/dataset"

def loadTrainsetSamples(dataset, evaluationMethod, subset="all"):
    filePath = f"{datasetRoot}/{dataset}/{dataset}.fasta"
    if (subset != 'all'):
        subsetFilePath = f"{datasetRoot}/{dataset}/{subset}.txt"
    else:
        subsetFilePath = None
    samples = IOUtils.loadSamples(filePath, subsetFilePath)

    with open(f"{datasetRoot}/{dataset}/answer_{evaluationMethod}.json") as fp:
        stdResults = json.load(fp)

    for id, std in stdResults.items():
        if (std == 'no answer'):
            stdResults[id] = None
        else:
            stdResults[id] = taxoTree.getTaxoNodeFromICTV(ICTVName=std)

    for sample in samples:
        sample.info["stdResult"] = stdResults.get(sample.id)

    return samples