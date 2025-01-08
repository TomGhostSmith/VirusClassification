import os
from Bio import SeqIO

from config import config
from utils import IOUtils
from module.esm import ESM
from entity.dataset import Dataset
from entity.evaluator import Evaluator
from module.virusPredModule import VirusPred
from module.minimapThresholdModule import MinimapThresholdModule

def generateVirusDataset(dataset:Dataset):
    minimapTh = MinimapThresholdModule(reference='VMRv4', factors=['60'])
    virusPred = VirusPred([minimapTh])
    evaluator = Evaluator([virusPred])
    samples = evaluator.loadSamples(dataset)
    virusNames = {sample.query for sample in samples if sample.results[virusPred.moduleName] is not None}

    IOUtils.showInfo('Loading sequences')
    allRecords = list()
    for _, major, minor in dataset.iterDatasets():
        config.majorDataset = major
        config.minorDataset = minor
        config.updatePath()

        with open(f"{config.datasetBase}/{config.datasetName}.fasta") as fp:
            for record in SeqIO.parse(fp, 'fasta'):
                allRecords.append(record)

        config.minorDataset = None
        config.updatePath()
    
    # no matter how many (minor-)dataset selected, the screened virus sequence should be store at a new minor-dataset
    config.minorDataset = 'virus'
    config.updatePath()
    if (os.path.exists(f"{config.datasetBase}/{config.datasetName}.fasta")):
        raise FileExistsError(f"subset 'virus' already exists. Please delete it or rename it")
    virusFP = open(f"{config.datasetBase}/{config.datasetName}.fasta", 'wt')

    config.minorDataset = 'nonVirus'
    config.updatePath()
    if (os.path.exists(f"{config.datasetBase}/{config.datasetName}.fasta")):
        raise FileExistsError(f"subset 'nonVirus' already exists. Please delete it or rename it")
    nonVirusFP = open(f"{config.datasetBase}/{config.datasetName}.fasta", 'wt')

    IOUtils.showInfo('Writting sequences to subsets')
    for record in allRecords:
        if (record.id in virusNames):
            SeqIO.write(record, virusFP, 'fasta')
        else:
            SeqIO.write(record, nonVirusFP, 'fasta')

    virusFP.close()
    nonVirusFP.close()


def main():
    dataset = Dataset("Challenge")
    generateVirusDataset(dataset)
    
if (__name__ == '__main__'):
    main()