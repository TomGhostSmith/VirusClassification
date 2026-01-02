import os
import sys
import json
import numpy
import pandas
import matplotlib.pyplot as plt
sys.path.append('./code')

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils
from prototype.module import Module

from config import config

modelRoot = "/Data/VirusClassification/model"
outputRoot = "/Data/VirusClassification"
queryFilePath = ""
querySubsetFilePath = None
config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

from module.pipeline import Pipeline
from module.virusPredModule import VirusPred
from module.minimapThresholdModule import MinimapThresholdModule
from module.esmIdentify import esmIdentify
from module.minimapMLMergeModule import MinimapMLMergeModule
from module.minimapThreshRankModule import MinimapThreshRankModule
from module.mlModule import MLModule
from module.vcontact import Vcontact

from module.phagcn import PhaGCN
from module.genomad import Genomad
from module.blast import Blast
from module.kraken import Kraken
from module.minimap import Minimap
from module.vitap import VITAP
from module.cat import CAT


from entity.taxoTree import taxoTree

# trainset = "VMRv4_ML_train"

def generateVcontactTrain(faaFile, p2cFile, taxoFile, trainset):
    # 1. read in all the training reference
    samples = IOUtils.loadSamples(f"{config.modelRoot}/{trainset}/{trainset}.fasta")
    
    # 2. use prodigal to convert them into proteins, and convert that to .gz file
    compress = False
    if (faaFile.endswith(".gz")):
        rawFaaFile = faaFile[:-3]
        compress = True
    else:
        rawFaaFile = faaFile
    NucleotideUtils.extractProtein(samples)
    NucleotideUtils.writeSampleProteinFasta(samples, rawFaaFile)
    if (compress):
        IOUtils.compress_to_gz(rawFaaFile)
        os.remove(rawFaaFile)

    # 3. build a protien2contig csv
    pIDs = list()
    cIDs = list()
    indexes = list()
    for sample in samples:
        for protein in sample.proteins:
            pIDs.append(protein.id)
            cIDs.append(sample.id)
            indexes.append(f"segment{protein.index}")
    df = pandas.DataFrame({
        "protein_id": list(pIDs),
        "contig_id": list(cIDs),
        "keywords": list(indexes)
    })
    df.to_csv(p2cFile)

    # 4. build a taxo csv, containing Organism/Name, origin, order, family, genus
    taxoInfo = list()
    for sample in samples:
        id = taxoTree.ICTVTree.accession2ID[sample.id]
        node = taxoTree.ICTVTree.species[id]
        taxo = dict()
        for n in node.path:
            taxo[n.rank] = n.name
        taxoInfo.append((trainset, 
                         sample.id, 
                         taxo.get("kingdom"), 
                         taxo.get("phylum"), 
                         taxo.get("class"), 
                         taxo.get("order"), 
                         taxo.get("family"), 
                         taxo.get("genus")))
    
    origin, name, kingdom, phylum, claz, order, family, genus = zip(*taxoInfo)
    df = pandas.DataFrame({
        "origin": origin,
        "Organism/Name": name,
        "kingdom": kingdom,
        "phylum": phylum,
        "class": claz,
        "order": order,
        "family": family,
        "genus": genus
    })

    df.to_csv(taxoFile)
    IOUtils.showInfo("Done")


def main():
    faaFile = "working/VMRv4_faa.gz"
    p2cFile = "working/VMRv4_p2c.csv"
    taxoFile = "working/VMRv4_taxo.csv"
    # generateVcontactTrain(faaFile, p2cFile, taxoFile, "VMRv4_ML_train")
    generateVcontactTrain(faaFile, p2cFile, taxoFile, "VMRv4")

if (__name__ == "__main__"):
    main()