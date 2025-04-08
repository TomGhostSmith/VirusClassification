import os
import sys
import json
import numpy
import pandas
import matplotlib.pyplot as plt
sys.path.append('./code')

from utils import IOUtils
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
from module.esm import ESM
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


def generateVcontactTrain():
    # 1. read in all the training reference
    
    # 2. use prodigal to convert them into proteins, and convert that to .gz file

    # 3. build a protien2contig csv

    # 4. build a taxo csv, containing Organism/Name, origin, order, family, genus
    pass