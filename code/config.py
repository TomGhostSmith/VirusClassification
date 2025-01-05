import os

class Config():
    def __init__(self) -> None:
        # self.majorDataset = 'refseq_2024_test'
        self.majorDataset = 'Challenge'
        
        # self.minorDataset = 'genus'
        self.minorDataset = None

        self.tempFolder = "/Data/VirusClassification/temp"

        self.ncbiFolder = "/Data/ICTVData/NCBI"
        self.ncbiAssemblyFolder = "/Data/ICTVData/NCBI/Assembly"
        self.ncbiNucleotideFolder = "/Data/ICTVData/NCBI/Nucleotide"

        ranks = [
            "root",
            "superkingdom",
            "realm",
            "kingdom",
            "subkingdom",
            "phylum",
            "subphylum",
            "class",
            "subclass",
            "order",
            "suborder",
            "family",
            "subfamily",
            "genus",
            "subgenus",
            "species",
            "serogroup",
            "serotype",
            "genotype",
            "clade",
            "isolate",
            "no rank"
        ]
        self.rankLevels = {rank: idx for (idx, rank) in enumerate(ranks)}

        self.evaluationRanks = ['superkingdom', 'realm', 'kingdom', 'subkingdom', 'phylum', 'subphylum', 'class', 'subclass', 'order', 'suborder', 'family', 'subfamily', 'genus', 'subgenus', 'species']
        self.minorDatasetRanks = ["nonVirus", "superkingdom", "realm", "kingdom", "phylum", "class", "order", "family", "genus", "species"]

        self.updatePath()

    
    def updatePath(self):
        if (self.minorDataset is None):
            self.resultBase = f"/Data/VirusClassification/results/{self.majorDataset}"
            self.datasetBase = f"/Data/VirusClassification/dataset/{self.majorDataset}"
            self.datasetName = self.majorDataset
        else:
            self.resultBase = f"/Data/VirusClassification/results/{self.majorDataset}/{self.minorDataset}"
            self.datasetBase = f"/Data/VirusClassification/dataset/{self.majorDataset}/{self.minorDataset}"
            self.datasetName = f"{self.majorDataset}-{self.minorDataset}"
        
        if (not os.path.exists(self.resultBase)):
            os.makedirs(self.resultBase)
        if (not os.path.exists(self.datasetBase)):
            os.makedirs(self.datasetBase)

# note for path and signal use:
# use ';' for separate params. e.g. a=1;b=2
# use '_' for separate words. e.g. contig_most_frequent
# use '-' for connect model and params. e.g. minimap-ref=xx;mode=xx
# use '.' for connect different models. e.g. minimap.ml
# use '|' for file name split. e.g. statistics|n=100;incl=50
# never use ',' because it will disturbe csv file 

config = Config()