import os

class Config():
    def __init__(self) -> None:
        # self.majorDataset = 'refseq_2024_test'
        self.majorDataset = ''
        
        # self.minorDataset = 'genus'
        # self.minorDataset = None

        self.dataRoot = "/Data/ICTVPublish"
        # self.dataRoot = "/Data/VirusClassification"



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

        self.resultCSVRanks = [
            "SequenceID", 
            "Realm (-viria)", 
            "Realm_score",
            "Subrealm (-vira)", 
            "Subrealm_score",
            "Kingdom (-virae)",
            "Kingdom_score",
            "Subkingdom (-virites)",
            "Subkingdom_score",
            "Phylum (-viricota)",
            "Phylum_score",
            "Subphylum (-viricotina)",
            "Subphylum_score",
            "Class (-viricetes)",
            "Class_score",
            "Subclass (-viricetidae)",
            "Subclass_score",
            "Order (-virales)",
            "Order_score",
            "Suborder (-virineae)",
            "Suborder_score",
            "Family (-viridae)",
            "Family_score",
            "Subfamily (-virinae)",
            "Subfamily_score",
            "Genus (-virus)",
            "Genus_score",
            "Subgenus (-virus)",
            "Subgenus_score",
            "Species (binomial)",
            "Species_score"
        ]

        self.resultRanks = ["realm", "subrealm", "kingdom", "subkingdom", "phylum", "subphylum", "class", "subclass", "order", "suborder", "family", "subfamily", "genus", "subgenus", "species"]


        self.evaluationRanks = ['superkingdom', 'realm', 'kingdom', 'subkingdom', 'phylum', 'subphylum', 'class', 'subclass', 'order', 'suborder', 'family', 'subfamily', 'genus', 'subgenus', 'species']
        self.minorDatasetRanks = ["nonVirus", "superkingdom", "realm", "kingdom", "phylum", "class", "order", "family", "genus", "species"]

        self.esmBatchSize = 64
        self.mlBatchSize = 64

        self.outputName = None

        # self.updatePath()

    
    def updatePath(self):
        self.tempFolder = f"{self.dataRoot}/cache"
        self.refFolder = f"{self.dataRoot}/model"

        self.ncbiFolder = f"{self.dataRoot}/NCBI"
        self.ncbiAssemblyFolder = f"{self.dataRoot}/NCBI/Assembly"
        self.ncbiNucleotideFolder = f"{self.dataRoot}/NCBI/Nucleotide"
        self.modelRoot = f"{self.dataRoot}/model"
    
        # if (self.minorDataset is None):
        self.resultBase = f"{self.dataRoot}/results/{self.majorDataset}"
        self.datasetBase = f"{self.dataRoot}/dataset/{self.majorDataset}"
        self.datasetName = self.majorDataset
        # else:
            # self.resultBase = f"/Data/VirusClassification/results/{self.majorDataset}/{self.minorDataset}"
            # self.datasetBase = f"/Data/VirusClassification/dataset/{self.majorDataset}/{self.minorDataset}"
            # self.datasetName = f"{self.majorDataset}-{self.minorDataset}"
        
        self.virusPredResultFolder = f"{self.dataRoot}/results/{self.majorDataset}/VirusPred"
        self.MLResultFolder = f"{self.dataRoot}/results/{self.majorDataset}/MLResult"
        
        # if (not os.path.exists(self.datasetBase)):
        #     raise ValueError("dataset folder not found")
        # if(not os.path.exists(self.refFolder)):
        #     raise ValueError("reference folder not found")
        if(not os.path.exists(self.modelRoot)):
            raise ValueError("model folder not found")
        if (not os.path.exists(self.tempFolder)):
            os.makedirs(self.resultBase)
        if (not os.path.exists(self.resultBase)):
            os.makedirs(self.resultBase)
        if (not os.path.exists(self.virusPredResultFolder)):
            os.makedirs(self.virusPredResultFolder)

# note for path and signal use:
# use ';' for separate params. e.g. a=1;b=2
# use '_' for separate words. e.g. contig_most_frequent
# use '-' for connect model and params. e.g. minimap-ref=xx;mode=xx
# use '.' for connect different models. e.g. minimap.ml
# use '!' for file name split. e.g. statistics!n=100;incl=50
# never use ',' because it will disturbe csv file 

config = Config()