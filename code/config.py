from entity.dataset import Dataset
# from moduleConfig.minimapConfig import minimapConfig

class Config():
    def __init__(self) -> None:
        minorDatasetRanks = ["nonVirus", "superkingdom", "realm", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
        # self.dataset = Dataset("Challenge")
        # self.dataset = Dataset("gen2")
        # self.dataset = Dataset("gen2_fold2020")
        # self.dataset = Dataset("gen2_fold2024")
        self.dataset = Dataset("refseq_2024_test", "genus")
        # self.dataset = Dataset("refseq_2024_test", minorDatasetRanks)
        # self.dataset = Dataset("genbank_2024_test", "genus")
        # self.dataset = Dataset("genbank_2024_test", minorDatasetRanks)

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

        if (self.dataset.isSingleRun()):
            self.majorDataset, self.minorDataset = self.dataset.iterDatasets()[0]
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

    def iterDatasets(self, func):
        for self.majorDataset, self.minorDataset in self.dataset.iterDatasets():
            self.updatePath()
            func()


config = Config()