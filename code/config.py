# reconstructed
import os

class Config():
    def __init__(self) -> None:        
        self.modelRoot = ""
        self.outputRoot = ""
        self.queryFilePath = ""
        self.querySubsetFilePath = None

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

        # self.updatePath()

    
    def updatePath(self):
        def getFileNameWithoutExt(filePath):
            return os.path.splitext(os.path.basename(filePath))[0]
        
        self.cacheResultFolder = f"{self.outputRoot}/cache/CachedResults"
        self.cacheAnalysisFolder = f"{self.outputRoot}/cache/CachedAnalysis"
        self.datasetName = getFileNameWithoutExt(self.queryFilePath)
        self.subsetName = getFileNameWithoutExt(self.querySubsetFilePath) if self.querySubsetFilePath is not None else "all"

        if(not os.path.exists(self.modelRoot)):
            raise ValueError("model folder not found")
        os.makedirs(self.cacheResultFolder, exist_ok=True)
        os.makedirs(self.cacheAnalysisFolder, exist_ok=True)


config = Config()