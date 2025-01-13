class Dataset():
    def __init__(self, majorDataset, minorDatasets=None):
        self.majorDataset = majorDataset
        if (minorDatasets is None):
            self.minorDatasets = list()
        elif isinstance(minorDatasets, str):
            self.minorDatasets = [minorDatasets]
        elif isinstance(minorDatasets, list):
            self.minorDatasets = minorDatasets
        else:
            raise ValueError("minorDataset should be None, name of subdataset or a list of names of subdatasets")
    
    # def iterDatasets(self):
    #     if (self.minorDatasets is None):
    #         return [(self, self.majorDataset, None)]
    #     else:
    #         return [(Dataset(self.majorDataset, minorDataset), self.majorDataset, minorDataset) for minorDataset in self.minorDatasets]