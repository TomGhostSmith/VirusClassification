from config import config
from entity.dataset import Dataset


def iterDatasets(dataset:Dataset, func, splitReports=True):
    if (splitReports):
        for d, config.majorDataset, config.minorDataset in dataset.iterDatasets():
            config.updatePath()
            func(d)
    else:
        func(dataset)