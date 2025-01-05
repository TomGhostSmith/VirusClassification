from config import config
from entity.dataset import Dataset


def iterDatasets(dataset:Dataset, func):
    for config.majorDataset, config.minorDataset in dataset.iterDatasets():
        config.updatePath()
        func()