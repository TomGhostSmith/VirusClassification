import time
import math
import multiprocessing
from tqdm import tqdm

from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module
from entity.proteinSample import ProteinSample
from moduleResult.plainResult import PlainResult
from moduleWorker.esmTaxoWorker import runSingle

if multiprocessing.current_process().name == "MainProcess":
    from entity.taxoTree import taxoTree
    from module.esmRunner import ESMRunner
    from utils.NucleotideUtils import NucleotideUtils

class ESMTaxo(Module):
    def __init__(self, modelName, maxLen, modelFolder, baseModelFolder, rank, pooling, batchSize=None, threads=None):
        super().__init__(f"ESM_taxo_{modelName}_{pooling}")
        self.name = modelName
        if (pooling not in ["vote", "sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        self.threads = threads if threads else multiprocessing.cpu_count()
        self.pooling = pooling
        self.maxLen = maxLen
        self.modelFolder = modelFolder
        self.baseModelFolder = baseModelFolder
        
        self.batchSize = batchSize
        self.rank = rank

        self.class_names = taxoTree.taxaNames[self.rank]

    def run(self, samples:list[Sample], keepProb=False, keepVotes=False, keepProteinRes=False, **kwargs)->list[PlainResult]:
        NucleotideUtils.extractProtein(samples)
        proteinsToRun:list[ProteinSample] = list()
        for sample in samples:
            for protein in sample.proteins:
                proteinsToRun.append(protein)

        model = ESMRunner(self.name, self.maxLen, self.modelFolder, self.baseModelFolder, len(self.class_names), self.batchSize)
        model.run(proteinsToRun, getProb=True)
        key = f"{self.name}_prob"

        # keepProteinRes = True
        IOUtils.showInfo(f"keepProb={keepProb}, keepVotes={keepVotes}, keepProteinRes={keepProteinRes}")

        ctx = multiprocessing.get_context("spawn")


        results = [None] * len(samples)

        # chunkSize = len(samples) / self.threads / 4  # each thread run 4 chunk
        chunkSize = 50
        chunkCount = math.ceil(len(samples) / chunkSize)
        # bar = tqdm(total=chunkCount, desc="getResult")
        bar = tqdm(total=chunkCount, desc="get ESMTaxo Results")

        with ctx.Pool(processes=self.threads) as pool:
            # asyncResults = [pool.apply_async(runSingle, [sample, keepVotes, keepProteinRes, self.pooling, self.class_names, self.name, i]) for i, sample in enumerate(samples)]
            asyncResults = []
            for i in range(chunkCount):
                ss = [s.simplify(proteinInfos=[f"{self.name}_prob"]) for s in samples[i * chunkSize : (i + 1)*chunkSize]]
                indexRange = range(i * chunkSize, min((i + 1)*chunkSize, len(samples)))
                asyncResults.append(pool.apply_async(runSingle, [ss, keepVotes, keepProteinRes, self.pooling, self.class_names, self.name, indexRange]))
            pool.close()
            while asyncResults:
                for asyncResult in asyncResults[:]:
                    if asyncResult.ready():
                        # IOUtils.showInfo(f"received")
                        for res, voteDict, proteinRes, index in asyncResult.get():
                            results[index] = res
                            # results[index] = [PlainResult(p, s) for p, s in res] if res else None
                            sample:Sample = samples[index]
                            if (keepVotes and voteDict):
                                sample.info[f"{self.moduleName}_votes"] = voteDict
                            if (keepProteinRes):
                                for protein, rawScores in zip(sample.proteins, proteinRes):
                                    # protein.addResult(self.moduleName, rawScores)
                                    protein.addResult(self.moduleName, [PlainResult(p, s) for p, s in rawScores] if rawScores else None)
                        bar.update(1)
                        asyncResults.remove(asyncResult)
                time.sleep(1)
            pool.join()
            bar.close()

        if (keepProb):
            for p in proteinsToRun:
                p.info[f"{self.moduleName}_prob"] = p.info.pop(key, None)
        else:
            for p in proteinsToRun:
                p.info.pop(key, None)
        return results
    
def getProbs(name, maxLen, modelFolder, baseModelFolder, n_class, batchSize, proteinsToRun):
    model = ESMRunner(name, maxLen, modelFolder, baseModelFolder, n_class, batchSize)
    model.run(proteinsToRun, getProb=True)
    key = f"{name}_prob"
    results = [p.info[key] for p in proteinsToRun]
    return results