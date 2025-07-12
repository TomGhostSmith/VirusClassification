# reconstructed
import os
import json
import pandas
from config import config
from prototype.module import Module
from moduleResult.plainResult import PlainResult
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from module.esmRunner import ESMRunner
from module.marker import Marker
from entity.taxoTree import taxoTree
from tqdm import tqdm
import base64
import numpy

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

class MLKNNModule(Module):
    def __init__(self, reference="VMRv4", marker=False, embedding="CLS", strategy="nearest", model="esm2_t33_256", pooling='sum'):
        self.strategy = strategy
        self.model = model
        self.pooling = pooling
        self.embedding = embedding
        self.reference = reference
        if (pooling not in ["mean", "sum"] and not pooling.startswith("top")):
            raise ValueError("Unknown pooling method")
        if (strategy not in ["nearest", "nearest_bound"]):
            raise ValueError("Unknown strategy")
        super().__init__(f'MLKNN-model={model},embedding={embedding},stratgy={strategy},pooling={pooling},marker={marker}')
        # self.baseName = self.moduleName
        self.resultDict:dict[str, MLResult] = dict()


        modelParams = {
            "esm2_t33_256": (256, f"{config.modelRoot}/realm/esm2_t33_256", "facebook/esm2_t33_650M_UR50D", 29, config.mlBatchSize),
            "esm2_t33_512": (512, f"{config.modelRoot}/realm/esm2_t33_512", "facebook/esm2_t33_650M_UR50D", 29, config.mlBatchSize),
            "esm2_t33_256_enlarge": (256, f"{config.modelRoot}/genus/esm2_t33_256_enlarge_genus", "facebook/esm2_t33_650M_UR50D", 3523, config.mlBatchSize),
            "esm2_t33_512_enlarge": (512, f"{config.modelRoot}/family/esm2_t33_512_enlarge", "facebook/esm2_t33_650M_UR50D", 3523, config.mlBatchSize)
        }

        if (model not in modelParams):
            raise ValueError("Unknown model")
        self.modelParam = modelParams[model]

        # cacheProbFile = f"{config.cacheResultFolder}/ESM_taxo_{model}_prob.tmp"
        self.cacheCLSEmbFile = f"{config.cacheResultFolder}/ESM_taxo_{model}_cls_emb.tmp"
        self.cacheAveEmbFile = f"{config.cacheResultFolder}/ESM_taxo_{model}_ave_emb.tmp"
        # cacheProbIndex = f"{config.cacheResultFolder}/ESM_taxo_{model}_prob.json"
        self.cacheCLSEmbIndex = f"{config.cacheResultFolder}/ESM_taxo_{model}_cls_emb.json"
        self.cacheAveEmbIndex = f"{config.cacheResultFolder}/ESM_taxo_{model}_ave_emb.json"

        useMarkerGene = "marker" if marker else "full"
        self.cacheClusterFile = f"{config.modelRoot}/{self.reference}/{embedding}_{strategy}_{useMarkerGene}.tsv"

        self.cachedSamples_cls = {"nextOffset": 0}
        self.nextOffset_cls = 0
        self.cachedSamples_ave = {"nextOffset": 0}
        self.nextOffset_ave = 0

        self.marker = marker  # only use marker gene or use all protein

    def train(self):
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        samples = IOUtils.loadSamples(referenceFasta)
        self.getEmbedding(samples)

        clusters:dict[str, list[numpy.ndarray]] = {}   # key: taxa node name in ICTV   value: a list of embeddings

        if (self.marker):
            markerModule = Marker(self.reference, "sum")
            if (not os.path.exists(markerModule.markerDB)):
                markerModule.buildDB()
            with open(markerModule.markerDB) as fp:
                markerMapping = json.load(fp)

        if (self.marker):
            if (self.strategy == "CLS"):
                for sample in samples:
                    for protein in sample.proteins:
                        node = taxoTree.ICTVTree.nodes[markerMapping[protein.id]]
                        for n in node.path:
                            if (n.name not in clusters):
                                clusters[n.name] = []
                            clusters[n.name].append(protein.info[f"{self.model}_CLSemb"])
            elif (self.strategy == 'ave'):
                for sample in samples:
                    for protein in sample.proteins:
                        node = taxoTree.ICTVTree.nodes[markerMapping[protein.id]]
                        for n in node.path:
                            if (n.name not in clusters):
                                clusters[n.name] = []
                            clusters[n.name].append(protein.info[f"{self.model}_aveemb"])
                                    
        else:
            for sample in samples:
                ICTVID = taxoTree.ICTVTree.accession2ID[sample.id]
                node = taxoTree.ICTVTree.species[ICTVID]
                names = set()
                for n in node.path:
                    names.add(n.name)
                    if (n.name not in clusters):
                        clusters[n.name] = []
                if (self.strategy == 'CLS'):
                    for protein in sample.proteins:
                        for n in names:
                            clusters[n].append(protein.info[f"{self.model}_CLSemb"])
                elif (self.strategy == 'ave'):
                    for protein in sample.proteins:
                        for n in names:
                            clusters[n].append(protein.info[f"{self.model}_aveemb"])

        with open(self.cacheClusterFile, 'wt') as fp:
            for name, embeddings in tqdm(list(clusters.items()), desc="saving cluster"):
                # calculate centroid and cov
                X = numpy.array(embeddings)
                mu = X.mean(axis=0).astype(numpy.float32)
                # note: be aware that inv_cov is 1280*1280 matrix. So we use diagonal-approximaetd Ma's distance
                # cov = numpy.cov(X, rowvar=False)
                # inv_cov = numpy.linalg.inv(cov + 1e-6 * numpy.eye(cov.shape[0]))
                var = X.var(axis=0).astype(numpy.float32)
                distances = numpy.sqrt(numpy.sum((X - mu) ** 2 / var, axis=1)).astype(numpy.float32)
                distances = numpy.sort(distances)
                fp.write(f"{name}\t{IOUtils.encodeBase64(mu)}\t{IOUtils.encodeBase64(var)}\t{IOUtils.encodeBase64(distances)}\n")

    def applyStrategy(self, distances, confidenceScores):
        if (self.strategy == 'nearest_bound'):
            distances = distances + numpy.where(confidenceScores == 0, numpy.inf, 0)

        return distances

        
    def run(self, samples:list[Sample]):
        if (not os.path.exists(self.cacheClusterFile)):
            self.train()

        self.getEmbedding(samples)

        clusters = []
        with open(self.cacheClusterFile) as fp:
            for line in fp:
                name, mu, var, dis_threshes = line.strip().split('\t')
                mu = IOUtils.decodeBase64(mu)
                var = IOUtils.decodeBase64(var)
                dis_threshes = IOUtils.decodeBase64(dis_threshes)
                clusters.append((name, mu, var, dis_threshes))


        results = list()
        
        
        for sample in tqdm(samples, desc="KNN"):

            if (self.embedding == 'CLS'):
                embeddings = numpy.array([protein.info[f"{self.model}_CLSemb"] for protein in sample.proteins])
            elif (self.embedding == 'ave'):
                embeddings = numpy.array([protein.info[f"{self.model}_aveemb"] for protein in sample.proteins])

            if (self.pooling == "mean"):
                aveEmbedding = numpy.mean(embeddings, axis=0)
                distances = numpy.zeros((len(clusters)), dtype=numpy.float32)
                confidenceScores = numpy.zeros((len(clusters)), dtype=numpy.float16)
                for idx, (name, mu, var, dis_threshes) in enumerate(clusters):
                    dis = numpy.sqrt(numpy.sum((aveEmbedding - mu) ** 2 / var, axis=1))
                    ranks = numpy.searchsorted(dis_threshes, dis, side='left')   # ideally, we should calculate average between left and right. But here, we think the identical number is almost impossible
                    scores = 1 - ranks / len(dis_threshes)

                    distances[idx] = dis
                    confidenceScores[idx] = scores

                distances = self.applyStrategy(distances, confidenceScores)

                selection = numpy.argmin(distances, axis=0).item()

                score = confidenceScores[selection, numpy.arange(confidenceScores.shape[1])].item()
                results.append(PlainResult(clusters[selection][0].name, score))


            else:
                distances = numpy.zeros((len(clusters), len(sample.proteins)), dtype=numpy.float32)
                confidenceScores = numpy.zeros((len(clusters), len(sample.proteins)), dtype=numpy.float16)
                for idx, (name, mu, var, dis_threshes) in enumerate(clusters):
                    dis = numpy.sqrt(numpy.sum((embeddings - mu) ** 2 / var, axis=1))
                    ranks = numpy.searchsorted(dis_threshes, dis, side='left')   # ideally, we should calculate average between left and right. But here, we think the identical number is almost impossible
                    scores = 1 - ranks / len(dis_threshes)

                    distances[idx] = dis
                    confidenceScores[idx] = scores

                distances = self.applyStrategy(distances, confidenceScores)
                
                
                selections, scores = self.applyStrategy(distances, confidenceScores)

                votes = numpy.zeros(len(clusters), dtype=numpy.float16)
                # TODO: haven't decide what is the weight of the vote. Distance? confidence?
                # perhaps vote with top 3 closest with confidence as weight?
                if (self.pooling == 'sum'):
                    pass
                elif (self.pooling.startswith('top')):
                    pass

            


        
        
        return results


    def getEmbedding(self, samples:list[Sample])->None:  # The embedding will be stored in the ProteinSample's info dict
        uncachedSamples = []
        for sample in samples:
            for protein in sample.proteins:
                if (f"{self.model}_CLEemb" not in sample.info or f"{self.model}_Aveemb" not in sample.info):
                    uncachedSamples.append(sample)
        
        if (len(uncachedSamples) == 0):
            return
        NucleotideUtils.extractProtein(uncachedSamples)

        essentialFiles = [self. cacheCLSEmbFile, self.cacheAveEmbFile, self.cacheCLSEmbIndex, self.cacheAveEmbIndex]
        allExists = True
        for f in essentialFiles:
            if (not os.path.exists(f)):
                allExists = False
        
        if (allExists):
            with open(self.cacheCLSEmbIndex) as fp:
                self.cachedSamples_cls = json.load(fp)
                self.nextOffset_cls = self.cachedSamples_cls["nextOffset"]
            with open(self.cacheAveEmbIndex) as fp:
                self.cachedSamples_ave = json.load(fp)
                self.nextOffset_ave = self.cachedSamples_ave["nextOffset"]

        proteinsToRun:list[ProteinSample] = list()
        for protein in uncachedSamples:
            if (protein.id not in self.cachedSamples_cls or protein.id not in self.cachedSamples_ave):
                proteinsToRun.append(protein)

        if (len(proteinsToRun) > 0):
            self.runESM(proteinsToRun)

        cachedResultFP_cls = open(self.cacheCLSEmbFile)
        cachedResultFP_ave = open(self.cacheAveEmbFile)
        for sample in tqdm(samples, desc="pooling"):
            cachedResultFP_cls.seek(self.cachedSamples_cls[sample.id])
            line = cachedResultFP_cls.readline().strip()
            text = line[line.find('\t')+1:]
            sample.info[f"{self.model}_CLSemb"] = IOUtils.decodeBase64(text)

            cachedResultFP_ave.seek(self.cachedSamples_ave[sample.id])
            line = cachedResultFP_ave.readline().strip()
            text = line[line.find('\t')+1:]
            sample.info[f"{self.model}_Aveemb"] = IOUtils.decodeBase64(text)

    
    def runESM(self, samples:list[ProteinSample])->None:
        model = ESMRunner(*self.modelParam)
        lines = model.run(samples)

        fp_cls = open(self.cacheCLSEmbFile, 'at')
        fp_ave = open(self.cacheAveEmbFile, 'at')
        
        for seq_name, (prob, cls, ave) in lines.items():
            self.cachedSamples_cls[seq_name] = nextOffset_cls
            self.cachedSamples_ave[seq_name] = nextOffset_ave

            clsText = f"{seq_name}\t{IOUtils.encodeBase64(cls)}\n"
            aveText = f"{seq_name}\t{IOUtils.encodeBase64(ave)}\n"

            fp_cls.write(clsText)
            fp_ave.write(aveText)
            nextOffset_cls += len(clsText)
            nextOffset_ave += len(aveText)

        self.cachedSamples_cls["nextOffset"] = nextOffset_cls
        self.cachedSamples_ave["nextOffset"] = nextOffset_ave

        fp_cls.close()
        fp_ave.close()

        del model

        for protein in samples:
            if (protein.id not in self.cachedSamples_cls):
                self.cachedSamples_cls[protein.id] = -1
            if (protein.id not in self.cachedSamples_ave):
                self.cachedSamples_ave[protein.id] = -1
        
        with open(self.cacheCLSEmbIndex, 'wt') as fp:
            json.dump(self.cachedSamples_cls, fp, indent=2)
        with open(self.cacheAveEmbIndex, 'wt') as fp:
            json.dump(self.cachedSamples_ave, fp, indent=2)