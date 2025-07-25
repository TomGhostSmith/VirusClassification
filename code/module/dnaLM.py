# reconstructed
import os
import json
import pandas
from config import config
from prototype.module import Module
from moduleResult.plainResult import PlainResult
from entity.sample import Sample
from entity.proteinSample import ProteinSample
from module.dnaLMRunner import DNALMRunner
from module.esmRunner import ESMRunner
from module.marker import Marker
from tqdm import tqdm
import numpy
import math
import multiprocessing
import time

from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils
from entity.taxoTree import taxoTree

class DNALM(Module):
    def __init__(self, reference, model, embedding="ave", marker=False, strategy="individual", pooling="nosplit", threads=multiprocessing.cpu_count()):
        self.pooling = pooling
        self.embedding = embedding
        self.marker = marker
        self.reference = reference
        if (embedding not in ["CLS", "ave"]):
            raise ValueError("Unsupported embedding type")
        if (pooling not in ["nosplit", "mean", "sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        super().__init__(f'DNALM-ref={reference},model={model},pooling={pooling}')

        models = {
            "DNABert2": "zhihan1996/DNABERT-2-117M",
            "DNABertS": "zhihan1996/DNABERT-S",
        }

        if (model not in models):
            raise ValueError("Unsupported DNA language model")
        batchSize = 1 if (self.pooling == "nosplit") else config.mlBatchSize
        self.modelParam = [models[model], batchSize]
        self.model = model

        self.strategy = strategy
        if (strategy not in ["individual", "nearest"]):
            raise ValueError("Unsupported strategy")

        self.cacheCLSEmbFile = f"{config.cacheResultFolder}/DNA_taxo_{model}_cls_emb.tmp"
        self.cacheAveEmbFile = f"{config.cacheResultFolder}/DNA_taxo_{model}_ave_emb.tmp"
        self.cacheCLSEmbIndex = f"{config.cacheResultFolder}/DNA_taxo_{model}_cls_emb.json"
        self.cacheAveEmbIndex = f"{config.cacheResultFolder}/DNA_taxo_{model}_ave_emb.json"

        useMarkerGene = "marker" if marker else "full"
        useFullSequence = "wholeSeq" if pooling == "nosplit" else "genes"
        self.cacheClusterFile = f"{config.modelRoot}/{self.reference}/{model}_{embedding}_{useMarkerGene}_{useFullSequence}.tsv"

        self.cachedSamples_cls = {"nextOffset": 0}
        self.nextOffset_cls = 0
        self.cachedSamples_ave = {"nextOffset": 0}
        self.nextOffset_ave = 0

        self.invStdVar = None
        self.refEmbeddings = None
        self.ref_sq = None

        self.uniqueNames = None
        self.inverse_indicies = None
        

    def loadRefClusterEmbeddings(self, recursive=True):
        referenceFasta = f"{config.modelRoot}/{self.reference}/{self.reference}.fasta"
        samples = IOUtils.loadSamples(referenceFasta)
        self.getEmbedding(samples)

        if (self.marker):
            markerModule = Marker(self.reference, "sum")
            if (not os.path.exists(markerModule.markerDB)):
                markerModule.buildDB()
            with open(markerModule.markerDB) as fp:
                markerMapping = json.load(fp)

        if (recursive):
            clusters:dict[str, list[numpy.ndarray]] = {}   # key: taxa node name in ICTV   value: a list of embeddings
            if (self.marker):
                if (self.embedding == "CLS"):
                    for sample in samples:
                        if (self.pooling == "nosplit"):
                            DNAs:list[Sample|ProteinSample] = [sample]
                        else:
                            DNAs = sample.cDNAs
                        for DNA in DNAs:
                            node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                            for n in node.paths:
                                if (n.name not in clusters):
                                    clusters[n.name] = []
                                clusters[n.name].append(DNA.info[f"{self.model}_CLSemb"])
                elif (self.embedding == 'ave'):
                    for sample in samples:
                        if (self.pooling == "nosplit"):
                            DNAs:list[Sample|ProteinSample] = [sample]
                        else:
                            DNAs = sample.cDNAs
                        for DNA in DNAs:
                            node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                            for n in node.paths:
                                if (n.name not in clusters):
                                    clusters[n.name] = []
                                clusters[n.name].append(DNA.info[f"{self.model}_aveemb"])
                                        
            else:
                for sample in samples:
                    if (self.pooling == "nosplit"):
                        DNAs:list[Sample|ProteinSample] = [sample]
                    elif (len(sample.cDNAs) == 0):
                        continue
                    else:
                        DNAs = sample.cDNAs
                    ICTVID = taxoTree.ICTVTree.accession2ID[sample.id]
                    node = taxoTree.ICTVTree.species[ICTVID]
                    names = set()
                    for n in node.paths:
                        names.add(n.name)
                        if (n.name not in clusters):
                            clusters[n.name] = []
                    if (self.embedding == 'CLS'):
                        for DNA in DNAs:
                            for n in names:
                                clusters[n].append(DNA.info[f"{self.model}_CLSemb"])
                    elif (self.embedding == 'ave'):
                        for DNA in DNAs:
                            for n in names:
                                clusters[n].append(DNA.info[f"{self.model}_aveemb"])
            return clusters
        else:
            clusters:list[list[str]] = []
            variance = []
            if (self.marker):
                if (self.embedding == "CLS"):
                    for sample in samples:
                        if (self.pooling == "nosplit"):
                            DNAs:list[Sample|ProteinSample] = [sample]
                        else:
                            DNAs = sample.cDNAs
                        for DNA in DNAs:
                            node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                            clusters.append([node.name])
                            variance.append(DNA.info[f"{self.model}_CLSemb"])
                elif (self.embedding == 'ave'):
                    for sample in samples:
                        if (self.pooling == "nosplit"):
                            DNAs:list[Sample|ProteinSample] = [sample]
                        else:
                            DNAs = sample.cDNAs
                        for DNA in DNAs:
                            node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                            clusters.append([node.name])
                            variance.append(DNA.info[f"{self.model}_aveemb"])
                                        
            else:
                for sample in samples:
                    if (self.pooling == "nosplit"):
                        DNAs:list[Sample|ProteinSample] = [sample]
                    elif (len(sample.cDNAs) == 0):
                        continue
                    else:
                        DNAs = sample.cDNAs
                    ICTVID = taxoTree.ICTVTree.accession2ID[sample.id]
                    node = taxoTree.ICTVTree.species[ICTVID]
                    if (self.embedding == 'CLS'):
                        for DNA in DNAs:
                            clusters.append([node.name])
                            variance.append(DNA.info[f"{self.model}_CLSemb"])
                    elif (self.embedding == 'ave'):
                        for DNA in DNAs:
                            clusters.append([node.name])
                            variance.append(DNA.info[f"{self.model}_aveemb"])
            embeddings = numpy.array(variance).astype(numpy.float64)
            var = embeddings.var(axis=0)
            self.invStdVar = (1.0/numpy.sqrt(var + 1e-8)).astype(numpy.float32)
            self.refEmbeddings = embeddings * self.invStdVar
            self.ref_sq = numpy.sum(self.refEmbeddings ** 2, axis=1, keepdims=True)
            return clusters


    def train(self):
        useMarkerGene = "marker" if self.marker else "full"
        IOUtils.showInfo(f"Train {self.reference} {self.model} {self.embedding} {useMarkerGene} KNN")
        clusters = self.loadRefClusterEmbeddings()

        with open(self.cacheClusterFile, 'wt') as fp:
            for name, embeddings in tqdm(list(clusters.items()), desc="saving cluster"):
                # calculate centroid and cov
                if (len(embeddings) <= 1):
                    continue
                X = numpy.array(embeddings).astype(numpy.float64)
                mu = X.mean(axis=0)
                # note: be aware that inv_cov is 1280*1280 matrix. So we use diagonal-approximaetd Ma's distance
                # cov = numpy.cov(X, rowvar=False)
                # inv_cov = numpy.linalg.inv(cov + 1e-6 * numpy.eye(cov.shape[0]))
                var = X.var(axis=0)
                if (numpy.min(var) == 0):  # in case some dimension has no var
                    continue
                distances = numpy.sqrt(numpy.sum((X - mu) ** 2 / var, axis=1)).astype(numpy.float32)  # this calculation should be down within float64 to prevent overflow
                distances = numpy.sort(distances)
                mu = mu.astype(numpy.float32)
                var = var.astype(numpy.float32)
                fp.write(f"{name}\t{IOUtils.encodeBase64(mu)}\t{IOUtils.encodeBase64(var)}\t{IOUtils.encodeBase64(distances)}\n")

    def applyStrategy(self, distances, confidenceScores):
        # here we use softmin weighted distance
        # size of [cluster,  sample]
        if (self.strategy == 'nearest'):
            return softmin(distances)
        if (self.strategy == 'nearest_bound'):
            distances = distances + numpy.where(confidenceScores == 0, numpy.inf, 0)
            return softmin(distances)
        if (self.strategy == 'confidence'):
            return confidenceScores
        if (self.strategy == 'product'):
            w = softmin(distances)
            return w * confidenceScores
        if (self.strategy.startswith('nearest_conf')):
            thresh = float(self.strategy[12:])
            distances = distances + numpy.where(confidenceScores < thresh, numpy.inf, 0)
            return softmin(distances)


    def extractWeightMatrix(self, clusters:list[tuple[str, list[numpy.ndarray]]], embeddings):
        distances = numpy.zeros((len(clusters), embeddings.shape[0]), dtype=numpy.float32)
        if (self.strategy == "individual"):
            emb = embeddings * self.invStdVar
            emb_sq = numpy.sum(emb ** 2, axis=1, keepdims=True).T
            re = self.refEmbeddings @ emb.T
            sq = self.ref_sq - 2 * re + emb_sq
            sq = numpy.maximum(sq, 0.0)
            distances = numpy.sqrt(sq)

            weights = softmin(distances)
            return weights

        else:
            confidenceScores = numpy.zeros((len(clusters), embeddings.shape[0]), dtype=numpy.float16)
            for idx, (name, mu, var, dis_threshes) in enumerate(clusters):
                mu = mu.astype(numpy.float64)
                var = var.astype(numpy.float64)
                dis = numpy.sqrt(numpy.sum((embeddings - mu) ** 2 / var, axis=1)).astype(numpy.float32)
                ranks = numpy.searchsorted(dis_threshes, dis, side='left')   # ideally, we should calculate average between left and right. But here, we think the identical number is almost impossible
                scores = 1 - ranks / len(dis_threshes)

                distances[idx] = dis
                confidenceScores[idx] = scores

            weights = self.applyStrategy(distances, confidenceScores)
            return weights
        
    def extractPrediction(self, scores):
        votes = numpy.bincount(self.inverse_indicies, weights=scores)
        total = sum(votes)
        idx = numpy.argmax(votes)
        return self.uniqueNames[idx], votes[idx]/total
        
    
    def runSingle(self, clusters, samples:list[Sample], indexs, t=0, queue=None):
        res = []
        incre = 0
        if (queue is None):
            bar = tqdm(total=len(samples))

        for index, sample in zip(indexs, samples):
            if (len(sample.proteins) == 0):
                res.append((None, index))
                continue

            if (self.pooling == "nosplit"):
                if (self.embedding == 'CLS'):
                    embeddings = numpy.array([sample.info[f"{self.model}_CLSemb"]])
                elif (self.embedding == 'ave'):
                    embeddings = numpy.array([sample.info[f"{self.model}_aveemb"]])

                weights = self.extractWeightMatrix(clusters, aveEmbedding)
                weights = weights.squeeze(1)

                pred, score = self.extractPrediction(weights)

                res.append((PlainResult(pred, score), index))

            else:
                if (self.embedding == 'CLS'):
                    embeddings = numpy.array([DNA.info[f"{self.model}_CLSemb"] for DNA in sample.cDNAs])
                elif (self.embedding == 'ave'):
                    embeddings = numpy.array([DNA.info[f"{self.model}_aveemb"] for DNA in sample.cDNAs])

                embeddings = embeddings.astype(numpy.float64)
                if (self.pooling == "mean"):
                    aveEmbedding = numpy.mean(embeddings, axis=0, keepdims=True)
                    weights = self.extractWeightMatrix(clusters, aveEmbedding)
                    weights = weights.squeeze(1)

                    pred, score = self.extractPrediction(weights)

                    res.append((PlainResult(pred, score), index))

                else:
                    weights = self.extractWeightMatrix(clusters, embeddings)
                    
                    if (self.pooling == 'sum'):
                        votes = numpy.sum(weights, axis=1)
                    elif (self.pooling.startswith('top')):
                        thresh = int(self.pooling[3:])
                        nonTopWeightsIdx = numpy.argpartition(-weights, thresh, axis=0)[thresh:]
                        weights[nonTopWeightsIdx, numpy.arange(weights.shape[1])] = 0
                        votes = numpy.sum(weights, axis=1)

                    pred, score = self.extractPrediction(votes)

                    res.append((PlainResult(pred, score), index))
            
            if (queue is None):
                bar.update(1)
            else:
                try:
                    queue.put((t, 1 + incre), block=False)
                    incre = 0
                except queue.Full:
                    incre += 1
        if (queue is None):
            bar.close()
        return res


        
    def run(self, samples:list[Sample]):
        if (not os.path.exists(self.cacheClusterFile) and self.strategy != "individual"):
            self.train()

        self.getEmbedding(samples)

        results = [None]*len(samples)
                
        if (self.strategy == "individual"):
            clusters = self.loadRefClusterEmbeddings(False)
        else:
            clusters = []
            with open(self.cacheClusterFile) as fp:
                for line in fp:
                    name, mu, var, dis_threshes = line.strip().split('\t')
                    mu = IOUtils.decodeBase64(mu, dtype=numpy.float32)
                    var = IOUtils.decodeBase64(var, dtype=numpy.float32)
                    dis_threshes = IOUtils.decodeBase64(dis_threshes, dtype=numpy.float32)
                    clusters.append((name, mu, var, dis_threshes))


        names = numpy.array([cluster[0] for cluster in clusters])
        self.uniqueNames, self.inverse_indicies = numpy.unique(names, return_inverse=True)

        if (self.threads == 1):
            res = self.runSingle(clusters, samples, list(range(len(samples))))
            r, i = zip(*res)
            results = r
        else:
            samplePerThread = math.ceil(len(samples)/self.threads)
            jobs = []
            pbars = []
            for t in range(self.threads):
                start = t*samplePerThread
                end = min((t+1)*samplePerThread, len(samples))
                pbar = tqdm(total=(end-start), desc=f"Thread {t}")
                pbars.append(pbar)
                jobs.append([clusters, samples[start : end], list(range(start, end)), t])

            listener, queue = IOUtils.getProgressListener(pbars)
            
                    
            with multiprocessing.Pool(processes=self.threads) as pool:
                asyncResults = [pool.apply_async(self.runSingle, [*job, queue]) for job in jobs]
                pool.close()
                while asyncResults:
                    for asyncResult in asyncResults[:]:
                        if asyncResult.ready():
                            res = asyncResult.get()
                            for r, idx in res:
                                results[idx] = r
                            asyncResults.remove(asyncResult)
                    time.sleep(1)
                pool.join()
            
            IOUtils.stopProgressListener(listener, queue)

        # clear the cached embedding, to prevent OOM
        for sample in samples:
            sample.info.pop(f"{self.model}_CLSemb", None)
            sample.info.pop(f"{self.model}_aveemb", None)
            for DNA in sample.cDNAs:
                DNA.info.pop(f"{self.model}_CLSemb", None)
                DNA.info.pop(f"{self.model}_aveemb", None)
        
        del self.refEmbeddings # this is huge
        return results
            


    def getEmbedding(self, samples:list[Sample])->None:  # The embedding will be stored in the ProteinSample's info dict
        uncachedSamples:list[Sample|ProteinSample] = []
        if (self.pooling == "nosplit"):
            for sample in samples:
                if (f"{self.model}_CLSemb" not in sample.info or f"{self.model}_aveemb" not in sample.info):
                    uncachedSamples.append(sample)
        else:
            NucleotideUtils.extractProtein(samples)
            for sample in samples:
                for DNA in sample.cDNAs:
                    if (f"{self.model}_CLSemb" not in DNA.info or f"{self.model}_aveemb" not in DNA.info):
                        uncachedSamples.append(DNA)
        
        if (len(uncachedSamples) == 0):
            return
        

        essentialFiles = [self.cacheCLSEmbFile, self.cacheAveEmbFile, self.cacheCLSEmbIndex, self.cacheAveEmbIndex]
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

        DNAsToRun:list[Sample|ProteinSample] = list()
        for DNA in uncachedSamples:
            if (DNA.id not in self.cachedSamples_cls or DNA.id not in self.cachedSamples_ave):
                DNAsToRun.append(DNA)

        if (len(DNAsToRun) > 0):
            self.runML(DNAsToRun)

        cachedResultFP_cls = open(self.cacheCLSEmbFile)
        cachedResultFP_ave = open(self.cacheAveEmbFile)
        for sample in tqdm(uncachedSamples, desc="embedding"):
            cachedResultFP_cls.seek(self.cachedSamples_cls[sample.id])
            line = cachedResultFP_cls.readline().strip()
            text = line[line.find('\t')+1:]
            sample.info[f"{self.model}_CLSemb"] = IOUtils.decodeBase64(text, dtype=numpy.float32)

            cachedResultFP_ave.seek(self.cachedSamples_ave[sample.id])
            line = cachedResultFP_ave.readline().strip()
            text = line[line.find('\t')+1:]
            sample.info[f"{self.model}_aveemb"] = IOUtils.decodeBase64(text, dtype=numpy.float32)

    
    def runML(self, samples:list[Sample|ProteinSample])->None:
        if (self.pooling == "nosplit"):
            threads = min(multiprocessing.cpu_count(), 16)
        else:
            threads = multiprocessing.cpu_count()
        model = DNALMRunner(*self.modelParam, threads=threads)
        lines = model.run(samples)

        fp_cls = open(self.cacheCLSEmbFile, 'at')
        fp_ave = open(self.cacheAveEmbFile, 'at')
        
        for seq_name, (cls, ave) in lines.items():
            self.cachedSamples_cls[seq_name] = self.nextOffset_cls
            self.cachedSamples_ave[seq_name] = self.nextOffset_ave

            clsText = f"{seq_name}\t{IOUtils.encodeBase64(cls.astype(numpy.float32))}\n"
            aveText = f"{seq_name}\t{IOUtils.encodeBase64(ave.astype(numpy.float32))}\n"

            fp_cls.write(clsText)
            fp_ave.write(aveText)
            self.nextOffset_cls += len(clsText)
            self.nextOffset_ave += len(aveText)

        self.cachedSamples_cls["nextOffset"] = self.nextOffset_cls
        self.cachedSamples_ave["nextOffset"] = self.nextOffset_ave

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


def softmin(distances):
    w = numpy.exp(-distances)
    s = numpy.sum(w, axis=0)
    msk = s != 0
    w[:, msk] /= s[msk]
    return w