import os
import json
import time
import math
import numpy
import multiprocessing
from tqdm import tqdm

from config import config
from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module
from entity.proteinSample import ProteinSample
from moduleWorker.proteinLMWorker import runSingle

if multiprocessing.current_process().name == "MainProcess":
    from module.marker import Marker
    from entity.taxoTree import taxoTree
    from utils.NucleotideUtils import NucleotideUtils
    from module.proteinLMRunner import ProteinLMRunner


class ProteinLM(Module):
    def __init__(self, reference, model, embedding="ave", marker=False, strategy="individual", pooling="nosplit", threads=multiprocessing.cpu_count()):
        self.pooling = pooling
        self.embedding = embedding
        self.marker = marker
        self.reference = reference
        self.threads = threads
        if (embedding not in ["ave"]):
            raise ValueError("Unsupported embedding type")
        if (pooling not in ["nosplit", "mean", "sum"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        super().__init__(f'ProteinLM-ref={reference},model={model},embedding={embedding},strategy={strategy},pooling={pooling}')

        models = {
            "DNABert2": "zhihan1996/DNABERT-2-117M",
            "DNABertS": "zhihan1996/DNABERT-S",
            "VitaxHyena": None
        }

        if (model not in models):
            raise ValueError("Unsupported Protein language model")
        self.modelParam = [models[model], config.proteinBatchSize]
        self.model = model

        self.strategy = strategy
        if (strategy not in ["individual", "nearest"]):
            raise ValueError("Unsupported strategy")

        # self.cacheCLSEmbFile = f"{config.cacheResultFolder}/Protein_taxo_{model}_cls_emb.tmp"
        self.cacheAveEmbFile = f"{config.cacheResultFolder}/Protein_taxo_{model}_ave_emb.tmp"
        # self.cacheCLSEmbIndex = f"{config.cacheResultFolder}/Protein_taxo_{model}_cls_emb.json"
        self.cacheAveEmbIndex = f"{config.cacheResultFolder}/Protein_taxo_{model}_ave_emb.json"

        useMarkerGene = "marker" if marker else "full"
        useFullSequence = "wholeSeq" if pooling == "nosplit" else "genes"
        self.cacheClusterFile = f"{config.modelRoot}/{self.reference}/{model}_{embedding}_{useMarkerGene}_{useFullSequence}.tsv"

        # self.cachedSamples_cls = {"nextOffset": 0}
        # self.nextOffset_cls = 0
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
                pass
                # if (self.embedding == "CLS"):
                #     for sample in samples:
                #         for protein in sample.proteins:
                #             node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                #             for n in node.paths:
                #                 if (n.name not in clusters):
                #                     clusters[n.name] = []
                #                 clusters[n.name].append(DNA.info[f"{self.model}_CLSemb"])
                # if (self.embedding == 'ave'):
                #     for sample in samples:
                #         if (self.pooling == "nosplit"):
                #             DNAs:list[Sample|ProteinSample] = [sample]
                #         else:
                #             DNAs = sample.cDNAs
                #         for DNA in DNAs:
                #             node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                #             for n in node.paths:
                #                 if (n.name not in clusters):
                #                     clusters[n.name] = []
                #                 clusters[n.name].append(DNA.info[f"{self.model}_aveemb"])
                                        
            else:
                for sample in samples:
                    node = taxoTree.getTaxoNodeFromAccession(sample.id).ICTVNode
                    names = set()
                    for n in node.paths:
                        names.add(n.name)
                        if (n.name not in clusters):
                            clusters[n.name] = []
                    # if (self.embedding == 'CLS'):
                    #     for protein in sample.proteins:
                    #         for n in names:
                    #             clusters[n].append(protein.info[f"{self.model}_CLSemb"])
                    if (self.embedding == 'ave'):
                        for protein in sample.proteins:
                            for n in names:
                                clusters[n].append(protein.info[f"{self.model}_aveemb"])
            return clusters
        else:
            clusters:list[list[str]] = []
            variance = []
            if (self.marker):
                pass
                # if (self.embedding == "CLS"):
                #     for sample in samples:
                #         if (self.pooling == "nosplit"):
                #             DNAs:list[Sample|ProteinSample] = [sample]
                #         else:
                #             DNAs = sample.cDNAs
                #         for DNA in DNAs:
                #             node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                #             clusters.append([node.name])
                #             variance.append(DNA.info[f"{self.model}_CLSemb"])
                # if (self.embedding == 'ave'):
                #     for sample in samples:
                #         if (self.pooling == "nosplit"):
                #             DNAs:list[Sample|ProteinSample] = [sample]
                #         else:
                #             DNAs = sample.cDNAs
                #         for DNA in DNAs:
                #             node = taxoTree.ICTVTree.nodes[markerMapping[DNA.id]]
                #             clusters.append([node.name])
                #             variance.append(DNA.info[f"{self.model}_aveemb"])
                                        
            else:
                for sample in samples:
                    node = taxoTree.getTaxoNodeFromAccession(sample.id).ICTVNode
                    # if (self.embedding == 'CLS'):
                    #     for protein in sample.proteins:
                    #         clusters.append([node.name])
                    #         variance.append(protein.info[f"{self.model}_CLSemb"])
                    if (self.embedding == 'ave'):
                        for protein in sample.proteins:
                            clusters.append([node.name])
                            variance.append(protein.info[f"{self.model}_aveemb"])
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
        
    def run(self, samples:list[Sample], **kwargs):
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
            res = runSingle(clusters, samples, list(range(len(samples))), self.pooling, self.embedding, self.strategy, self.model, self.invStdVar, self.refEmbeddings, self.ref_sq, self.inverse_indicies, self.uniqueNames)
            results = [r for r, _ in res]
        else:
            samplePerThread = math.ceil(len(samples)/self.threads)
            jobs = []
            pbars = []

            shms = []

            isv, shm = IOUtils.createSharedMemory(self.invStdVar)
            shms.append(shm)
            ref, shm = IOUtils.createSharedMemory(self.refEmbeddings)
            shms.append(shm)
            refsq, shm = IOUtils.createSharedMemory(self.ref_sq)
            shms.append(shm)
            ind, shm = IOUtils.createSharedMemory(self.inverse_indicies)
            shms.append(shm)
            # isv = self.invStdVar
            # ref = self.refEmbeddings
            # refsq = self.ref_sq
            # ind = self.inverse_indicies

            for t in range(self.threads):
                start = t*samplePerThread
                end = min((t+1)*samplePerThread, len(samples))
                if (end <= start):
                    continue
                # pbar = tqdm(total=(end-start), desc=f"Thread {t}")
                # pbars.append(pbar)
                jobs.append([clusters, samples[start : end], list(range(start, end)), self.pooling, self.embedding, self.strategy, self.model, isv, ref, refsq, ind, self.uniqueNames, 0])
            pbars = [tqdm(total=len(samples), desc="Merge")]
            listener, queue = IOUtils.getProgressListener(pbars)
            
            ctx = multiprocessing.get_context("spawn")
            # ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=self.threads) as pool:
                asyncResults = [pool.apply_async(runSingle, [*job, queue]) for job in jobs]
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
            for shm in shms:
                IOUtils.unlinkSharedMemory(shm)

        # clear the cached embedding, to prevent OOM
        for sample in samples:
            for protein in sample.proteins:
                # protein.info.pop(f"{self.model}_CLSemb", None)
                protein.info.pop(f"{self.model}_aveemb", None)
            
        del self.refEmbeddings # this is huge
        return results
            


    def getEmbedding(self, samples:list[Sample])->None:  # The embedding will be stored in the ProteinSample's info dict
        uncachedSamples:list[ProteinSample] = []
        NucleotideUtils.extractProtein(samples)
        for sample in samples:
            for protein in sample.proteins:
                # if (f"{self.model}_CLSemb" not in protein.info or f"{self.model}_aveemb" not in DNA.info):
                if (f"{self.model}_aveemb" not in protein.info):
                    uncachedSamples.append(protein)
        
        if (len(uncachedSamples) == 0):
            return
        

        # essentialFiles = [self.cacheCLSEmbFile, self.cacheAveEmbFile, self.cacheCLSEmbIndex, self.cacheAveEmbIndex]
        essentialFiles = [self.cacheAveEmbFile, self.cacheAveEmbIndex]
        allExists = True
        for f in essentialFiles:
            if (not os.path.exists(f)):
                allExists = False
        
        if (allExists):
            # with open(self.cacheCLSEmbIndex) as fp:
            #     self.cachedSamples_cls = json.load(fp)
            #     self.nextOffset_cls = self.cachedSamples_cls["nextOffset"]
            with open(self.cacheAveEmbIndex) as fp:
                self.cachedSamples_ave = json.load(fp)
                self.nextOffset_ave = self.cachedSamples_ave["nextOffset"]

        proteinsToRun:list[ProteinSample] = list()
        for protein in uncachedSamples:
            # if (protein.id not in self.cachedSamples_cls or protein.id not in self.cachedSamples_ave):
            if (protein.id not in self.cachedSamples_ave):
                proteinsToRun.append(protein)

        if (len(proteinsToRun) > 0):
            q = multiprocessing.SimpleQueue()
            proc = multiprocessing.Process(target=runML, args=(self.modelParam, proteinsToRun, self.cacheAveEmbFile, self.nextOffset_ave, q))
            proc.start()
            cachedSamples_ave = q.get()
            proc.join()

            self.cachedSamples_ave.update(cachedSamples_ave)
            self.nextOffset_ave = cachedSamples_ave["nextOffset"]

            # with open(self.cacheCLSEmbIndex, 'wt') as fp:
            #     json.dump(self.cachedSamples_cls, fp, indent=2)
            with open(self.cacheAveEmbIndex, 'wt') as fp:
                json.dump(self.cachedSamples_ave, fp, indent=2)

        # cachedResultFP_cls = open(self.cacheCLSEmbFile)
        cachedResultFP_ave = open(self.cacheAveEmbFile)
        for sample in tqdm(uncachedSamples, desc="embedding"):
            # cachedResultFP_cls.seek(self.cachedSamples_cls[sample.id])
            # line = cachedResultFP_cls.readline().strip()
            # text = line[line.find('\t')+1:]
            # sample.info[f"{self.model}_CLSemb"] = IOUtils.decodeBase64(text, dtype=numpy.float32)

            cachedResultFP_ave.seek(self.cachedSamples_ave[sample.id])
            line = cachedResultFP_ave.readline().strip()
            text = line[line.find('\t')+1:]
            sample.info[f"{self.model}_aveemb"] = IOUtils.decodeBase64(text, dtype=numpy.float32)
        
        # cachedResultFP_cls.close()
        cachedResultFP_ave.close()

    
def runML(param, samples:list[ProteinSample], cacheAveEmbFile, nextOffset_ave, queue)->None:
    # if (self.pooling == "nosplit"):
    #     threads = min(multiprocessing.cpu_count(), 16)
    # else:
    #     threads = multiprocessing.cpu_count()

    model = ProteinLMRunner(*param)
    lines = model.run(samples)
    model.clean()
    del model

    # fp_cls = open(self.cacheCLSEmbFile, 'at')
    fp_ave = open(cacheAveEmbFile, 'at')

    cachedSamples_ave = {}
    
    for seq_name, ave in lines.items():
        # self.cachedSamples_cls[seq_name] = self.nextOffset_cls
        cachedSamples_ave[seq_name] = nextOffset_ave

        # clsText = f"{seq_name}\t{IOUtils.encodeBase64(cls.astype(numpy.float32))}\n"
        aveText = f"{seq_name}\t{IOUtils.encodeBase64(ave.astype(numpy.float32))}\n"

        # fp_cls.write(clsText)
        fp_ave.write(aveText)
        # self.nextOffset_cls += len(clsText)
        nextOffset_ave += len(aveText)

    # self.cachedSamples_cls["nextOffset"] = self.nextOffset_cls
    cachedSamples_ave["nextOffset"] = nextOffset_ave

    # fp_cls.close()
    fp_ave.close()

    for protein in samples:
        # if (protein.id not in self.cachedSamples_cls):
        #     self.cachedSamples_cls[protein.id] = -1
        if (protein.id not in cachedSamples_ave):
            cachedSamples_ave[protein.id] = -1

    queue.put(cachedSamples_ave)
