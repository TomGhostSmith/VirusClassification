#This file is modified from https://github.com/ChengPENG-wolf/ViraLM/blob/main/viralm.py
import os
import json
import torch
import multiprocessing
import transformers
from tqdm import tqdm
from datasets import Dataset
from typing import Dict, Sequence

from config import config
from utils import IOUtils
from entity.proteinSample import ProteinSample

if multiprocessing.current_process().name == "MainProcess":
    from torch import nn
    from torch.nn import Softmax
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from concurrent.futures import ProcessPoolExecutor
    from utils.parallelUtils import WorkerPool

class ESMRunner():
    def __init__(self, modelName, maxLen, modelFolder, baseModelFolder, n_class, batchSize=None):
        self.modelName = modelName

        self.maxLen = maxLen
        self.modelFolder = modelFolder
        self.baseModelFolder = baseModelFolder
        self.n_class = n_class
        if (batchSize is None):
            batchSize = config.mlBatchSize
        self.batchSize = batchSize


        self.cacheProbFile = f"{config.cacheResultFolder}/ESM_taxo_{self.modelName}_prob.tmp"
        self.cacheCLSEmbFile = f"{config.cacheResultFolder}/ESM_taxo_{self.modelName}_cls_emb.tmp"
        self.cacheAveEmbFile = f"{config.cacheResultFolder}/ESM_taxo_{self.modelName}_ave_emb.tmp"
        self.cacheProbIndex = f"{config.cacheResultFolder}/ESM_taxo_{self.modelName}_prob.json"
        self.cacheCLSEmbIndex = f"{config.cacheResultFolder}/ESM_taxo_{self.modelName}_cls_emb.json"
        self.cacheAveEmbIndex = f"{config.cacheResultFolder}/ESM_taxo_{self.modelName}_ave_emb.json"

        self.cachedSamples_prob = {"nextOffset": 0}
        self.nextOffset_prob = 0
        self.cachedSamples_cls = {"nextOffset": 0}
        self.nextOffset_cls = 0
        self.cachedSamples_ave = {"nextOffset": 0}
        self.nextOffset_ave = 0


    def loadModel(self):
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.baseModelFolder,
                                                                                num_labels=self.n_class,
                                                                                trust_remote_code=True,
                                                                                torch_dtype=torch.float16,
                                                                                )

        self.model.load_state_dict(torch.load(f"{self.modelFolder}/pytorch_model.bin", map_location=torch.device('cpu')), strict=False)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if torch.cuda.device_count() > 1:
            # print(f'\nRunning on {torch.cuda.device_count()} GPUs.')
            self.model = nn.DataParallel(self.model)
        else:
            # print(f'\nRunning on {self.device}.')
            pass
        
        self.model.to(self.device)
        self.model.eval()

    def collateData(self, batch:Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in batch] for key in ("input_ids", "accession"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = labels
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
    def esm(self, proteins:list[ProteinSample]):
        # force read all of the caches to append
        if (os.path.exists(self.cacheProbIndex)):
            with open(self.cacheProbIndex) as fp:
                self.cachedSamples_prob = json.load(fp)
                self.nextOffset_prob = self.cachedSamples_prob["nextOffset"]
        if (os.path.exists(self.cacheCLSEmbIndex)):
            with open(self.cacheCLSEmbIndex) as fp:
                self.cachedSamples_cls = json.load(fp)
                self.nextOffset_cls = self.cachedSamples_cls["nextOffset"]
        if (os.path.exists(self.cacheAveEmbIndex)):
            with open(self.cacheAveEmbIndex) as fp:
                self.cachedSamples_ave = json.load(fp)
                self.nextOffset_ave = self.cachedSamples_ave["nextOffset"]

        IOUtils.showInfo(f"run {len(proteins)} proteins on ESM {self.modelName}")

        if (not torch.cuda.is_available()):
            devices = ["cpu"]
        elif (config.modelParallel):
            devices = ["auto"]
        else:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            
        cmds = [["python", "code/tools/esm.py", self.modelFolder, self.baseModelFolder, str(self.maxLen), str(self.batchSize), str(self.n_class), device] for device in devices]

        pool = WorkerPool(cmds, desc="ESM")
        lines = pool.run(proteins)

        probLines = []
        clsLines = []
        aveLines = []

        for line in lines:
            seq_name, probText, clsText, aveText = line.split("\t")
            probText = f"{seq_name}\t{probText}\n"
            clsText = f"{seq_name}\t{clsText}\n"
            aveText = f"{seq_name}\t{aveText}\n"

            self.cachedSamples_prob[seq_name] = self.nextOffset_prob
            self.cachedSamples_cls[seq_name] = self.nextOffset_cls
            self.cachedSamples_ave[seq_name] = self.nextOffset_ave


            probLines.append(probText)
            clsLines.append(clsText)
            aveLines.append(aveText)

            self.nextOffset_prob += len(probText)
            self.nextOffset_cls += len(clsText)
            self.nextOffset_ave += len(aveText)

        self.cachedSamples_prob["nextOffset"] = self.nextOffset_prob
        self.cachedSamples_cls["nextOffset"] = self.nextOffset_cls
        self.cachedSamples_ave["nextOffset"] = self.nextOffset_ave


        fp_prob = open(self.cacheProbFile, 'at')
        fp_cls  = open(self.cacheCLSEmbFile, 'at')
        fp_ave  = open(self.cacheAveEmbFile, 'at')

        fp_prob.writelines(probLines)
        fp_cls.writelines(clsLines)
        fp_ave.writelines(aveLines)

        fp_prob.close()
        fp_cls.close()
        fp_ave.close()
        
        for protein in proteins:
            if (protein.id not in self.cachedSamples_prob):
                self.cachedSamples_prob[protein.id] = -1
            if (protein.id not in self.cachedSamples_cls):
                self.cachedSamples_cls[protein.id] = -1
            if (protein.id not in self.cachedSamples_ave):
                self.cachedSamples_ave[protein.id] = -1

        with open(self.cacheProbIndex, 'wt') as fp:
            json.dump(self.cachedSamples_prob, fp, indent=2)
        with open(self.cacheCLSEmbIndex, 'wt') as fp:
            json.dump(self.cachedSamples_cls, fp, indent=2)
        with open(self.cacheAveEmbIndex, 'wt') as fp:
            json.dump(self.cachedSamples_ave, fp, indent=2)

        
    def run(self, proteins:list[ProteinSample], getProb=False, getCls=False, getAve=False, **kwargs):
        essentialFiles = []
        if getProb:
            essentialFiles += [self.cacheProbFile, self.cacheProbIndex]
        if getCls:
            essentialFiles += [self.cacheCLSEmbFile, self.cacheCLSEmbIndex]
        if getAve:
            essentialFiles += [self.cacheAveEmbFile, self.cacheAveEmbIndex]
        allExists = True
        for f in essentialFiles:
            if (not os.path.exists(f)):
                allExists = False
        
        if (allExists):
            if (getProb):
                with open(self.cacheProbIndex) as fp:
                    self.cachedSamples_prob = json.load(fp)
            if (getCls):
                with open(self.cacheCLSEmbIndex) as fp:
                    self.cachedSamples_cls = json.load(fp)
            if (getAve):
                with open(self.cacheAveEmbIndex) as fp:
                    self.cachedSamples_ave = json.load(fp)

        proteinsToRun:list[ProteinSample] = list()
        for protein in proteins:
            if ((getProb and protein.id not in self.cachedSamples_prob)
                or (getCls and protein.id not in self.cachedSamples_cls)
                or (getAve and protein.id not in self.cachedSamples_ave)):
            # if (protein.id not in self.cachedSamples_prob):
                proteinsToRun.append(protein)

        if (len(proteinsToRun) > 0):
            self.esm(proteinsToRun)
        
        if (getProb):
            cachedResultFP_prob = open(self.cacheProbFile)
            for protein in proteins:
                offset = self.cachedSamples_prob[protein.id]
                if (offset == -1):
                    continue
                cachedResultFP_prob.seek(offset)
                line = cachedResultFP_prob.readline().strip()
                text = line[line.find('\t')+1:]
                try:
                    protein.info[f"{self.modelName}_prob"] = IOUtils.decodeBase64(text)
                except Exception as e:
                    print(f"offset={offset}")
                    print(f"'{line}'")
                    print(f"'{text}'")
                    print(e)
                    raise Exception
            cachedResultFP_prob.close()
        if (getCls):
            cachedResultFP_cls = open(self.cacheCLSEmbFile)
            for protein in proteins:
                offset = self.cachedSamples_cls[protein.id]
                if (offset == -1):
                    protein.info[f"{self.modelName}_CLSemb"] = None
                else:
                    cachedResultFP_cls.seek(offset)
                    line = cachedResultFP_cls.readline().strip()
                    text = line[line.find('\t')+1:]
                    protein.info[f"{self.modelName}_CLSemb"] = IOUtils.decodeBase64(text)
            cachedResultFP_cls.close()
        if (getAve):
            cachedResultFP_ave = open(self.cacheAveEmbFile)
            for protein in proteins:
                offset = self.cachedSamples_ave[protein.id]
                if (offset == -1):
                    protein.info[f"{self.modelName}_aveemb"] = None
                else:
                    cachedResultFP_ave.seek(offset)
                    line = cachedResultFP_ave.readline().strip()
                    text = line[line.find('\t')+1:]
                    protein.info[f"{self.modelName}_aveemb"] = IOUtils.decodeBase64(text)
            cachedResultFP_ave.close()


def tokenize(testset, tokenizer, batchSize):
    def tokenize_function(examples):  # do not padding in the tokenizer but in batch: prevent too many padding if one sequence is too long
        return tokenizer(examples["sequence"], truncation=True)

    tokenized_datasets = testset.map(tokenize_function, 
                                    batched=True, 
                                    batch_size=batchSize,
                                    remove_columns=["sequence"], 
                                    num_proc=multiprocessing.cpu_count()).with_format("torch")
    return tokenized_datasets