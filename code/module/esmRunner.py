#This file is modified from https://github.com/ChengPENG-wolf/ViraLM/blob/main/viralm.py

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.nn import Softmax
from Bio import SeqIO
from torch import nn
import transformers
import subprocess
import multiprocessing
from datasets import Dataset
from tqdm import tqdm
import torch
import json
import math
import csv
import os
import base64

from entity.proteinSample import ProteinSample
from config import config
from Bio import SeqIO
from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils

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
    
    def esm(self, proteins):
        def tokenize_function(examples):  # do not padding in the tokenizer but in batch: prevent too many padding if one sequence is too long
            return self.tokenizer(examples["sequence"], truncation=True)

        labels = []
        sequences = []
        for protein in proteins:
            labels.append(protein.id)
            sequences.append(str(protein.seq.seq).upper())
        
        testset = Dataset.from_dict({
            "accession": labels,
            "sequence": sequences
        })
        

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modelFolder,
            model_max_length=self.maxLen,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )


        tokenized_datasets = testset.map(tokenize_function, 
                                         batched=True, 
                                         batch_size=self.batchSize,
                                         remove_columns=["sequence"], 
                                         num_proc=multiprocessing.cpu_count()).with_format("torch")
        test_loader = DataLoader(tokenized_datasets, batch_size=self.batchSize, collate_fn=self.collateData)

        softmax = Softmax(dim=0)

        self.loadModel()

        fp_prob = open(self.cacheProbFile, 'at')
        fp_cls = open(self.cacheCLSEmbFile, 'at')
        fp_ave = open(self.cacheAveEmbFile, 'at')

        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                labels = batch['labels']
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}

                outputs = self.model(**batch, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                cls_embedding = last_hidden_state[:, 0, :]
                # ave_embedding = torch.mean(last_hidden_state, dim=1)  # note: this won't work because there are padding
                masks = batch['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden_state * masks
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = masks.sum(dim=1)
                ave_embedding = sum_hidden / lengths
                cls_embeddings = cls_embedding.detach().cpu().contiguous().numpy()
                ave_embeddings = ave_embedding.detach().cpu().contiguous().numpy()

                logits = outputs.logits.cpu().numpy()

                for i in torch.arange(len(labels)):
                    probabilities = softmax(torch.tensor(logits[i])).numpy()
                    # embedding_str = base64.b64encode(embedding.tobytes()).decode('ascii')
                    # embedding = numpy.frombuffer(base64.b64decode(embedding_str), dtype=numpy.float16)  # note: we are using float16
                    seq_name = labels[i]

                    self.cachedSamples_prob[seq_name] = self.nextOffset_prob
                    self.cachedSamples_cls[seq_name] = self.nextOffset_cls
                    self.cachedSamples_ave[seq_name] = self.nextOffset_ave

                    probText = f"{seq_name}\t{IOUtils.encodeBase64(probabilities)}\n"
                    clsText = f"{seq_name}\t{IOUtils.encodeBase64(cls_embeddings[i])}\n"
                    aveText = f"{seq_name}\t{IOUtils.encodeBase64(ave_embeddings[i])}\n"

                    fp_prob.write(probText)
                    fp_cls.write(clsText)
                    fp_ave.write(aveText)
                    self.nextOffset_prob += len(probText)
                    self.nextOffset_cls += len(clsText)
                    self.nextOffset_ave += len(aveText)

        self.cachedSamples_prob["nextOffset"] = self.nextOffset_prob
        self.cachedSamples_cls["nextOffset"] = self.nextOffset_cls
        self.cachedSamples_ave["nextOffset"] = self.nextOffset_ave

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

        
    def run(self, proteins:list[ProteinSample], getProb=False, getCls=False, getAve=False):
        essentialFiles = [self.cacheProbFile, self.cacheCLSEmbFile, self.cacheAveEmbFile, self.cacheProbIndex, self.cacheCLSEmbIndex, self.cacheAveEmbIndex]
        allExists = True
        for f in essentialFiles:
            if (not os.path.exists(f)):
                allExists = False
        
        if (allExists):
            with open(self.cacheProbIndex) as fp:
                self.cachedSamples_prob = json.load(fp)
                self.nextOffset_prob = self.cachedSamples_prob["nextOffset"]
            with open(self.cacheCLSEmbIndex) as fp:
                self.cachedSamples_cls = json.load(fp)
                self.nextOffset_cls = self.cachedSamples_cls["nextOffset"]
            with open(self.cacheAveEmbIndex) as fp:
                self.cachedSamples_ave = json.load(fp)
                self.nextOffset_ave = self.cachedSamples_ave["nextOffset"]

        proteinsToRun:list[ProteinSample] = list()
        for protein in proteins:
            if (protein.id not in self.cachedSamples_prob or protein.id not in self.cachedSamples_cls or protein.id not in self.cachedSamples_ave):
                proteinsToRun.append(protein)

        if (len(proteinsToRun) > 0):
            IOUtils.showInfo(f"run {len(proteinsToRun)} proteins on ESM {self.modelName}")
            self.esm(proteinsToRun)

            with open(self.cacheProbIndex, 'wt') as fp:
                json.dump(self.cachedSamples_prob, fp, indent=2)
            with open(self.cacheCLSEmbIndex, 'wt') as fp:
                json.dump(self.cachedSamples_cls, fp, indent=2)
            with open(self.cacheAveEmbIndex, 'wt') as fp:
                json.dump(self.cachedSamples_ave, fp, indent=2)
        
        if (getProb):
            cachedResultFP_prob = open(self.cacheProbFile)
            for protein in proteins:
                offset = self.cachedSamples_prob[protein.id]
                if (offset == -1):
                    continue
                cachedResultFP_prob.seek(offset)
                line = cachedResultFP_prob.readline().strip()
                text = line[line.find('\t')+1:]
                protein.info[f"{self.modelName}_prob"] = IOUtils.decodeBase64(text)
            cachedResultFP_prob.close()
        if (getCls):
            cachedResultFP_cls = open(self.cacheCLSEmbFile)
            for protein in proteins:
                offset = self.cachedSamples_cls[protein.id]
                if (offset == -1):
                    protein.info[f"{self.model}_CLSemb"] = None
                else:
                    cachedResultFP_cls.seek(offset)
                    line = cachedResultFP_cls.readline().strip()
                    text = line[line.find('\t')+1:]
                    protein.info[f"{self.model}_CLSemb"] = IOUtils.decodeBase64(text)
            cachedResultFP_cls.close()
        if (getAve):
            cachedResultFP_ave = open(self.cacheAveEmbFile)
            for protein in proteins:
                offset = self.cachedSamples_ave[protein.id]
                if (offset == -1):
                    protein.info[f"{self.model}_aveemb"] = None
                else:
                    cachedResultFP_ave.seek(offset)
                    line = cachedResultFP_ave.readline().strip()
                    text = line[line.find('\t')+1:]
                    protein.info[f"{self.model}_aveemb"] = IOUtils.decodeBase64(text)
            cachedResultFP_ave.close()
