#This file is modified from https://github.com/ChengPENG-wolf/ViraLM/blob/main/viralm.py

from transformers import AutoTokenizer, AutoModel
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
import math
import time
import csv
import os
import gc
import base64

from entity.sample import Sample
from entity.proteinSample import ProteinSample
from config import config
from Bio import SeqIO
from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils


class DNALMRunner():
    def __init__(self, modelName, batchSize=15, threads=multiprocessing.cpu_count(), max_seg_length=4096):
        self.modelName = modelName
        self.batchSize = batchSize
        # self.manualLoadConfig = manualLoadConfig
        self.threads = threads
        self.max_seg_length = max_seg_length
        self.batchSizes = []
        self.max_seg_lengths = []
        i = 256
        # bsMultiply = [4, 3, 3, 3, 3, ...]
        bs = batchSize
        # initial = True
        while i <= max_seg_length:
            self.max_seg_lengths.append(i)
            self.batchSizes.append(bs)
            # if (initial):
            #     bs = bs * 4
            #     initial = False
            # else:
            bs = bs * 3
            i = i * 2
        self.batchSizes = list(reversed(self.batchSizes))

    def loadModel(self):
        manualLoadConfig = None
        if (manualLoadConfig):
            self.model = AutoModel.from_pretrained(self.modelName,
                                                   config = self.manualLoadConfig,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float32)

        else:
            self.model = AutoModel.from_pretrained(self.modelName,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float32)

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

    

    def run(self, samples:list[Sample|ProteinSample]):  # can take both DNAs and cDNAs
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # we have implemented parallel, and no need to use parallel in tokenizer
        labels = []
        sequences = []
        embeddings = {}
        totalCounts = {}
        for sample in samples:
            labels.append(sample.id)
            sequences.append(str(sample.seq.seq).upper())
            embeddings[sample.id] = 0
            totalCounts[sample.id] = 0
        # else:
        #     labels.append(sample[0])
        #     sequences.append(str(sample[1]).upper())
        #     embeddings[sample[0]] = 0
        #     totalCounts[sample[0]] = 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modelName,
            trust_remote_code=True
        )

        # input_ids = []
        # real_masks = []
        # new_labels = []
        data = {l: {
            "labels": [],
            "input_ids": [],
            "real_mask": [],
            "attention_mask": []
        } for l in self.max_seg_lengths}

        threads = self.threads if (len(samples) > self.threads) else len(samples)

        if (threads == 1):
            input_ids, real_masks, new_labels = tokenize_function(self.tokenizer, self.max_seg_lengths, sequences, labels)
            for l, inputList, realMaskList, newLabelList in zip(self.max_seg_lengths, input_ids, real_masks, new_labels):
                data[l]["labels"] = newLabelList
                data[l]["input_ids"] = inputList
                data[l]["real_mask"] = realMaskList
                data[l]["attention_mask"] = realMaskList > 0
        else:
            bs = math.ceil(len(samples)/threads)
            threads = math.ceil(len(samples) / bs)  # this can help adjust the number of threads if too few samples
            pbar = tqdm(total=threads, desc="tokenize")
            jobs = [(self.tokenizer, self.max_seg_lengths, sequences[i * bs : (i+1) * bs], labels[i * bs : (i+1) * bs]) for i in range(threads)]
            with multiprocessing.Pool(processes=threads) as pool:
                asyncResults = [pool.apply_async(tokenize_function, param) for param in jobs]
                # for i in range(threads):
                #     asyncResults.append(pool.apply_async(self.tokenize_function, ))
                
                pool.close()
                
                while asyncResults:
                    for asyncResult in asyncResults[:]:
                        if (asyncResult.ready()):
                            segments, masks, newLabels = asyncResult.get()
                            for l, inputList, realMaskList, newLabelList in zip(self.max_seg_lengths, segments, masks, newLabels):
                                if (len(newLabelList) != 0):
                                    data[l]["labels"] += newLabelList
                                    data[l]["input_ids"].append(inputList)
                                    data[l]["real_mask"].append(realMaskList)
                                    data[l]["attention_mask"].append(realMaskList > 0)
                            # input_ids.append(segments)
                            # real_masks.append(masks)
                            # new_labels += newLabels
                            pbar.update(1)

                            asyncResults.remove(asyncResult)
                    time.sleep(1)
                pool.join()
            for l in self.max_seg_lengths:
                if (len(data[l]["labels"]) != 0):
                    data[l]["input_ids"] = torch.cat(data[l]["input_ids"], dim=0)
                    data[l]["real_mask"] = torch.cat(data[l]["real_mask"], dim=0)
                    data[l]["attention_mask"] = torch.cat(data[l]["attention_mask"], dim=0)
            pbar.close()

        dataLoaders = []
        for l, bs in zip(self.max_seg_lengths, self.batchSizes):
            if (len(data[l]["labels"]) != 0):
                testset = Dataset.from_dict(data[l]).with_format("torch")
                test_loader = DataLoader(testset, batch_size=bs)
                dataLoaders.append(test_loader)

        self.loadModel()

        with torch.no_grad():
            for l, test_loader in zip(self.max_seg_lengths, dataLoaders):
                for batch in tqdm(test_loader, total=len(test_loader), desc=f"{l} token inference"):
                    labels = batch['labels']
                    real_mask = batch["real_mask"].to(self.device)
                    batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels" and k != "real_mask"}

                    outputs = self.model(**batch, output_hidden_states=True)
                    last_hidden_state = outputs[0]
                    masks = real_mask.unsqueeze(-1)
                    masked_hidden = last_hidden_state * masks
                    sum_hidden = masked_hidden.sum(dim=1).detach().cpu().contiguous().numpy()
                    lengths = masks.sum(dim=1).detach().cpu().contiguous().numpy()

                    for idx, label in enumerate(labels):
                        embeddings[label] += sum_hidden[idx]
                        totalCounts[label] += lengths[idx]
                if (torch.cuda.is_available()):
                    torch.cuda.empty_cache()
                    

        for k, v in embeddings.items():
            embeddings[k] = v / totalCounts[k]
        return embeddings
    
    def clean(self):
        del self.model
        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
def tokenize_function(tokenizer, max_seg_lengths, sequences, labels):
    idLists = tokenizer(sequences)["input_ids"]
    newLabels = [[] for _ in max_seg_lengths]
    segments = [[] for _ in max_seg_lengths]
    masks = [[] for _ in max_seg_lengths]
    maxLength = max_seg_lengths[-1]
    for tokens, label in zip(idLists, labels):
        tokens = torch.tensor(tokens)

        if (len(tokens) > maxLength):
            stride = maxLength // 2
            segLength = maxLength
            for start in range(0, len(tokens) - stride, stride):
                end = start + segLength
                segment = tokens[start:end]
                segLen = segment.shape[0]
                if (segLen < segLength):
                    padLen = segLength - segLen
                    segment = torch.cat([segment, torch.full((padLen, ), tokenizer.pad_token_id, dtype=segment.dtype)])
                mask = torch.full((segLength, ), 0.5, dtype=torch.float16)
                if start == 0:
                    mask[:stride] = 1
                elif end == len(tokens):
                    mask[stride:] = 1
                elif end > len(tokens):
                    mask[stride : segLen] = 1
                    mask[segLen:] = 0
                segments[-1].append(segment)
                masks[-1].append(mask)
                newLabels[-1].append(label)
        else:
            for bin_id, segLength in enumerate(max_seg_lengths):
                if (len(tokens) <= segLength):
                    segment = tokens
                    segLen = segment.shape[0]
                    padLen = segLength - segLen
                    segment = torch.cat([segment, torch.full((padLen, ), tokenizer.pad_token_id, dtype=segment.dtype)])
                    mask = torch.ones(segLength, dtype=torch.float16)
                    mask[segLen:] = 0
                    segments[bin_id].append(segment)
                    masks[bin_id].append(mask)
                    newLabels[bin_id].append(label)
                    break
    
    for bin_id in range(len(max_seg_lengths)):
        if (len(newLabels[bin_id]) != 0):
            segments[bin_id] = torch.stack(segments[bin_id])
            masks[bin_id] = torch.stack(masks[bin_id])            
    return segments, masks, newLabels
