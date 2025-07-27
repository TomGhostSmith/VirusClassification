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

        input_ids = []
        real_masks = []
        new_labels = []

        threads = self.threads if (len(samples) > self.threads) else len(samples)

        if (threads == 1):
            input_ids, real_masks, new_labels = tokenize_function(self.tokenizer, self.max_seg_length, sequences, labels)
        else:
            bs = math.ceil(len(samples)/threads)
            threads = math.ceil(len(samples) / bs)  # this can help adjust the number of threads if too few samples
            pbar = tqdm(total=threads, desc="tokenize")
            jobs = [(self.tokenizer, self.max_seg_length, sequences[i * bs : (i+1) * bs], labels[i * bs : (i+1) * bs]) for i in range(threads)]
            with multiprocessing.Pool(processes=threads) as pool:
                asyncResults = [pool.apply_async(tokenize_function, param) for param in jobs]
                # for i in range(threads):
                #     asyncResults.append(pool.apply_async(self.tokenize_function, ))
                
                pool.close()
                
                while asyncResults:
                    for asyncResult in asyncResults[:]:
                        if (asyncResult.ready()):
                            segments, masks, newLabels = asyncResult.get()
                            input_ids.append(segments)
                            real_masks.append(masks)
                            new_labels += newLabels
                            pbar.update(1)

                            asyncResults.remove(asyncResult)
                    time.sleep(1)
                pool.join()
            input_ids = torch.cat(input_ids, dim=0)
            real_masks = torch.cat(real_masks, dim=0)
            pbar.close()

        attn_masks = real_masks > 0


        
        testset = Dataset.from_dict({
            "labels": new_labels,
            "input_ids": input_ids,
            "real_mask": real_masks,
            "attention_mask": attn_masks
        }).with_format("torch")

        test_loader = DataLoader(testset, batch_size=self.batchSize)

        self.loadModel()

        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc="Inference"):
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

        for k, v in embeddings.items():
            embeddings[k] = v / totalCounts[k]
        return embeddings
    
    def clean(self):
        del self.model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
def tokenize_function(tokenizer, max_seg_length, sequences, labels):
    idLists = tokenizer(sequences)["input_ids"]
    newLabels = []
    segments = []
    masks = []
    for tokens, label in zip(idLists, labels):
        tokens = torch.tensor(tokens)
        stride = max_seg_length // 2
        segLength = max_seg_length

        if (len(tokens) <= segLength):
            segment = tokens
            segLen = segment.shape[0]
            padLen = segLength - segLen
            segment = torch.cat([segment, torch.full((padLen, ), tokenizer.pad_token_id, dtype=segment.dtype)])
            mask = torch.ones(segLength, dtype=torch.float16)
            mask[segLen:] = 0
            segments.append(segment)
            masks.append(mask)
            newLabels.append(label)
        else:
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
                segments.append(segment)
                masks.append(mask)
                newLabels.append(label)
    segments = torch.stack(segments)
    masks = torch.stack(masks)
    return segments, masks, newLabels
