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
import csv
import os
import base64

from entity.sample import Sample
from entity.proteinSample import ProteinSample
from config import config
from Bio import SeqIO
from utils import IOUtils
from utils.NucleotideUtils import NucleotideUtils


class DNALMRunner():
    def __init__(self, modelName, batchSize=64, threads=multiprocessing.cpu_count()):
        self.modelName = modelName
        self.batchSize = batchSize
        # self.manualLoadConfig = manualLoadConfig
        self.threads = threads

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


    def run(self, samples:list[Sample|ProteinSample]|list[tuple[str, str]]):  # can take both DNAs and cDNAs
        def tokenize_function(dna):
            dna["input_ids"] = self.tokenizer(dna["sequence"])["input_ids"]
            return dna
            return self.tokenizer(dna)  # do not use truncation because DNABert allows long sequence
        
        labels = []
        sequences = []
        for sample in samples:
            if (isinstance(sample, Sample) or isinstance(sample, ProteinSample)):
                labels.append(sample.id)
                sequences.append(str(sample.seq.seq).upper())
            else:
                labels.append(sample[0])
                sequences.append(str(sample[1]).upper())
        
        testset = Dataset.from_dict({
            "accession": labels,
            "sequence": sequences
        })


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modelName,
            trust_remote_code=True
        )

        tokenized_datasets = testset.map(tokenize_function, 
                                         batched=True, 
                                         batch_size=self.batchSize, 
                                         remove_columns=["sequence"], 
                                         num_proc=self.threads).with_format("torch")
        # tokenized_datasets = testset.map(tokenize_function, 
        #                                  batched=True, 
        #                                  batch_size=self.batchSize, 
        #                                  remove_columns=["sequence"])
        # tokenized_datasets = tokenized_datasets.with_format("torch")
        test_loader = DataLoader(tokenized_datasets, batch_size=self.batchSize, collate_fn=self.collateData)

        softmax = Softmax(dim=0)
        result = {}

        self.loadModel()

        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                labels = batch['labels']
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}

                outputs = self.model(**batch, output_hidden_states=True)
                last_hidden_state = outputs[0]
                cls_embedding = last_hidden_state[:, 0, :]
                # ave_embedding = torch.mean(last_hidden_state, dim=1)  # note: this won't work because there are padding
                masks = batch['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden_state * masks
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = masks.sum(dim=1)
                ave_embedding = sum_hidden / lengths
                cls_embeddings = cls_embedding.detach().cpu().contiguous().numpy()
                ave_embeddings = ave_embedding.detach().cpu().contiguous().numpy()

                # logits = outputs.logits.cpu().numpy()

                for i in torch.arange(len(labels)):
                    # probabilities = softmax(torch.tensor(logits[i])).numpy()
                    # embedding_str = base64.b64encode(embedding.tobytes()).decode('ascii')
                    # embedding = numpy.frombuffer(base64.b64decode(embedding_str), dtype=numpy.float16)  # note: we are using float16
                    seq_name = labels[i]

                    result[seq_name] = (cls_embeddings[i], ave_embeddings[i])


        # lines = dict()
        # with open(self.tempResTSV, 'wt') as fp:
        #     # write head 
        #     line = "seq_name\t" + "\t".join([f'class_{i}' for i in range(self.n_class)]) + "\n"
        #     lines["title"] = line
        #     fp.write(line)
        #     for seq_name, probabilities in result.items():
        #         probabilities = [str(p) for p in probabilities]
        #         line = f"{seq_name}\t" + "\t".join(probabilities) + "\n"
        #         fp.write(line)
        #         lines[seq_name] = line
        
        return result