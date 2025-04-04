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
import torch
import math
import csv
import os

from entity.sample import Sample
from config import config
from Bio import SeqIO
from utils import IOUtils

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "accession"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = labels
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class ESMRunner():
    def __init__(self, maxLen, modelFolder, baseModelFolder, n_class, batchSize=64, cachedResult=None):
        self.tempDNAFasta = f"{config.cacheFolder}/DNAs.fasta"
        self.tempProFasta = f"{config.cacheFolder}/proteins.fasta"
        self.tempProCSV = f"{config.cacheFolder}/proteins.csv"
        self.tempResCSV = f"{config.cacheFolder}/res.csv"

        self.maxLen = maxLen
        self.modleFolder = modelFolder
        self.baseModelFolder = baseModelFolder
        self.n_class = n_class
        self.batchSize = batchSize

        self.useCache = cachedResult is not None

        if (self.useCache):
            self.tempResCSV = cachedResult

    def loadModel(self):
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.baseModelFolder,
                                                                                num_labels=self.n_class,
                                                                                trust_remote_code=True,
                                                                                torch_dtype=torch.float16,
                                                                                )

        self.model.load_state_dict(torch.load(f"{self.modleFolder}/pytorch_model.bin", map_location=torch.device('cpu')), strict=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modleFolder,
            model_max_length=self.maxLen,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )

        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        
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

        def runSingleProdigal(self, idx, dnaFasta, outputFasta):
            subprocess.run(f"prodigal-gv -i {dnaFasta} -a {outputFasta} -p meta", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return idx


    def run(self, samples:list[Sample]):
        if (self.useCache):
            return
        
        # 1. store samples to a fasta file
        # with open(self.tempDNAFasta, 'wt') as fp:
        #     for sample in samples:
        #         SeqIO.write(sample.seq, fp, 'fasta')
        
        # 2. convert DNA.fasta to protein.fasta (redundant though, do not store result for each one)
        recordPerThread = 100
        splitCount = math.ceil(len(samples)/recordPerThread)
        proteinFastaFP = open(self.tempProFasta, 'wt')

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            asyncResults = list()
            for i in range(splitCount):
                IOUtils.writeSampleFasta(samples[recordPerThread * i : recordPerThread * (i+1)], f"{self.tempDNAFasta}.{i}")
        
                asyncResults.append(pool.apply_async(self.runSingleProdigal, 
                                                [i, f"{self.tempDNAFasta}.{i}", f"{self.tempProFasta}.{i}"]))
                
            for asyncResult in asyncResults:
                idx = asyncResult.get()
                with open(f"{self.tempProFasta}.{idx}") as fp:
                    proteinFastaFP.writelines(fp.readlines())
        # subprocess.run(f"prodigal-gv -i {self.tempDNAFasta} -a {self.tempProFasta} -p meta", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for i in range(splitCount):
            os.remove(f"{self.tempDNAFasta}.{i}")

        # 3. preprocee protein.fasta to a csv file
        with open(self.tempProCSV, "w") as f:
            f.write(f'sequence,accession\n')
            for record in SeqIO.parse(self.tempProFasta, "fasta"):
                sequence = str(record.seq).upper()
                f.write(f'{sequence},{record.id}\n')

        # 4. load model, run and save result
        self.loadModel()
        self.runModel()

    def runModel(self):
        def tokenize_function(examples):
            return self.tokenizer(examples["sequence"], truncation=True)

        test_dataset = load_dataset('csv', data_files={'test': self.tempProCSV}, cache_dir=config.cacheFolder)
        tokenized_datasets = test_dataset.map(tokenize_function, batched=True, batch_size=self.batchSize, remove_columns=["sequence"])
        tokenized_datasets = tokenized_datasets.with_format("torch")
        test_loader = DataLoader(tokenized_datasets["test"], batch_size=self.batchSize, collate_fn=self.data_collator)

        softmax = Softmax(dim=0)
        result = {}


        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                labels = batch['labels']
                batch.pop('labels')
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                logits = outputs.logits.cpu().numpy()

                for i in torch.arange(len(labels)):
                    probabilities = softmax(torch.tensor(logits[i])).tolist()

                    seq_name = labels[i]

                    if seq_name not in result:
                        result[seq_name] = []

                    result[seq_name].append(probabilities)


        fieldnames = ['seq_name'] + [f'class_{i}' for i in range(self.n_class)]

        with open(self.tempResCSV, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for seq_name, probabilities in result.items():
                row = {'seq_name': seq_name}

                for idx, prob in enumerate(probabilities[0]):
                    row[f'class_{idx}'] = prob

                writer.writerow(row)
