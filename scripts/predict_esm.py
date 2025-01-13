#This file is modified from https://github.com/ChengPENG-wolf/ViraLM/blob/main/viralm.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.nn import Softmax
from Bio import SeqIO
from torch import nn
import transformers
import argparse
import torch
import csv
import os

parser = argparse.ArgumentParser(description='a Python library designed for predicting viral taxonomy '
                                             'from metagenomic data.')


parser.add_argument('--input', type=str, help='path to the input file (fasta format, protein sequences)')
parser.add_argument('--output', type=str, help='output directory for storing results', default='result')
parser.add_argument('--batch_size', type=int, help='batch size for processing sequences', default=64)
parser.add_argument('--max_len', type=int, required=True, help='maximum sequence length for the pretrained model', default=256)
parser.add_argument('--model_pth', metavar='DIR',
                    help='directory containing the trained model')
parser.add_argument('--base_model_pth', metavar='DIR',
                    help='path to the base model')
parser.add_argument('--n_class', type=int, required=True, help='number of classes for prediction')

args = parser.parse_args()


input_pth = args.input
output_pth = args.output
batch_size = args.batch_size
cache_dir = f'{output_pth}/cache'
max_len = args.max_len
model_pth = args.model_pth
base_model_pth = args.base_model_pth
filename = input_pth.rsplit('/')[-1].split('.')[0]
n_class = args.n_class

if not os.path.exists(model_pth):
    print(f'Error: Model directory {model_pth} missing or unreadable')
    exit(1)

if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

def preprocee_data(input_pth):
    filename = input_pth.rsplit('/')[-1].split('.')[0]
    with open(f"{cache_dir}/{filename}_temp.csv", "w") as f:
        f.write(f'sequence,accession\n')
        for record in SeqIO.parse(input_pth, "fasta"):
            sequence = str(record.seq).upper()
            f.write(f'{sequence},{record.id}\n')


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

def tokenize_function(examples):
    return tokenizer(examples["sequence"], truncation=True)



preprocee_data(input_pth)

model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model_pth,
                                                                        # cache_dir=cache_dir,
                                                                        num_labels=n_class,
                                                                        trust_remote_code=True)

model.load_state_dict(torch.load(model_pth+"/pytorch_model.bin", map_location=torch.device('cpu'), weights_only=True), strict=False)

tokenizer = AutoTokenizer.from_pretrained(
    model_pth,
    model_max_length=max_len,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
test_dataset = load_dataset('csv', data_files={'test': f'{cache_dir}/{filename}_temp.csv'}, cache_dir=cache_dir)
tokenized_datasets = test_dataset.map(tokenize_function, batched=True, batch_size=batch_size, remove_columns=["sequence"])
tokenized_datasets = tokenized_datasets.with_format("torch")
test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if torch.cuda.device_count() > 1:
    print(f'\nRunning on {torch.cuda.device_count()} GPUs.')
    model = nn.DataParallel(model)
else:
    print(f'\nRunning on {device}.')
model.to(device)

softmax = Softmax(dim=0)
model.eval()
result = {}


with torch.no_grad():
    for step, batch in enumerate(test_loader):
        labels = batch['labels']
        batch.pop('labels')
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()

        for i in torch.arange(len(labels)):
            probabilities = softmax(torch.tensor(logits[i])).tolist()

            seq_name = labels[i]

            if seq_name not in result:
                result[seq_name] = []

            result[seq_name].append(probabilities)


fieldnames = ['seq_name'] + [f'class_{i}' for i in range(n_class)]

with open(f'{output_pth}/result.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for seq_name, probabilities in result.items():
        row = {'seq_name': seq_name}

        for idx, prob in enumerate(probabilities[0]):
            row[f'class_{idx}'] = prob

        writer.writerow(row)

print("Result saved to result.csv")


