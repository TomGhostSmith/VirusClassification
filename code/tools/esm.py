#This file is modified from https://github.com/ChengPENG-wolf/ViraLM/blob/main/viralm.py
import sys
sys.path.append("code")
import json
from transformers import AutoTokenizer
from datasets import Dataset
import multiprocessing
from utils import IOUtils
import transformers
from concurrent.futures import ProcessPoolExecutor
import torch
from torch.nn import Softmax
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from entity.proteinSample import ProteinSample

from utils import parallelUtils


def loadDataset(stdin, tokenizer, batchSize):
    accessions = []
    sequences = []

    for sample in IOUtils.readSamples(stdin, ProteinSample):
        accessions.append(sample.id)
        sequences.append(str(sample.seq.seq))

    testset = Dataset.from_dict({
        "accession": accessions,
        "sequence": sequences
    })


    def tokenize_function(examples):  # do not padding in the tokenizer but in batch: prevent too many padding if one sequence is too long
        return tokenizer(examples["sequence"], truncation=True)
    
    def collateData(batch):
        input_ids, accessions = tuple([instance[key] for instance in batch] for key in ("input_ids", "accession"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return dict(
            input_ids=input_ids,
            accessions=accessions,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    
    l = parallelUtils.acquire_lock(parallelUtils.CPULock)

    tokenized_datasets = testset.map(tokenize_function, 
                                    batched=True, 
                                    batch_size=batchSize,
                                    remove_columns=["sequence"], 
                                    num_proc=multiprocessing.cpu_count()).with_format("torch")
    
    parallelUtils.release_lock(l)

    test_loader = DataLoader(tokenized_datasets, batch_size=batchSize, collate_fn=collateData)

    return test_loader

def loadModel(baseModelFolder, modelFolder, n_class, device):
    if (device  != "auto"):
        device = {"": device}
    model = transformers.AutoModelForSequenceClassification.from_pretrained(baseModelFolder,
                                                                            num_labels=n_class,
                                                                            trust_remote_code=True,
                                                                            torch_dtype=torch.float16,
                                                                            device_map=device
                                                                            )

    model.load_state_dict(torch.load(f"{modelFolder}/pytorch_model.bin", map_location=torch.device('cpu')), strict=False)

    return model

def runESM(test_loader, model, device):
    softmax = Softmax(dim=0)

    with torch.no_grad():
        # for batch in tqdm(test_loader, total=len(test_loader)):
        for batch in test_loader:
            accessions = batch['accessions']
            if (device.startswith("cuda")):
                batch = {k: v.to(device) for k, v in batch.items() if k != "accessions"}
            elif (device == "auto"):
                batch = {k: v.to(model.device) for k, v in batch.items() if k != "accessions"}
            else:
                batch = {k: v for k, v in batch.items() if k != "accessions"}

            outputs = model(**batch, output_hidden_states=True)
            
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

            for i in torch.arange(len(accessions)):
                probabilities = softmax(torch.tensor(logits[i])).numpy()
                seq_name = accessions[i]

                sys.stdout.write(f"{seq_name}\t{IOUtils.encodeBase64(probabilities)}\t{IOUtils.encodeBase64(cls_embeddings[i])}\t{IOUtils.encodeBase64(ave_embeddings[i])}\n")



def main():
    modelFolder = sys.argv[1]
    baseModelFolder = sys.argv[2]
    maxLen = int(sys.argv[3])
    batchSize = int(sys.argv[4])
    n_class = int(sys.argv[5])
    device = sys.argv[6]

    tokenizer = AutoTokenizer.from_pretrained(
        modelFolder,
        model_max_length=maxLen,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    test_loader = loadDataset(sys.stdin, tokenizer, batchSize)
    model = loadModel(baseModelFolder, modelFolder, n_class, device)
    model.eval()

    runESM(test_loader, model, device)

if (__name__ == "__main__"):
    main()