#This file is modified from https://github.com/ChengPENG-wolf/ViraLM/blob/main/viralm.py
import sys
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


def loadDataset(stdin, tokenizer, batchSize):
    labels = []
    sequences = []

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        labels.append(s["label"])
        sequences.append(s["seq"])

    testset = Dataset.from_dict({
        "accession": labels,
        "sequence": sequences
    })


    def tokenize_function(examples):  # do not padding in the tokenizer but in batch: prevent too many padding if one sequence is too long
        return tokenizer(examples["sequence"], truncation=True)
    
    def collateData(batch):
        input_ids, labels = tuple([instance[key] for instance in batch] for key in ("input_ids", "accession"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = labels
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    tokenized_datasets = testset.map(tokenize_function, 
                                    batched=True, 
                                    batch_size=batchSize,
                                    remove_columns=["sequence"], 
                                    num_proc=multiprocessing.cpu_count()).with_format("torch")

    test_loader = DataLoader(tokenized_datasets, batch_size=batchSize, collate_fn=collateData)

    return test_loader

def loadModel(baseModelFolder, modelFolder, n_class, device):
    # load models
    model = transformers.AutoModelForSequenceClassification.from_pretrained(baseModelFolder,
                                                                            num_labels=n_class,
                                                                            trust_remote_code=True,
                                                                            torch_dtype=torch.float16,
                                                                            )

    model.load_state_dict(torch.load(f"{modelFolder}/pytorch_model.bin", map_location=torch.device('cpu')), strict=False)

    if torch.cuda.device_count() > 1:
        # print(f'\nRunning on {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    else:
        # print(f'\nRunning on {device}.')
        pass
    
    model.to(device)

def runESM(test_loader, model, device, cacheProbFile, cacheCLSEmbFile, cacheAveEmbFile, nextOffset_prob, nextOffset_cls, nextOffset_ave):
    softmax = Softmax(dim=0)
    
    probLines = []
    clsLines = []
    aveLines = []

    cachedSamples_prob = {}
    cachedSamples_cls  = {}
    cachedSamples_ave  = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            labels = batch['labels']
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

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

            for i in torch.arange(len(labels)):
                probabilities = softmax(torch.tensor(logits[i])).numpy()
                # embedding_str = base64.b64encode(embedding.tobytes()).decode('ascii')
                # embedding = numpy.frombuffer(base64.b64decode(embedding_str), dtype=numpy.float16)  # note: we are using float16
                seq_name = labels[i]

                cachedSamples_prob[seq_name] = nextOffset_prob
                cachedSamples_cls[seq_name] = nextOffset_cls
                cachedSamples_ave[seq_name] = nextOffset_ave

                probText = f"{seq_name}\t{IOUtils.encodeBase64(probabilities)}\n"
                clsText = f"{seq_name}\t{IOUtils.encodeBase64(cls_embeddings[i])}\n"
                aveText = f"{seq_name}\t{IOUtils.encodeBase64(ave_embeddings[i])}\n"

                probLines.append(probText)
                clsLines.append(clsText)
                aveLines.append(aveText)

                nextOffset_prob += len(probText)
                nextOffset_cls += len(clsText)
                nextOffset_ave += len(aveText)

    cachedSamples_prob["nextOffset"] = nextOffset_prob
    cachedSamples_cls["nextOffset"] = nextOffset_cls
    cachedSamples_ave["nextOffset"] = nextOffset_ave

    # write results at the end to avoid interruption and offset not updated

    fp_prob = open(cacheProbFile, 'at')
    fp_cls  = open(cacheCLSEmbFile, 'at')
    fp_ave  = open(cacheAveEmbFile, 'at')

    fp_prob.writelines(probLines)
    fp_cls.writelines(clsLines)
    fp_ave.writelines(aveLines)

    fp_prob.close()
    fp_cls.close()
    fp_ave.close()

    sys.stdout.write(json.dumps(cachedSamples_prob) + "\n")
    sys.stdout.write(json.dumps(cachedSamples_cls) + "\n")
    sys.stdout.write(json.dumps(cachedSamples_ave) + "\n")


def main():
    # modelFolder = sys.argv[1]
    # baseModelFolder = sys.argv[2]
    # cacheProbFile   = sys.argv[3]
    # cacheCLSEmbFile = sys.argv[4]
    # cacheAveEmbFile = sys.argv[5]
    # maxLen = int(sys.argv[6])
    # batchSize = int(sys.argv[7])
    # n_class = int(sys.argv[8])
    # nextOffset_prob = int(sys.argv[9])
    # nextOffset_cls  = int(sys.argv[10])
    # nextOffset_ave  = int(sys.argv[11])
    modelFolder = "/Data/VirusClassification/model/genus/esm2_t33_256_enlarge_genus"
    baseModelFolder = "facebook/esm2_t33_650M_UR50D"
    cacheProbFile   = f"/Data/VirusClassification/cache/CachedResults/ESM_taxo_esm2_t33_256_enlarge_genus_prob.tmp"
    cacheCLSEmbFile = f"/Data/VirusClassification/cache/CachedResults/ESM_taxo_esm2_t33_256_enlarge_genus_cls_emb.tmp"
    cacheAveEmbFile = f"/Data/VirusClassification/cache/CachedResults/ESM_taxo_esm2_t33_256_enlarge_genus_ave_emb.tmp"
    maxLen = 256
    batchSize = 64
    n_class = 3523
    nextOffset_prob = 0
    nextOffset_cls  = 0
    nextOffset_ave  = 0

    tokenizer = AutoTokenizer.from_pretrained(
        modelFolder,
        model_max_length=maxLen,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    # run tokenizer with multiprocessing in an separate function to avoid inheret huge cache
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    test_loader = loadDataset(sys.stdin, tokenizer, batchSize)
    model = loadModel(baseModelFolder, modelFolder, n_class, device)
    model.eval()

    runESM(test_loader, model, device, cacheProbFile, cacheCLSEmbFile, cacheAveEmbFile, nextOffset_prob, nextOffset_cls, nextOffset_ave)        

if (__name__ == "__main__"):
    main()