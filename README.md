# VirTaxonomer

A deep-learning based model to taxonomize virus

### Model Architecture

VirTaxonomer has two major modules: Virus Identify Module and Virus Classification Module.

The Virus Identify Module has two parts: Minimap part and ESM prediction part. For each sequence, Minimap is used to check if the sequence can be aligned to a reference sequence in VMRv4. If the alignment is of low quality, the ESM will determine if the sequence is a virus

The Virus Classification Module has two parts as well: Minimap part and machine learning part. For the sequence with high quality alignment obtained by Minimap, they will be classified according to the corresponding species. For the other sequences, there are 7 machine learning models to classify them.

The overall workflow of VirTaxonomer is shown below.

![](image/workflow.png)

### How to run VirTaxonomer

1. Download models from [https://zenodo.org/records/14649352](https://zenodo.org/records/14649352), extract them and put the files in the corresponding folder

```
model_root
├── class
│   └── esm2_t33_512
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── family
│   └── esm2_t33_512
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── genus
│   └── esm2_t33_256
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── kingdom
│   └── esm2_t33_256
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── mapping
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.json
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonClass_mapping.csv
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonFamily_mapping.csv
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonGenus_mapping.csv
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonKingdom_mapping.csv
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonOrder_mapping.csv
│   ├── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonPhylum_mapping.csv
│   └── VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.jsonRealm_mapping.csv
├── order
│   └── esm2_t33_512
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── phylum
│   └── esm2_t33_512
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── realm
│   └── esm2_t33_512
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── viral_identify
│   └── esm2_t30_512
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
└── VMRv4
├── VMRv4.fasta
└── VMRv4_names.json
```

2. Set the environment. Conda is recommended

```
conda env create -f environment.yml
conda activate VirTaxonomer
conda install minimap prodigal-gv -c bioconda
```


3. Modify `runVirTaxonomer.sh`. You may need to change:

   - `fasta_path`: where to find the fasta file to predict
   - `model_path`: the path of model root folder
   - `output_path`: the path of output folder

   The following parameters can be modified optionally:

   - `ML_strategy`: the value can be `highest` or `bottomup`, which changes how machine learning module works
   - `restrict`: the value can be `genus` or `species`, which restrict the output rank
   - `batch_size`: the batch size for machine learning module

   If you want to get the same result as our provided, use the following parameters:

   - Result 1: `ML_strategy=bottomup`, `restrict=genus`
   - Result 2: `ML_strategy=bottomup`, `restrict=species`
   - Result 3: `ML_strategy=highest`, `restrict=genus`

4. Run `runVirTaxonomer.sh` 

   