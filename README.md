# VirTaxonomer

A deep-learning based model to taxonomize virus

### How to run VirTaxonomer

1. Download model from URL, extract it and put the file in the corresponding folder

```
data_root
└── model
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
    │   └── esm2_t33_512
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
conda install minimap prodigal-gv -c bioconda
conda activate VirTaxonomer
```


3. Modify `runVirTaxonomer.sh`. You may need to change:

   - `fasta_path`: where to find the fasta file to predict
   - `data_root`: the path storing model, results, etc.

   The following parameters can be modified optionally:

   - `ML_strategy`: the value can be `highest` or `bottomup`, which changes how machine learning module works
   - `restrict`: the value can be `genus` or `species`, which restrict the output rank

4. Run `runVirTaxonomer.sh` and the result will be stored at `data_root/results`