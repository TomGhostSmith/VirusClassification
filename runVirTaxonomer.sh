export PYTHONPATH=$PYTHONPATH:./code

fasta_path="/Data/VirusClassification/dataset/refseq_2024_test/phylum/refseq_2024_test-phylum.fasta"
data_root="/Data/ICTVPublish2"
ML_strategy="highest"
restrict="genus"

python ./code/main.py --input $fasta_path --data $data_root --ML $ML_strategy --restrict $restrict