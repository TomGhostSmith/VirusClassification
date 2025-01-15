export PYTHONPATH=$PYTHONPATH:./code

# fasta_path="/Data/VirusClassification/dataset/Challenge/Challenge.fasta"
fasta_path="/Data/VirusClassification/dataset/refseq_2024_test/phylum/refseq_2024_test-phylum.fasta"
model_path="/Data/ICTVPublish2/model"
output_path="/Data/ICTVPublish2/results"
batch_size=128
ML_strategy="bottomup"
restrict="genus"

python ./code/main.py --input $fasta_path --model $model_path --output $output_path --ML $ML_strategy --restrict $restrict --batchsize $batch_size