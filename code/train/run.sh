#esmtrain

export HF_ENDPOINT=https://hf-mirror.com



cd /home/wzy/tools/DNABERT_2-main/finetune


export DATA_PATH=/home/wzy/data/train_taxa/train_Genus_onlyvalid5seq_enlarge_genus_protein
#export DATA_PATH=/home/wzy/data/train_taxa/train_Genus_test5seq_fewsamples

outpath=${DATA_PATH}/out_esm2_t33_650M_UR50D

export MAX_LENGTH=1022 # Please set the number as 0.25 * your sequence length.
# e.g., set it as 250 if your DNA sequences have 1000 nucleotide bases
# This is because the tokenized will reduce the sequence length by about 5 times
export LR=3e-5

per_device_eval_batch_size=2
num_train_epochs=10
save_steps=10000
warmup_steps=5000

# Training use DataParallel
python esm2_train_taxa_protein.py \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name balance \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${per_device_eval_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs ${num_train_epochs} \
    --save_steps ${save_steps} \
    --output_dir ${outpath}/esm2_t33_650M_UR50D_MAX_LENGTH_${MAX_LENGTH}_per_device_batch_size_${per_device_eval_batch_size}_num_train_epochs_${num_train_epochs}_save_steps_${save_steps}_lr_${LR}_save_steps_${save_steps} \
    --evaluation_strategy steps \
    --eval_steps ${save_steps} \
    --warmup_steps ${warmup_steps} \
    --logging_steps 2000 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \
    --save_model True --metric_for_best_model "accuracy"
    