import pandas as pd

input_file="/mnt/ssd/wzy/data/viral_taxassign/ICTVdata/shitao_data/refseq/gene_level/viral_env_balance_1795254_esm2_t30_150M_UR50D_MAX_LENGTH_512_per_device_batch_size_12_num_train_epochs_5_save_steps_50000_lr_3e-5_save_steps_50000_checkpoint-1400000/result.csv"


df = pd.read_csv(input_file)

df['seq_name'] = df['seq_name'].apply(lambda x: x.rsplit('_', 1)[0])


df['class_0'] = pd.to_numeric(df['class_0'])
df['class_0_mean'] = df.groupby('seq_name')['class_0'].transform('mean')

df['class_1'] = pd.to_numeric(df['class_1'])
df['class_1_mean'] = df.groupby('seq_name')['class_1'].transform('mean')

df['prediction'] = df.apply(lambda x: 'virus' if x['class_1_mean'] > x['class_0_mean'] else 'non-virus', axis=1)


df_result = df[['seq_name', 'prediction', 'class_0_mean','class_1_mean']].drop_duplicates()

df_result.to_csv(input_file + ".merge_gene_to_contig.csv", index=False)

