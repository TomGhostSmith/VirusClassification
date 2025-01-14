import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from collections import defaultdict

# 预测结果
prediction_file = "/Data/VirusClassification/dataset/refseq_2024_test/phylum/res/result.csv"

#层级信息
level = "Phylum"
predictions_df = pd.read_csv(prediction_file)

# 读取包含层级类别名和对应的 ID的文件
taxamap_file = f'/Data/VirusClassification/model/mapping/VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.json{level}_mapping.csv'



taxamap_df = pd.read_csv(taxamap_file)

taxamap_dict = {row[f'{level} ID']: row[f'{level}'] for _, row in taxamap_df.iterrows()}


predictions_df['seq_name'] = predictions_df['seq_name'].apply(lambda x: x.rsplit('_', 1)[0])

mean_values_df = predictions_df.groupby('seq_name').mean()

# 确定class_x 列的最大值及其列名
class_columns = [col for col in mean_values_df.columns if col.startswith("class_")]
mean_values_df["prediction_score"] = mean_values_df[class_columns].max(axis=1)
mean_values_df["prediction"] = mean_values_df[class_columns].idxmax(axis=1).str.extract(r'class_(\d+)')[0]
mean_values_df = mean_values_df.reset_index()

mean_values_df['prediction'] = mean_values_df['prediction'].astype(str)


taxamap_dict = {str(key): value for key, value in taxamap_dict.items()}

mean_values_df['taxa_prediction'] = mean_values_df['prediction'].map(taxamap_dict)


mean_values_df.to_csv(prediction_file+'.predictions_with_contig_level_scores.csv', index=False)

print(f'Results saved to {prediction_file}.predictions_with_contig_level_scores.csv')


new_order = ['seq_name','prediction','prediction_score','taxa_prediction']


df = mean_values_df[new_order]

df.to_csv(prediction_file+'.predictions_with_contig_level_scores_res.csv', index=False)

print(f'Results saved to {prediction_file}.predictions_with_contig_level_scores_res.csv')


#####只保留不含unknown的行，因为训练时候扩充标签了，每个层级有一些Unknown标签，结果需要删掉
df_filtered = df[~df['taxa_prediction'].str.contains('Unknown')]

df_filtered.to_csv(prediction_file+'.predictions_with_contig_level_scores_res_noUnknown.csv', index=False)

print(f'Results saved to {prediction_file}.predictions_with_contig_level_scores_res_noUnknown.csv')
