contig_file=genbank_2024_test-genus.fasta

prodigal-gv -i ${contig_file} -o ${contig_file}.genes \
-d ${contig_file}.genes.fasta -a ${contig_file}.proteins.faa -p meta

#####################
#${contig_file}.proteins.faa 需要删掉不需要esm预测的序列



######################
#cd path to predict_esm.py

#####################
######################
#viral_identify
#Model_path=
######################
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python predict_esm.py \
--model_pth ${Model_path} \
--input ${contig_file}.proteins.faa \
--output viral_identify_res \
--base_model_pth facebook/esm2_t30_150M_UR50D --max_len 512 --n_class 2

##这样得到的是每个protein是否是病毒，需要对每个序列取均值，得到最终病毒识别结果
##参考predict_merge_gene_to_contig_classi_res.py
output=result.csv.merge_gene_to_contig.csv

sed -n 's/^\([^,]*\),virus,.*/\1/p' ${output} > ${output}.phage.tsv

#####################
#层级分类
#${contig_file}.proteins.faa 需要删掉非病毒序列

######################
######################
#genus....
#Model_path= ;max_len  n_class 需要根据最终确定的模型来看
#n_class=
#max_len=
#层级分类都是esm2_t33_650M_UR50D
######################
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python predict_esm.py \
--model_pth ${Model_path} \
--input ${contig_file}.proteins.faa \
--output viral_identify_res \
--base_model_pth facebook/esm2_t33_650M_UR50D --max_len ${max_len} --n_class ${n_class}

##这样得到的是每个protein的类别，需要对每个序列取均值，得到最终预测结果
#只保留不含unknown的行，因为训练时候扩充标签了，每个层级有一些Unknown标签，结果需要删掉
##参考gen_contig_level_res_from_prob.py

