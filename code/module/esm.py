import pandas

from config import config
from prototype.module import Module
from moduleResult.virusPredictionResult import VirusPredictionResult
from module.esmRunner import ESMRunner
from entity.sample import Sample

class ESM(Module):
    def __init__(self, shortName):
        self.viruses = set()
        # names = {
        #     "150M_256": "esm2_t30_150M_UR50D_dnainput_scl_MAX_LENGTH_256_predicted_virus_names.tsv",
        #     "150M_512": "esm2_t30_150M_UR50D_MAX_LENGTH_512_predicted_virus_names.tsv",
        #     "650M_256": "esm2_t33_650M_UR50D_MAX_LENGTH_256_predicted_virus.tsv",
        #     "650M_256_merge": "esm2_t33_650M_UR50D_MAX_LENGTH_256_12_result.csv.merge_gene_to_contig.csv.phage.tsv"
        # }
        # self.models = {
        #     "150M_512": f"{config.modelRoot}/viral_identify/esm2_t30_512"
        # }
        # self.fullName = names[shortName]
        # self.modelPath = self.models[shortName]
        self.model = ESMRunner(512, f"{config.modelRoot}/viral_identify/esm2_t30_512", "facebook/esm2_t30_150M_UR50D", 2, config.esmBatchSize)
        
        super().__init__(f'esm-{shortName}')

    def run(self, samples:list[Sample]):
        results = list()
        viruses = set()
        self.model.run(samples)

        df = pandas.read_csv(self.model.tempResCSV)

        df['seq_name'] = df['seq_name'].apply(lambda x: x.rsplit('_', 1)[0])


        df['class_0'] = pandas.to_numeric(df['class_0'])
        df['class_0_mean'] = df.groupby('seq_name')['class_0'].transform('mean')

        df['class_1'] = pandas.to_numeric(df['class_1'])
        df['class_1_mean'] = df.groupby('seq_name')['class_1'].transform('mean')

        df['prediction'] = df.apply(lambda x: 'virus' if x['class_1_mean'] > x['class_0_mean'] else 'non-virus', axis=1)


        df_result = df[['seq_name', 'prediction', 'class_0_mean','class_1_mean']].drop_duplicates()

        for row in df_result.itertuples:
            if row.prediction == 'virus':
                viruses.add(row.seq_name)

        for sample in self.viruses():
            if sample.id in viruses:
                results.append(VirusPredictionResult())
            else:
                results.append(None)
        
        return results