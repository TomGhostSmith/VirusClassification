import os
import pandas
from config import config
from prototype.module import Module
from moduleResult.mlResult import MLResult
from entity.sample import Sample
from module.esmRunner import ESMRunner

class MLModule(Module):
    def __init__(self, strategy="topdown", thresh=0.45, gen='1111000'):
        self.strategy = strategy
        self.thresh = thresh
        self.gen = gen
        super().__init__(f'ML-stratgy={strategy};th={thresh}, gen={gen}')
        # self.baseName = self.moduleName
        self.resultDict:dict[str, MLResult] = dict()

        realmParams = [
            (256, f"{config.modelRoot}/realm/esm2_t33_256"),
            (512, f"{config.modelRoot}/realm/esm2_t33_512")
        ]
        kingdomParams = [
            (256, f"{config.modelRoot}/kingdom/esm2_t33_256"),
            (512, f"{config.modelRoot}/kingdom/esm2_t33_512")
        ]
        phylumParams = [
            (256, f"{config.modelRoot}/phylum/esm2_t33_256"),
            (512, f"{config.modelRoot}/phylum/esm2_t33_512")
        ]
        classParams = [
            (256, f"{config.modelRoot}/class/esm2_t33_256"),
            (512, f"{config.modelRoot}/class/esm2_t33_512")
        ]
        orderParams = [
            (512, f"{config.modelRoot}/order/esm2_t33_512")
        ]
        familyParams = [
            (512, f"{config.modelRoot}/family/esm2_t33_512")
        ]
        genusParams = [
            (256, f"{config.modelRoot}/genus/esm2_t33_256"),
        ]

        realmParam = realmParams[int(gen[0])]
        kingdomParam = kingdomParams[int(gen[1])]
        phylumParam = phylumParams[int(gen[2])]
        classParam = classParams[int(gen[3])]
        orderParam = orderParams[int(gen[4])]
        familyParam = familyParams[int(gen[5])]
        genusParam = genusParams[int(gen[6])]

        self.modelParams = {
            "realm": (*realmParam, "facebook/esm2_t33_650M_UR50D", 29, config.mlBatchSize),
            "kingdom": (*kingdomParam, "facebook/esm2_t33_650M_UR50D", 40, config.mlBatchSize),
            "phylum": (*phylumParam, "facebook/esm2_t33_650M_UR50D", 51, config.mlBatchSize),
            "class": (*classParam, "facebook/esm2_t33_650M_UR50D", 76, config.mlBatchSize),
            "order": (*orderParam, "facebook/esm2_t33_650M_UR50D", 981, config.mlBatchSize),
            "family": (*familyParam, "facebook/esm2_t33_650M_UR50D", 1129, config.mlBatchSize),
            "genus": (*genusParam, "facebook/esm2_t33_650M_UR50D", 3523, config.mlBatchSize),
        }

        
    def run(self, samples:list[Sample]):
        unterminatedSamples = samples
        if (self.strategy.startswith('topdown')):
            for rank in list(self.modelParams.keys()):
                unterminatedSamples = self.runModel(unterminatedSamples, rank)
                if (len(unterminatedSamples) == 0):
                    break
        elif (self.strategy.startswith('bottomup')):
            for rank in reversed(list(self.modelParams.keys())):
                unterminatedSamples = self.runModel(unterminatedSamples, rank)
                if (len(unterminatedSamples) == 0):
                    break
        else:  # highest, we need to run all the rank
            for rank in list(self.modelParams.keys()):
                self.runModel(samples, rank)

        results = list()
        for sample in samples:
            if (sample.id in self.resultDict):
                results.append(self.resultDict[sample.id])
            else:
                results.append(None)
        
        return results


    def runModel(self, samples:list[Sample], rank:str)->list[Sample]:

        abbr = self.modelParams[rank][1].split('/')[-1]
        cachedRes = f"{config.resultRoot}/cachedResults/{config.datasetName}/MLResult/{rank}_{abbr}.csv"
        if (os.path.exists(cachedRes)):
            df_filtered = pandas.read_csv(cachedRes)
        else:

            model = ESMRunner(*self.modelParams[rank])
            model.run(samples)

            
            level = rank.capitalize()
            predictions_df = pandas.read_csv(model.tempResCSV)

            taxamap_file = f'{config.modelRoot}/mapping/VMR_MSL39_v4.json.processed_data.json.nosub_addunknown.json{level}_mapping.csv'

            taxamap_df = pandas.read_csv(taxamap_file)

            taxamap_dict = {row[f'{level} ID']: row[f'{level}'] for _, row in taxamap_df.iterrows()}


            predictions_df['seq_name'] = predictions_df['seq_name'].apply(lambda x: x.rsplit('_', 1)[0])

            mean_values_df = predictions_df.groupby('seq_name').mean()

            class_columns = [col for col in mean_values_df.columns if col.startswith("class_")]
            mean_values_df["prediction_score"] = mean_values_df[class_columns].max(axis=1)
            mean_values_df["prediction"] = mean_values_df[class_columns].idxmax(axis=1).str.extract(r'class_(\d+)')[0]
            mean_values_df = mean_values_df.reset_index()

            mean_values_df['prediction'] = mean_values_df['prediction'].astype(str)

            taxamap_dict = {str(key): value for key, value in taxamap_dict.items()}

            mean_values_df['taxa_prediction'] = mean_values_df['prediction'].map(taxamap_dict)

            new_order = ['seq_name','prediction','prediction_score','taxa_prediction']

            df = mean_values_df[new_order]

            df_filtered = df[~df['taxa_prediction'].str.contains('Unknown')]

            del model

        thisRes = dict()

        for row in df_filtered.itertuples():
            id = row.seq_name
            score = row.prediction_score
            res = row.taxa_prediction
            thisRes[id] = (res, score)


            # if (id not in self.resultDict):
            #     self.resultDict[id] = MLResult(self.strategy, self.thresh)
            # self.resultDict[id].addResult(res, score)

        unTerminatedSamples:list[Sample] = list()
        for sample in samples:
            if sample.id not in self.resultDict:
                self.resultDict[sample.id] = MLResult(self.strategy, self.thresh)
            if (sample.id in thisRes):
                self.resultDict[sample.id].addResult(*thisRes[sample.id])
            if not (self.resultDict[sample.id].terminate):
                unTerminatedSamples.append(sample)
        
        return unTerminatedSamples