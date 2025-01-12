import os
import json
import pandas
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from collections import defaultdict
from tqdm import tqdm

from config import config
from entity.sample import Sample
from prototype.mergeModule import MergeModule
from module.pipeline import Pipeline
from entity.taxoTree import taxoTree
from entity.taxoNode import TaxoNode
from entity.dataset import Dataset
from prototype.result import Result

class Evaluator():
    def __init__(self, models):
        if (isinstance(models, list)):
            self.models = models
        else:
            self.models = [models]


    def checkVirusFilter(self, dataset:Dataset=None):
        self.loadSamples(dataset)
        nameList = [model.moduleName for model in self.models]
        tps = list()
        tns = list()
        fps = list()
        fns = list()

        for model in self.models:
            tp, tn, fp, fn = 0, 0, 0, 0
            for sample in self.allSamples:
                if sample.stdResult is None:
                    if (sample.results[model.moduleName] is None):
                        tn += 1
                    else:
                        fp += 1
                else:
                    if (sample.results[model.moduleName] is None):
                        fn += 1
                    else:
                        tp += 1
            tps.append(tp)
            fps.append(fp)
            tns.append(tn)
            fns.append(fn)
        

        # generate report for focused samples
        df = pandas.DataFrame({
            'model': nameList,
            'totalVirus': [tps[0] + fns[0]]*len(nameList),
            'totalNonvirus': [tns[0] + fps[0]]*len(nameList),
            "TrueVirus": tps,
            "FalseVirus": fps,
            "TrueNonvirus": tns,
            "FalseNonVirus": fns
        })

        df.to_csv(f"{config.resultBase}/Summary_VirusPrediction.csv")



    # sampleRange could be: 'all', 'withResult', 'interested'
    # if dataset = None, then use the dataset in config
    def evaluate(self, sampleRange='all', dataset:Dataset=None, showDetails=True):
        self.loadSamples(dataset)
        title = 'Statistics'
        if sampleRange == 'interested':
            with open(f"{config.tempFolder}/interestedSamples.txt") as fp:
                interestedSamples = {l.strip() for l in fp}
            
            interestedSamples = list()
            for sample in self.allSamples:
                if sample.name in interestedSamples:
                    interestedSamples.append(sample)
        else:
            interestedSamples = list()
            for s in self.allSamples:
                if s.stdResult is not None:
                    interestedSamples.append(s)

        summaryRes = list()

        with open(f"{config.tempFolder}/resCount") as fp:
            pipelineIndex = int(fp.readline().strip())
        
        pipelineResTitle = ["pred minimap ref", "pred minimap mode", "pred minimap factor",
                            "esm",
                            "taxo minimap ref", "taxo minimap mode", "taxo minimap code",
                            "ML strategy", "ML cutoff", "ML gen",
                            "merge",
                            "pipelineIndex"]
        pipelineNames = list()

        requireTitle = not os.path.exists(f"{config.resultBase}/pipelines.csv")
        pipelineNameFP = open(f"{config.resultBase}/pipelines.csv", 'at')

        if (requireTitle):
            pipelineNameFP.write(",".join(pipelineResTitle) + "\n")
        
        modelNames = list()

        for model in tqdm(self.models, desc="evaluating"):
            validSamples = list()
            for sample in interestedSamples:
                if (sample.results[model.moduleName] is not None):
                    validSamples.append(sample)
            
            if (sampleRange == 'withResult'):
                focusedSamples = validSamples
            else:
                focusedSamples = interestedSamples

            totalCount = len(self.allSamples)
            includeCount = len(focusedSamples)
            validCount = len(validSamples)

            # generate report for focused samples
            df, weightedStat = self.analyseStatistics(focusedSamples, model.moduleName)
            summaryRes.append(weightedStat)

            if (isinstance(model, Pipeline)):
                df.to_csv(f"{config.resultBase}/{title}!n={totalCount};incl={includeCount};valid={validCount}!pipeline-{pipelineIndex}.csv", index=False)
            else:
                df.to_csv(f"{config.resultBase}/{title}!n={totalCount};incl={includeCount};valid={validCount}!{model.moduleName}.csv", index=False)            

            # perform separate calculation for MergeModel
            if (isinstance(model, MergeModule) and showDetails):
                partitions = dict()
                for sample in focusedSamples:
                    source = sample.info[model.moduleName]
                    if source not in partitions:
                        partitions[source] = list()
                    partitions[source].append(sample)


                for source, samples in partitions.items():
                    df, _ = self.analyseStatistics(samples, model.moduleName)
                    # print(f" ({source})")

                    df.to_csv(f"{config.resultBase}/{title}!n={totalCount};incl={includeCount};valid={validCount}!Source={source};m={len(samples)}!{model.moduleName}.csv", index=False)
            
            if (isinstance(model, Pipeline)):
                pipelineNames.append(model.getParamList())
                params = model.getParamList()
                params.append(str(pipelineIndex))
                pipelineNameFP.write(",".join(params) + "\n")
                modelNames.append(f"pipeline-{pipelineIndex}")
                pipelineIndex += 1
            else:
                modelNames.append(model.moduleName)

        with open(f"{config.tempFolder}/resCount", 'w') as fp:
            fp.write(str(pipelineIndex))

        pipelineNameFP.close()
                    
        summaryRes = list(zip(*summaryRes))

        summaryDF =  pandas.DataFrame({
            "model": modelNames,
            "Summary_Acc": summaryRes[0],
            "Summary_Precision": summaryRes[1],
            "Summary_Recall": summaryRes[2],
            "Summary_F1": summaryRes[3],
            "Summary_BinaryAcc": summaryRes[4],
            "Summary_BinaryPrecision": summaryRes[5],
            "Summary_BinaryRecall": summaryRes[6],
            "Summary_BinaryF1": summaryRes[7]
        })

        summaryDF.to_csv(f"{config.resultBase}/Summary_Evaluation.csv", index=False)

    # sampleRange could be: 'all', 'intersection', 'interested'
    def compare(self, sampleRange='all', dataset:Dataset=None):
        self.loadSamples(dataset)
        if (len(self.models) < 2):
            raise ValueError("You should compare at least 2 models")
        
        if sampleRange == 'interested':
            with open(f"{config.tempFolder}/interestedSamples.txt") as fp:
                interestedSamples = {l.strip() for l in fp}
            
            interestedSamples = list()
            for sample in self.allSamples:
                if sample.name in interestedSamples:
                    interestedSamples.append(sample)
        else:
            interestedSamples = self.allSamples
        
        if (sampleRange == 'intersection'):
            validSamples = list()
            for sample in interestedSamples:
                if (None not in sample.results.values()):
                    validSamples.append(sample)
            interestedSamples = validSamples

        totalCount = len(self.allSamples)
        includeCount = len(interestedSamples)
        with pandas.ExcelWriter(f"{config.resultBase}/Compare!n={totalCount};incl={includeCount}.xlsx", engine='xlsxwriter') as writer:
            for model in self.models:
                validCount = 0
                for sample in interestedSamples:
                    if (sample.results[model.moduleName] is not None):
                        validCount += 1


                # generate report for focused samples
                df, _ = self.analyseStatistics(interestedSamples, model.moduleName)

                nameDict = {
                    "ML-param=contig_most_frequent_t33_512": "t33_512",
                    "ML-param=contig_most_frequent_family_finetune_t33_256": "family_ft_t33_256",
                    "minimap-ref=VMRv4;mode=ont": "minimap"
                }
                name = nameDict[model.moduleName]

                df.to_excel(writer, sheet_name=name, index=False)

    def loadSamples(self, dataset:Dataset=None):
        def load():
            with open(f"{config.datasetBase}/length.json") as fp:
                lengths = json.load(fp)
            with open(f"{config.datasetBase}/isATCG.json") as fp:
                isATCG = json.load(fp)
            with open(f"{config.datasetBase}/answer.json") as fp:
                answers = json.load(fp)
            sampleList = list()
            for name in lengths.keys():
                sampleList.append(Sample(name, isATCG[name], lengths[name], answers[name]))
            return sampleList
        
        self.allSamples = list()
        if (dataset is not None):
            for _, major, minor in dataset.iterDatasets():
                config.majorDataset = major
                config.minorDataset = minor
                config.updatePath()
                thisSample = load()
                n = major if minor is None else f"{major}-{minor}"
                for model in tqdm(self.models, desc=n):
                    model.run()
                    model.getResults(thisSample)  # we have to getResults here, otherwise the path in config will be wrong
                self.allSamples += thisSample

            if (dataset.minorDatasets is not None and len(dataset.minorDatasets) == 1):
                config.minorDataset = dataset.minorDatasets[0]
            else:
                config.minorDataset = None

            config.updatePath()
        else:
            self.allSamples = load()
            for model in self.models:
                model.getResults(self.allSamples)
        return self.allSamples

        
    def analyseStatistics(self, samples, modelName):
        def macro_accuracy(y_true, y_pred):
            classes = set(y_true)
            class_accuracies = []
            for cls in classes:
                tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == cls)
                total_cls = y_true.count(cls)
                if total_cls > 0:
                    class_accuracies.append(tp / total_cls)
                else:
                    class_accuracies.append(0)
            return sum(class_accuracies) / len(classes)
        
        def getWeightedStatistics(statistics, ranks, sampleCounts):
            total = 0
            totalCount = 0
            for rank in ranks:
                if sampleCounts[rank] > 0:
                    total += statistics[rank] * sampleCounts[rank]
                    totalCount += sampleCounts[rank]
            return total / totalCount
            

        # Initialize metrics storage
        acc_sum, precision_sum, recall_sum, f1_sum = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        overall_acc, overall_precision, overall_recall, overall_f1 = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        binary_acc, binary_precision, binary_recall, binary_f1 = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        prediction_counts = defaultdict(int)
        true_label_counts = defaultdict(int)
        prediction_with_label_counts = defaultdict(int)

        taxonomic_levels = ["Realm", "Subrealm", "Kingdom", "Subkingdom", "Phylum", "Subphylum", "Class",
                        "Subclass", "Order", "Suborder", "Family", "Subfamily", "Genus", "Subgenus", "Species"]
        
        predictions = defaultdict(dict)
        true_labels = defaultdict(dict)

        # extract labels
        for sample in samples:
            stdNode = sample.stdResult
            pred:Result = sample.results[modelName]
            predictNode = pred.node if pred is not None else None

            # currently we do not care about virus prediction
            # stdNode can still be None because there are nonVirus samples 
            if (stdNode is not None):
                for node in stdNode.ICTVNode.path:
                    true_labels[sample][node.rank.capitalize()] = node.name

            if (predictNode is not None):
                for node in predictNode.ICTVNode.path:
                    predictions[sample][node.rank.capitalize()] = node.name

        # Calculate metrics for each taxonomic level
        for level in taxonomic_levels:
            y_true = []
            y_pred = []

            bin_true = list()
            bin_pred = list()

            for contig in true_labels.keys():
                true_label = true_labels[contig].get(level, 'Unknown')
                pred_label = predictions[contig].get(level, 'Unknown') if contig in predictions else 'Unknown'

                if pred_label != 'Unknown':
                    prediction_counts[level] += 1
                if pred_label != 'Unknown' and true_label != 'Unknown':
                    prediction_with_label_counts[level] += 1
                if true_label != 'Unknown':
                    true_label_counts[level] += 1

                if true_label != 'Unknown' and pred_label != 'Unknown':
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                
                bin_true.append(true_label != 'Unknown')
                bin_pred.append(pred_label != 'Unknown')

            if len(y_true) != 0:
                # Calculate macro-average metrics
                acc_sum[level] = macro_accuracy(y_true, y_pred)
                precision_sum[level] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall_sum[level] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1_sum[level] = f1_score(y_true, y_pred, average='macro')

                # Overall metrics (weighted)
                overall_acc[level] = accuracy_score(y_true, y_pred)
                overall_precision[level] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                overall_recall[level] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                overall_f1[level] = f1_score(y_true, y_pred, average='weighted')

                # known/unknown metrics
                binary_acc[level] = accuracy_score(bin_true, bin_pred)
                binary_precision[level] = precision_score(bin_true, bin_pred, pos_label=True)
                binary_recall[level] = recall_score(bin_true, bin_pred, pos_label=True)
                binary_f1[level] = f1_score(bin_true, bin_pred, pos_label=True)
            else:
                acc_sum[level] = precision_sum[level] = recall_sum[level] = f1_sum[level] = '-'
                overall_acc[level] = overall_precision[level] = overall_recall[level] = overall_f1[level] = '-'
                binary_acc[level] = binary_precision[level] = binary_recall[level] = binary_f1[level] = '-'

        # Save results to CSV
        results_data = {
            "Level": taxonomic_levels,
            "ACC_macro": [acc_sum[level] for level in taxonomic_levels],
            "Precision_macro": [precision_sum[level] for level in taxonomic_levels],
            "Recall_macro": [recall_sum[level] for level in taxonomic_levels],
            "F1_macro": [f1_sum[level] for level in taxonomic_levels],
            "ACC_overall": [overall_acc[level] for level in taxonomic_levels],
            "Precision_overall": [overall_precision[level] for level in taxonomic_levels],
            "Recall_overall": [overall_recall[level] for level in taxonomic_levels],
            "F1_overall": [overall_f1[level] for level in taxonomic_levels],
            "ACC_Binary": [binary_acc[level] for level in taxonomic_levels],
            "Precision_Binary": [binary_precision[level] for level in taxonomic_levels],
            "Recall_Binary": [binary_recall[level] for level in taxonomic_levels],
            "F1_Binary": [binary_f1[level] for level in taxonomic_levels],
            "#Predictions": [prediction_counts[level] for level in taxonomic_levels],
            "#Predictions_with_labels": [prediction_with_label_counts[level] for level in taxonomic_levels],
            "#True_labels_available": [true_label_counts[level] for level in taxonomic_levels]
        }

        # print(f"{modelName}: {overall_acc['Genus']:.3f} (n={len(samples)})", end='')

        weightedAccuracy = getWeightedStatistics(overall_acc, taxonomic_levels, prediction_with_label_counts)
        weightedPrecision = getWeightedStatistics(overall_precision, taxonomic_levels, prediction_with_label_counts)
        weightedRecall = getWeightedStatistics(overall_recall, taxonomic_levels, prediction_with_label_counts)
        weightedF1 = getWeightedStatistics(overall_f1, taxonomic_levels, prediction_with_label_counts)
        weightedBinaryAccuracy = getWeightedStatistics(binary_acc, taxonomic_levels, prediction_with_label_counts)
        weightedBinaryPrecision = getWeightedStatistics(binary_precision, taxonomic_levels, prediction_with_label_counts)
        weightedBinaryRecall = getWeightedStatistics(binary_recall, taxonomic_levels, prediction_with_label_counts)
        weightedBinaryF1 = getWeightedStatistics(binary_f1, taxonomic_levels, prediction_with_label_counts)

        weightedRes = (weightedAccuracy, weightedPrecision, weightedRecall, weightedF1, weightedBinaryAccuracy, weightedBinaryPrecision, weightedBinaryRecall, weightedBinaryF1)

        return pandas.DataFrame(results_data), weightedRes
        
    
    # Define function to calculate macro-average accuracy
