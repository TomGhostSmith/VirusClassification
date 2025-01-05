import json
import pandas
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from collections import defaultdict

from config import config
from entity.sample import Sample
from prototype.mergeModule import MergeModule
from entity.taxoTree import taxoTree
from entity.taxoNode import TaxoNode
from prototype.result import Result

class Evaluator():
    def __init__(self, models):
        if (isinstance(models, list)):
            self.models = models
        else:
            self.models = [models]
        
        for model in self.models:
            model.run()

        self.allSamples = self.loadSamples()

    # sampleRange could be: 'all', 'withResult', 'interested'
    def evaluate(self, sampleRange='all'):
        if sampleRange == 'interested':
            with open(f"{config.tempFolder}/interestedSamples.txt") as fp:
                interestedSamples = {l.strip() for l in fp}
            
            interestedSamples = list()
            for sample in self.allSamples:
                if sample.name in interestedSamples:
                    interestedSamples.append(sample)
        else:
            interestedSamples = self.allSamples

        for model in self.models:
            model.getResults(interestedSamples)

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
            df = self.analyseStatistics(focusedSamples, model.moduleName)
            # print(f" (all)")


            df.to_csv(f"{config.resultBase}/Statistics|n={totalCount};incl={includeCount};valid={validCount}|{model.moduleName}.csv")            

            # perform separate calculation for MergeModel
            if (isinstance(model, MergeModule)):
                partitions = dict()
                for sample in focusedSamples:
                    source = sample.info[model.moduleName]
                    if source not in partitions:
                        partitions[source] = list()
                    partitions[source].append(sample)
                

                for source, samples in partitions.items():
                    df = self.analyseStatistics(samples, model.moduleName)
                    # print(f" ({source})")

                    df.to_csv(f"{config.resultBase}/Statistics|n={totalCount};incl={includeCount};valid={validCount}|Source={source};m={len(samples)}|{model.moduleName}.csv")

    # sampleRange could be: 'all', 'intersection', 'interested'
    def compare(self, sampleRange='all'):
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


        for model in self.models:
            model.getResults(interestedSamples)

        
        if (sampleRange == 'intersection'):
            validSamples = list()
            for sample in interestedSamples:
                if (None not in sample.results.values()):
                    validSamples.append(sample)
            interestedSamples = validSamples

        totalCount = len(self.allSamples)
        includeCount = len(interestedSamples)
        with pandas.ExcelWriter(f"{config.resultBase}/Compare|n={totalCount};incl={includeCount}.xlsx", engine='xlsxwriter') as writer:
            for model in self.models:
                validCount = 0
                for sample in interestedSamples:
                    if (sample.results[model.moduleName] is not None):
                        validCount += 1


                # generate report for focused samples
                df = self.analyseStatistics(interestedSamples, model.moduleName)

                nameDict = {
                    "ML-param=contig_most_frequent_t33_512": "t33_512",
                    "ML-param=contig_most_frequent_family_finetune_t33_256": "family_ft_t33_256",
                    "minimap-ref=VMRv4;mode=ont": "minimap"
                }
                name = nameDict[model.moduleName]

                df.to_excel(writer, sheet_name=name, index=False)

    def loadSamples(self):
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

        # Initialize metrics storage
        acc_sum, precision_sum, recall_sum, f1_sum = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        overall_acc, overall_precision, overall_recall, overall_f1 = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
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

            for contig in true_labels.keys():
                true_label = true_labels[contig].get(level, 'Unknown')
                pred_label = predictions[contig].get(level, 'Unknown') if contig in predictions else 'Unknown'

                if pred_label != 'Unknown':
                    prediction_counts[level] += 1
                if pred_label != 'Unknown' and true_label != 'Unknown':
                    prediction_with_label_counts[level] += 1
                if true_label != 'Unknown':
                    true_label_counts[level] += 1

                if true_label != 'Unknown':
                    y_true.append(true_label)
                    y_pred.append(pred_label)

            if true_label_counts[level] != 0:
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
            else:
                acc_sum[level] = precision_sum[level] = recall_sum[level] = f1_sum[level] = '-'
                overall_acc[level] = overall_precision[level] = overall_recall[level] = overall_f1[level] = '-'

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
            "#Predictions": [prediction_counts[level] for level in taxonomic_levels],
            "#Predictions_with_labels": [prediction_with_label_counts[level] for level in taxonomic_levels],
            "#True_labels_available": [true_label_counts[level] for level in taxonomic_levels]
        }

        # print(f"{modelName}: {overall_acc['Genus']:.3f} (n={len(samples)})", end='')

        return pandas.DataFrame(results_data)
        
    
    # Define function to calculate macro-average accuracy
