import json
import pandas
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from config import config
config.updatePath()
from entity.taxoTree import taxoTree
from prototype.result import Result

def evaluate(dataset, index):
    config.majorDataset = dataset
    config.updatePath()
    predFileName = f"{config.resultBase}/result-{index}.json"
    stdFileName = f"{config.datasetBase}/answer.json"
    with open(predFileName) as fp:
        preds = json.load(fp)
    with open(stdFileName) as fp:
        truths = json.load(fp)
    df, _ = analyseStatistics(preds, truths)
    df.to_csv(f'{config.resultBase}/statistics-{index}.csv', index=False)


def analyseStatistics(preds, truths):
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
        for id, std in truths.items():
            if (std == 'no answer'):
                continue
            stdNode = taxoTree.getTaxoNodeFromICTV(ICTVName=std)
            pred = preds[id]
            if (pred == 'no result'):
                predictNode = None
            else:
                predictNode = taxoTree.getTaxoNodeFromICTV(ICTVName=pred)

            # currently we do not care about virus prediction
            # stdNode can still be None because there are nonVirus samples 
            if (stdNode is not None):
                for node in stdNode.ICTVNode.path:
                    true_labels[id][node.rank.capitalize()] = node.name

            if (predictNode is not None):
                for node in predictNode.ICTVNode.path:
                    predictions[id][node.rank.capitalize()] = node.name

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

def main():
    # evaluate('refseq_2024_test', 10)
    # evaluate('genbank_2024_test', 12)
    evaluate('Challenge', 11)

if (__name__ == "__main__"):
    main()