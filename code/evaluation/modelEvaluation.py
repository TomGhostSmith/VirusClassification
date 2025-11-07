import os
# import sys
import json
import math
import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
# sys.path.append('./code')

from utils import IOUtils
from prototype.module import Module

from config import config

datasetRoot = "/Data/VirusClassification/dataset"

from utils.NucleotideUtils import NucleotideUtils

from entity.taxoTree import taxoTree
from tools.evaluate import analyseStatistics

import matplotlib.markers as mmarkers
from matplotlib.cm import get_cmap

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage


def testModel(models:dict[str, Module], dataset, evaluationMethod, subset='all', missingLabel="Unknown"):
    IOUtils.showInfo(f"Test {len(models)} models on {dataset}")
    queryFilePath = f"{datasetRoot}/{dataset}/{dataset}.fasta"
    if (subset != 'all'):
        querySubsetFilePath = f"{datasetRoot}/{dataset}/{subset}.txt"
    else:
        querySubsetFilePath = None
    modelRoot = config.modelRoot
    outputRoot = config.outputRoot
    config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

    
    samples = IOUtils.loadSamples(config.queryFilePath, config.querySubsetFilePath)

    for modelDesc, model in models.items():
        IOUtils.showInfo(f'getting {model.moduleName} results')
        model.getResults(samples)
    
    # if (subset is not 'all'):
    #     IOUtils.showInfo(f"dataset {dataset} subset {subset} finished. Skip the evaluation")
    #     return # currently we cannot evaluate a subset, because the answer is not complete
    IOUtils.showInfo('calculating statistics')
    # with open("/Data/VirusClassification/dataset/refseq_2024_test/answer.json") as fp:
    with open(f"{datasetRoot}/{dataset}/answer_{evaluationMethod}.json") as fp:
        stdResults = json.load(fp)

    for id, std in stdResults.items():
        if (std == 'no answer'):
            stdResults[id] = None
        else:
            stdResults[id] = taxoTree.getTaxoNodeFromICTV(ICTVName=std)

    GTs = dict()
    for sample in samples:
        sample.info["stdResult"] = stdResults.get(sample.id)
        GTs[sample.id] = stdResults.get(sample.id)

    

    summaryDict = dict()
    summaryDict["index"] = list()
    summaryDict["model"] = list()
    analysisDict = dict()
    analysisDict["model"] = list()
    analysisDict["statistics"] = list()
    analysisDict["order"] = list()
    analysisDict["family"] = list()
    analysisDict["genus"] = list()
    analysisDict["species"] = list()
    for r in config.resultRanks:
        summaryDict[r.capitalize()] = list()

    # check and get a valid name
    analysisIndex = 0
    fileName = f"{config.analysisFolder}/table/{dataset}_{subset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.analysisFolder}/table/{dataset}_{subset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"


    withSubRank = False

    rankLevels = list()
    for r in config.resultRanks:
        if (withSubRank or not r.startswith('sub')):
            rankLevels.append(r)
    fig, ax = plt.subplots(figsize=(2*(1 + len(rankLevels)), 6))
    x = numpy.arange(len(rankLevels))
    width = 2 / (len(models) + 2)

    modelRecalls = {r: list() for r in rankLevels}
    modelPrecisions = {r: list() for r in rankLevels}
    
    availableColors = numpy.vstack((plt.get_cmap('tab20').colors, plt.get_cmap('tab20b').colors, plt.get_cmap('tab20c').colors))
    # availableColors = numpy.vstack((availableColors[::2], availableColors[1::2]))

    with pandas.ExcelWriter(fileName) as writer:
    # for _ in range(1):
    #     writer = pandas.ExcelWriter(fileName)
        for idx, (modelDesc, model) in tqdm(list(enumerate(list(models.items()))), desc="evaluate", unit="model"):
            preds = dict()
            for sample in samples:
                res = sample.results[model.moduleName]
                if (res is not None):
                    preds[sample.id] = res.node
                else:
                    preds[sample.id] = None
            df, _ = analyseStatistics(preds, GTs, missingLabel)
            # df.to_csv(f"{config.analysisFolder}/kraken-refseq-performance.csv")
            # df.to_csv(f"{config.analysisFolder}/model-{idx}-{dataset}-performance_{evaluationMethod}.csv")
            df.to_excel(writer, sheet_name=f"model-{idx}", index=False)

            summaryDict["index"].append(idx)
            summaryDict["model"].append(model.moduleName)

            recallList = list()
            accuracyList = list()
            precisionList = list()

            totalPrecision = 0
            totalPrecisionCount = 0
            totalRecall = 0
            totalRecallCount = 0

            recalls = list()
            precisions = list()
            F1s = list()
            focusRanks = ["order", "family", "genus", "species"]
            for _, row in df.iterrows():
                bin_recall = f"{float(row.Recall_Binary):.3f}" if row.Recall_Binary != '-' else '-'
                bin_precision = f"{float(row.Precision_Binary):.3f}" if row.Precision_Binary != '-' else '-'
                overall_acc = f"{float(row.ACC_overall):.3f}" if row.ACC_overall != '-' else '-'
                summaryDict[row.Level].append(f"{bin_recall}/{bin_precision}/{overall_acc}")

                rec = row.Recall_Binary if row.Recall_Binary != '-' else 0
                acc = row.ACC_overall * rec if row.ACC_overall != '-' else 0
                prec = row.Precision_overall if row.Precision_overall != '-' else 0


                if (row.Level.lower() in rankLevels):
                    modelRecalls[row.Level.lower()].append(rec)
                    modelPrecisions[row.Level.lower()].append(prec)

                    recallList.append(rec)
                    accuracyList.append(acc)
                    precisionList.append(prec)
                    sampleCount = row["#True_labels_available"]
                    totalPrecision += prec * sampleCount
                    totalPrecisionCount += sampleCount
                    totalRecall += rec * sampleCount
                    totalRecallCount += sampleCount
                
                if (row.Level.lower() in focusRanks):
                    recalls.append(rec)
                    precisions.append(prec)
                    F1s.append(2*rec*prec / (rec+prec) if (rec+prec > 0) else 0)  # note: this is not a typical F1 score, but a coverage-precision F1 score

            # bars = ax.bar(x + idx*width, recallList, width, label=f"model {idx} recall", alpha=0.7)
            # bars = ax.bar(x*1.5 + idx*width, recallList, width, alpha=0.3, color=availableColors[idx])
            ax.bar(x*2 + idx*width, recallList, width, color='white', hatch='/', edgecolor=availableColors[idx])
            # ax.bar(x + idx*width, accuracyList, width, color=bars[0].get_facecolor(), label=f"model {idx} accuracy", hatch='//')
            # ax.bar(x*1.5 + idx*width, accuracyList, width, color='white', hatch='/', edgecolor=bars[0].get_facecolor(), label=f"{modelDesc}")
            # ax.bar(x*1.5 + idx*width, accuracyList, width, color=bars[0].get_facecolor(), alpha=0.9, label=f"{modelDesc}")
            ax.bar(x*2 + idx*width, accuracyList, width, color=availableColors[idx], alpha=1, label=f"{modelDesc}")

            # modelRecalls.append(totalRecall / totalRecallCount if totalRecallCount > 0 else 0)
            # modelPrecisions.append(totalPrecision / totalPrecisionCount if totalPrecisionCount > 0 else 0)

            analysisDict["model"].append(modelDesc)
            analysisDict["model"].append(modelDesc)
            analysisDict["model"].append(modelDesc)
            analysisDict["statistics"].append("coverage")
            analysisDict["statistics"].append("precision")
            analysisDict["statistics"].append("coverage-precision F1")
            for stat in [recalls, precisions, F1s]:
                for r, s in zip(focusRanks, stat):
                    analysisDict[r].append(s)

        
        pandas.DataFrame(summaryDict).to_excel(writer, sheet_name='summary', index=False)
        pandas.DataFrame(analysisDict).to_excel(writer, sheet_name='analysis', index=False)
        
    ax.set_xlabel("rank")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Performance of {len(models)} models on {dataset}. Missing label are set as {missingLabel}")
    ax.set_xticks(x*2 + width * (len(models) - 1)/2)
    ax.set_xticklabels(rankLevels)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if (withSubRank):
        fileNamePrefix = f"{config.analysisFolder}/figure/performance_withSub_{dataset}_{missingLabel}"
    else:
        fileNamePrefix = f"{config.analysisFolder}/figure/performance_{dataset}_{missingLabel}"

    i = 0
    fileName = f"{fileNamePrefix}_{i}.png"
    while (os.path.exists(fileName)):
        i += 1
        fileName = f"{fileNamePrefix}_{i}.png"

    plt.savefig(fileName, bbox_inches="tight")
    plt.close()


    # save scatter plot

    markers = [m for m in mmarkers.MarkerStyle.markers.keys() if isinstance(m, str) and m not in (".", ",", " ", "")][:22]
    markers = markers * (math.ceil(len(models)/ len(markers)))
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for idx, (modelDesc, recall, precision) in enumerate(zip(list(models.keys()), modelRecalls["order"], modelPrecisions["order"])):
        axs[0, 0].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100, label=modelDesc)
        axs[0, 0].set_title("order")
        axs[0, 0].grid(True)
        axs[0, 0].set_xlabel("Recall")
        axs[0, 0].set_ylabel("Precision")
    for idx, (recall, precision) in enumerate(zip(modelRecalls["family"], modelPrecisions["family"])):
        axs[0, 1].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100)
        axs[0, 1].set_title("family")
        axs[0, 1].grid(True)
        axs[0, 1].set_xlabel("Recall")
        axs[0, 1].set_ylabel("Precision")
    for idx, (recall, precision) in enumerate(zip(modelRecalls["genus"], modelPrecisions["genus"])):
        axs[1, 0].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100)
        axs[1, 0].set_title("genus")
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel("Recall")
        axs[1, 0].set_ylabel("Precision")
    for idx, (recall, precision) in enumerate(zip(modelRecalls["species"], modelPrecisions["species"])):
        axs[1, 1].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100)
        axs[1, 1].set_title("species")
        axs[1, 1].grid(True)
        axs[1, 1].set_xlabel("Recall")
        axs[1, 1].set_ylabel("Precision")
    
    

    fig.suptitle(f"Performance of {len(models)} models on {dataset}. Missing label are set as {missingLabel}")
    fig.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.xlim((int(min(modelRecalls) * 10 - 1))/10, 1)
    # plt.ylim(0.5, 1)

    if (withSubRank):
        fileNamePrefix = f"{config.analysisFolder}/figure/performance_withSub_{dataset}_{missingLabel}_scatter"
    else:
        fileNamePrefix = f"{config.analysisFolder}/figure/performance_{dataset}_{missingLabel}_scatter"

    i = 0
    fileName = f"{fileNamePrefix}_{i}.png"
    while (os.path.exists(fileName)):
        i += 1
        fileName = f"{fileNamePrefix}_{i}.png"

    fig.savefig(fileName, bbox_inches="tight")
    plt.close()

    

    IOUtils.showInfo('Done')
    return samples

def testModelVirusIdentity(models:dict[str, Module], dataset):
    queryFilePath = f"{datasetRoot}/{dataset}/{dataset}.fasta"
    querySubsetFilePath = None
    modelRoot = config.modelRoot
    outputRoot = config.outputRoot
    config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

    # from tools.evaluate import analyseStatistics
    samples = IOUtils.loadSamples(config.queryFilePath, config.querySubsetFilePath)

    for modelDesc, model in models.items():
        IOUtils.showInfo(f'getting {model.moduleName} results')
        model.getResults(samples)
    IOUtils.showInfo('calculating statistics')
    # with open(f"{datasetRoot}/{dataset}/answer_virusIdentify.json") as fp:
    #     stdResults = json.load(fp)

    stdResults = {sample.id: False for sample in samples}

    summaryDict = dict()
    summaryDict["index"] = list()
    summaryDict["model"] = list()
    # summaryDict["Recall"] = list()
    summaryDict["Precision"] = list()
    summaryDict["Accuracy"] = list()

    # check and get a valid name
    analysisIndex = 0
    fileName = f"{config.analysisFolder}/table/analysis_{dataset}_virusIdentify_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.analysisFolder}/table/analysis_{dataset}_virusIdentify_{analysisIndex}.xlsx"
    

    for idx, (modelDesc, model) in enumerate(list(models.items())):
        summaryDict["index"].append(idx)
        summaryDict["model"].append(modelDesc)
        TP, TN, FP, FN = 0, 0, 0, 0
        for sample in samples:
            res = sample.results[model.moduleName]
            if (res is not None and res.node is not None):
                if (stdResults[sample.id] == True):
                    TP += 1
                else:
                    FP += 1
            else:
                if (stdResults[sample.id] == True):
                    FN += 1
                else:
                    TN += 1
        # summaryDict["Recall"].append(TP / (TP + FN))
        summaryDict["Precision"].append(TP / (TP + FP))
        summaryDict["Accuracy"].append((TP + TN) / (TP + TN + FP + FN))

    with pandas.ExcelWriter(fileName) as writer:
        pandas.DataFrame(summaryDict).to_excel(writer, sheet_name='summary', index=False)
        

    IOUtils.showInfo('Done')

def sampleWiseAnalysis(models:dict[str, Module], dataset, evaluationMethod, subset='all', missingLabel="Unknown", analyseList:list[tuple[str, str]]=[]):
    additionalInfo = set()
    if (analyseList is not None):
        for pair in analyseList:
            if (len(pair) == 2):
                factor1, factor2 = pair
            elif (len(pair) == 3):
                factor1, factor2, _ = pair
            else:
                raise ValueError(f"Unknown analyse task: {pair}")
            if (isinstance(factor1, list)):
                if (len(factor1) != 2):
                    raise ValueError(f"Missing bins for customized feature {factor1[0]}")
                if (factor1[0] not in ["protein_length", "length", "protein_count"]):
                    additionalInfo.add(factor1[0])
            if (isinstance(factor2, list)):
                if (len(factor2) != 2):
                    raise ValueError(f"Missing bins for customized feature {factor2[0]}")
                if (factor2[0] not in ["protein_length", "length", "protein_count"]):
                    additionalInfo.add(factor2[0])
            
            if (isinstance(factor1, str) and factor1 not in models):
                raise ValueError(f"Unknown model name '{factor1}'")
            if (isinstance(factor2, str) and factor2 not in models):
                raise ValueError(f"Unknown model name '{factor2}'")
        
    samples = testModel(models, dataset, evaluationMethod, subset, missingLabel)

    NucleotideUtils.extractProtein(samples)
    
    # sheet 1: all information
    sampleIDs = list()
    sampleLengths = list()
    sampleProteinCounts = list()
    sampleProteinLengths = list()
    stdResults = list()
    stdResultRank = list()
    modelResults:dict[str, list[tuple[str, str, str, str]]] = {model.moduleName: list() for model in models.values()} # for each model, provide a list of (model_result, model_result_rank, LCA_rank)

    summaryDict = {
        "id": [],
        "length": [],
        "protein_count": [],
        "protein_length": [],
        "ground_truth": [],
        "ground_truth_rank": []
    }
    for k in additionalInfo:
        summaryDict[k] = []

    with open("/Data/VirusClassification/model/VMRv4_ML_train/taxonCount.json") as jsonFP:
        taxonDistribution = json.load(jsonFP)

    taxoCounts:dict[str, list[str]] = {rank: [] for rank in config.resultRanks}

    for sample in tqdm(samples, desc="sample wise result"):
        sampleIDs.append(sample.id)
        sampleLengths.append(sample.length)
        sampleProteinCounts.append(len(sample.proteins))
        sampleProteinLengths.append(sum(p.length for p in sample.proteins))
        std = sample.info["stdResult"]
        for k in additionalInfo:
            summaryDict[k].append(sample.info.get(k))

        if (std is not None and std.ICTVNode is not None):
            stdNode = std.ICTVNode
            stdResults.append(stdNode.name)
            stdResultRank.append(stdNode.rank)
        else:
            stdNode = None
            stdResults.append('N/A')
            stdResultRank.append('N/A')

        if (stdNode is not None):
            rankC = {}
            for n in stdNode.path:
                if n.rank in taxonDistribution:
                    rankC[n.rank] = taxonDistribution[n.rank].get(n.name, 0)
            for r in config.resultRanks:
                taxoCounts[r].append(rankC.get(r, 0))
        else:
            for r in config.resultRanks:
                taxoCounts[r].append(0)


        for modelDesc, model in models.items():
            pred = sample.results[model.moduleName]
            if (pred is not None and pred.node is not None):
                pred = pred.node.ICTVNode
                if (stdNode is not None):
                    LCANode = taxoTree.ICTVTree.findLCA([pred, stdNode])
                    modelResults[model.moduleName].append((pred.name, pred.rank, LCANode.name, LCANode.rank))
                else:
                    modelResults[model.moduleName].append((pred.name, pred.rank, "N/A", "N/A"))
            else:
                modelResults[model.moduleName].append(("N/A", "N/A", "N/A", "N/A"))


    summaryDict["id"] = sampleIDs
    summaryDict["length"] = sampleLengths
    summaryDict["protein_count"] = sampleProteinCounts
    summaryDict["protein_length"] = sampleProteinLengths
    summaryDict["ground_truth"] = stdResults
    summaryDict["ground_truth_rank"] = stdResultRank

    for r, taxonCountList in taxoCounts.items():
        summaryDict[f"{r}_count_in_train"] = taxonCountList

    for modelDesc, model in models.items():
        preds, predRanks, LCAs, LCARanks = zip(*modelResults[model.moduleName])
        summaryDict[f"{modelDesc}_pred"] = preds
        summaryDict[f"{modelDesc}_pred_rank"] = predRanks
        summaryDict[f"{modelDesc}_LCA"] = LCAs
        summaryDict[f"{modelDesc}_LCA_rank"] = LCARanks

    # for (k, v) in summaryDict.items():
    #     IOUtils.showInfo(f"{k}: {len(v)}")
    
    summaryDF = pandas.DataFrame(summaryDict)
    analysisIndex = 0
    fileName = f"{config.analysisFolder}/samplewise/samples_{dataset}_{subset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.analysisFolder}/samplewise/samples_{dataset}_{subset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    
    with pandas.ExcelWriter(fileName) as writer:
        summaryDF.to_excel(writer, sheet_name="raw results", index=False)

    analysis(models, analyseList, summaryDF, dataset, subset, f"sampleAnalysis_{dataset}_{subset}_{evaluationMethod}_{missingLabel}")


# input: 
# - analyseList: a list of analyse tasks: [factor1, factor2, constraint=None]
#   - factor 1 can be a field with bins, or a model name
#   - factor 2 can be a field with bins, or a model name, or a set of model names
#   - constraint can be a function f(row)->bool to focus on specific samples
# - summaryDF is the data frame with all information needed
# - outputname is the file name for output. If exists, it will add number behind it
def analysis(modelDict, analyseList, summaryDF, dataset, subset, outputName="analysis"):
    for field in ["id", "length", "protein_count", "protein_length", "ground_truth"]:
        if field not in summaryDF:
            raise ValueError(f"Required field {field} is missing")
        
    models = set()
    fields = set()
    analyseTasks = []
    

    modelRankResults = getModelRankResults(summaryDF, modelDict.keys())
    # sheet 2: performance related analysis
    DFs:dict[str, pandas.DataFrame] = dict()  # desc: DF
    DFs["raw results"] = summaryDF
    imgs:dict[str, str] = dict()

    # draw the plot
    def getMetrics(series):
        correct = (series == "correct").sum()
        error = (series == "wrong").sum()
        noPred = (series == "No_pred has_GT").sum()
        hasGT = correct + error + noPred
        hasPred = (series == "has_pred No_GT").sum()
        leave = (series == "No_pred No_GT").sum()
        noGT = hasPred + leave
        return pandas.Series({
            "recall": (correct + error) / hasGT if hasGT > 0 else 0, 
            "correct": correct / hasGT if hasGT > 0 else 0, 
            "overPredict": -hasPred / noGT if noGT > 0 else 0
            })
    
    rankLevels = list()
    for r in config.resultRanks:
        if (not r.startswith('sub')):
            rankLevels.append(r)
    availableColors = numpy.vstack((plt.get_cmap('tab20').colors, plt.get_cmap('tab20b').colors, plt.get_cmap('tab20c').colors))
    fig, ax = plt.subplots(figsize=(2*(1 + len(rankLevels)), 12))
    x = numpy.arange(len(rankLevels))
    width = 2 / (len(modelDict) + 2)
    pattern_handles = []
    for idx, model in enumerate(modelDict.keys()):
        tmpDF = pandas.DataFrame(modelRankResults[model])
        tmpDF = tmpDF[rankLevels]
        metrics = tmpDF.apply(getMetrics)
        recalls = metrics.loc["recall"].values
        corrects = metrics.loc["correct"].values
        overPredicts = metrics.loc["overPredict"].values
        ax.bar(x*2 + idx*width, recalls, width, color='white', hatch='//', edgecolor=availableColors[idx])
        ax.bar(x*2 + idx*width, corrects, width, color=availableColors[idx], alpha=1)
        ax.bar(x*2 + idx*width, overPredicts, width, color='white', hatch='....', edgecolor=availableColors[idx])
        pattern_handles.append(Patch(facecolor=availableColors[idx], label=model))

    pattern_handles += [
        Line2D([0], [0], color='white', linewidth=1),
        Patch(facecolor='gray', edgecolor='gray', label='Correct prediction'),
        Patch(facecolor='white', hatch='////', edgecolor='gray', label='Wrong prediction'),
        Patch(facecolor='white', hatch="....", edgecolor='gray', label='Unlabelled prediction')
    ]
    # ax.legend(handles=pattern_handles, title="Prediction Type", loc='upper right')

    ax.set_xticks(x*2 + width * (len(modelDict) - 1)/2)
    ax.set_xticklabels(rankLevels)
    ax.legend(handles=pattern_handles, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("rank")
    ax.axhline(0, color="black", linewidth=1)
    # ax.set_ylabel("Proportion")


    yticks = numpy.arange(-1, 1.1, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y):.2f}" for y in yticks])

    ax.annotate(
        "", xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
        xytext=(0, -0.05), textcoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2)
    )

    ax.text(-2, 0.5, "samples w/ ground truth", ha="left", va="center", fontsize=10, rotation=90)
    ax.text(-2, -0.5, "samples w/o ground truth", ha="left", va="center", fontsize=10, rotation=90)


    fileNamePrefix = f"{config.analysisFolder}/figure/newPerformance_{dataset}_{subset}"

    i = 0
    fileName = f"{fileNamePrefix}_{i}.png"
    while (os.path.exists(fileName)):
        i += 1
        fileName = f"{fileNamePrefix}_{i}.png"

    plt.savefig(fileName, bbox_inches="tight")
    plt.close()

    if (not analyseList):
        IOUtils.showInfo("Nothing to analyse. Exit")
        return

    for pair in analyseList:
        # check #member of pair
        if (len(pair) == 2):
            factor1, factor2 = pair
            constraint = None
        elif (len(pair) == 3):
            factor1, factor2, constraint = pair
            analyseTasks.append(factor1, factor2, constraint)
        else:
            IOUtils.showInfo(f"Unknown analysis combination: {pair}")
            continue

        # check if factor1 is valid
        if (isinstance(factor1, list)):
            if (len(factor1) != 2):
                IOUtils.showInfo(f"Missing bins for customized feature {factor1[0]}", "ERROR")
                continue
            if (factor1[0] not in summaryDF):
                IOUtils.showInfo(f"Field {factor1[0]} is missing", "ERROR")
                continue
            fields.add(factor1[0])
        elif (isinstance(factor1, str) and f"{factor1}_pred" in summaryDF):
            models.add(factor1)
        else:
            IOUtils.showInfo(f"Unknown factor: {factor1}")
            continue

        # check if factor2 is valid
        if (isinstance(factor2, list)):
            if (len(factor2) != 2):
                IOUtils.showInfo(f"Missing bins for customized feature {factor2[0]}", "ERROR")
                continue
            if (factor2[0] not in summaryDF):
                IOUtils.showInfo(f"Field {factor2[0]} is missing", "ERROR")
                continue
            fields.add(factor2[0])
        elif (isinstance(factor2, set)):
            missing = False
            for model in factor2:
                if f"{factor2}_pred" not in summaryDF:
                    IOUtils.showInfo(f"model {model} is missing", "ERROR")
                    missing = True
            if (missing):
                continue
        elif (isinstance(factor2, str) and f"{factor2}_pred" in summaryDF):
            models.add(factor2)
        else:
            IOUtils.showInfo(f"Unknown factor: {factor2}")
            continue

        analyseTasks.append((factor1, factor2, constraint))
    
    models = list(models)

    for model in models:
        if (model not in modelDict):
            IOUtils.showInfo(f"Unknown model: {model}")


    for factor1, factor2, constraint in tqdm(analyseTasks, desc="sample wise analysis"):
        if (isinstance(factor1, list) and isinstance(factor2, list)): 
            factor2factorConfusion(summaryDF, factor1, factor2, constraint, DFs, imgs)
        elif (isinstance(factor1, str) and isinstance(factor2, list)):
            model2factorConfusion(summaryDF, modelRankResults, factor1, factor2, constraint, DFs, imgs)
        elif (isinstance(factor1, str) and isinstance(factor2, set)):
            model2modelsConfusion(summaryDF, modelRankResults, factor1, factor2, constraint, DFs, imgs)
        elif (isinstance(factor1, str) and isinstance(factor2, str)):
            model2modelConfusion(summaryDF, modelRankResults, factor1, factor2, constraint, DFs, imgs)
        else:
            IOUtils.showInfo(f"Unknown analysis combination: '{factor1}' vs '{factor2}' under constraint {constraint}")
    
    analysisIndex = 0
    fileName = f"{config.analysisFolder}/samplewise/{outputName}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.analysisFolder}/samplewise/{outputName}_{analysisIndex}.xlsx"
    
    with pandas.ExcelWriter(fileName) as writer:
        sheetNames = dict()
        contentDF = {
            "sheet": list(range(1, len(DFs))),
            "content": list(DFs.keys())[1:]
        }
        pandas.DataFrame(contentDF).to_excel(writer, sheet_name="contents", index=False)
        for idx, (desc, df) in enumerate(DFs.items()):
            if (desc == "raw results"):
                df.to_excel(writer, sheet_name='raw results', index=False)
            else:
                sheetNames[desc] = f"sheet {idx}"
                df.to_excel(writer, sheet_name=f"sheet {idx}", index=(not desc.endswith("confusion matrics")))
        
    # insert images
    wb = load_workbook(fileName)

    for sheet, img in imgs.items():
        ws = wb[sheetNames[sheet]]
        ws.add_image(XLImage(img), 'S1')

    wb.save(fileName)

    IOUtils.showInfo("Analyse finished")

def getModelRankResults(summaryDF:pandas.DataFrame, models):
    modelRankResults:dict[str, dict[str, list[str]]] = {modelDesc: {r: [] for r in config.evaluationRanks} for modelDesc in models}
    for _, row in summaryDF.iterrows():
        # get ground truth labels
        stdRanks = {r: None for r in config.evaluationRanks}
        std = taxoTree.ICTVTree.nodes.get(row["ground_truth"])
        if (std is not None):
            for n in std.path:
                stdRanks[n.rank] = n.name  # non-standard rank will be discarded later
        
        # get rank result for each model
        for modelDesc in models:
            predRanks = {r: None for r in config.evaluationRanks}
            pred = taxoTree.ICTVTree.nodes.get(row[f"{modelDesc}_pred"])
            if pred is not None:
                for n in pred.path:
                    predRanks[n.rank] = n.name

            for r in config.evaluationRanks:
                if (stdRanks[r] is None):
                    if (predRanks[r] is None):
                        modelRankResults[modelDesc][r].append("No_pred No_GT")
                    else:
                        modelRankResults[modelDesc][r].append("has_pred No_GT")
                else:
                    if (predRanks[r] is None):
                        modelRankResults[modelDesc][r].append("No_pred has_GT")
                    elif (predRanks[r] == stdRanks[r]):
                        modelRankResults[modelDesc][r].append("correct")
                    else:
                        modelRankResults[modelDesc][r].append("wrong")
    return modelRankResults


def model2modelConfusion(summaryDF:pandas.DataFrame, modelRankResults, factor1, factor2, constraint, DFs, imgs):
    modelDesc1 = factor1
    modelDesc2 = factor2

    modelLCAs = []
    modelGTLCAs = []
    LCA1s = []
    LCA2s = []
    validIndexes = []
    for idx, row in summaryDF.iterrows():
        if (constraint and not constraint(row)):
            continue
        validIndexes.append(idx)
        std = taxoTree.ICTVTree.nodes.get(row["ground_truth"])
        if (std is None):  # currently we only focus those with GT
            LCA1s.append("N/A")
            LCA2s.append("N/A")
            modelLCAs.append("No GT")
            modelGTLCAs.append("No GT")
            continue
        pred1 = taxoTree.ICTVTree.nodes.get(row[f"{modelDesc1}_pred"])
        pred2 = taxoTree.ICTVTree.nodes.get(row[f"{modelDesc2}_pred"])

        if (pred1 is not None and pred2 is not None):
            LCANode1 = taxoTree.ICTVTree.findLCA([pred1, std])
            LCANode2 = taxoTree.ICTVTree.findLCA([pred2, std])
            LCANode3 = taxoTree.ICTVTree.findLCA([pred1, pred2])
            LCANode4 = taxoTree.ICTVTree.findLCA([pred1, pred2, std])
            LCA1s.append(LCANode1.rank)
            LCA2s.append(LCANode2.rank)
            modelLCAs.append(LCANode3.rank)
            modelGTLCAs.append(LCANode4.rank)
        elif (pred1 is not None):
            LCANode1 = taxoTree.ICTVTree.findLCA([pred1, std])
            LCA1s.append(LCANode1.rank)
            LCA2s.append("N/A")
            modelLCAs.append(f"{modelDesc1}_only")
            modelGTLCAs.append(f"{modelDesc1}_only")
        elif (pred2 is not None):
            LCANode2 = taxoTree.ICTVTree.findLCA([pred2, std])
            LCA1s.append("N/A")
            LCA2s.append(LCANode2.rank)
            modelLCAs.append(f"{modelDesc2}_only")
            modelGTLCAs.append(f"{modelDesc2}_only")
        else:
            LCA1s.append("N/A")
            LCA2s.append("N/A")
            modelLCAs.append("N/A")
            modelGTLCAs.append("N/A")
    
    # df1: cross table of two models
    tmpDF = pandas.DataFrame({
        modelDesc1: LCA1s,
        modelDesc2: LCA2s
        }
    )
    cellText = f"{modelDesc1} \\ {modelDesc2}"
    tmpDF[cellText] = pandas.Categorical(tmpDF[modelDesc1], categories=["N/A"] + config.evaluationRanks, ordered=True)
    tmpDF["tmp2"] = pandas.Categorical(tmpDF[modelDesc2], categories=["N/A"] + config.evaluationRanks, ordered=True)
    analyseDF = pandas.crosstab(tmpDF[cellText], tmpDF[f"tmp2"], dropna=False)
    DFs[f'{modelDesc1} vs {modelDesc2}: difference'] = analyseDF
    
    # df2: similarity table of two models
    tmpDF = pandas.DataFrame({
        "modelLCA": modelLCAs,
        "modelGTLCA": modelGTLCAs
        }
    )
    cellText = "model_LCA \\ model_GT_LCA"
    tmpDF[cellText] = pandas.Categorical(tmpDF["modelLCA"], categories=["N/A"] + config.evaluationRanks + ["No GT", f"{modelDesc1}_only", f"{modelDesc2}_only"], ordered=True)
    tmpDF["tmp2"] = pandas.Categorical(tmpDF["modelGTLCA"], categories=["N/A"] + config.evaluationRanks + ["No GT", f"{modelDesc1}_only", f"{modelDesc2}_only"], ordered=True)
    analyseDF = pandas.crosstab(tmpDF[cellText], tmpDF[f"tmp2"], dropna=False)
    DFs[f'{modelDesc1} vs {modelDesc2}: similarity'] = analyseDF

    # df3: confusion matrics
    tmpDF = {}
    stackDFs = list()
    for r in config.evaluationRanks:
        r1 = modelRankResults[modelDesc1][r]
        r2 = modelRankResults[modelDesc2][r]
        tmpDF[f"model1_{r}"] = [r1[i] for i in validIndexes]
        tmpDF[f"model2_{r}"] = [r2[i] for i in validIndexes]
    tmpDF = pandas.DataFrame(tmpDF)
    
    cellText = f"{modelDesc1} \\ {modelDesc2}"
    categories = ["correct", "wrong", "No_pred has_GT", "has_pred No_GT", "No_pred No_GT"]
    for r in config.evaluationRanks:
        stackDFs.append(pandas.DataFrame([[r, "", "", "", "", ""]], columns=['confusion matrics'] + categories))
        stackDFs.append(pandas.DataFrame([[cellText] + categories], columns=['confusion matrics'] + categories))
        tmpDF[cellText] = pandas.Categorical(tmpDF[f"model1_{r}"], categories=categories, ordered=True)
        tmpDF["tmp"] = pandas.Categorical(tmpDF[f"model2_{r}"], categories=categories, ordered=True)
        rankDF = pandas.crosstab(tmpDF[cellText], tmpDF[f"tmp"], dropna=False)
        rankDF["confusion matrics"] = categories
        stackDFs.append(rankDF)

    stackDF = pandas.concat(stackDFs)
    DFs[f'{modelDesc1} vs {modelDesc2}: confusion matrics'] = stackDF

def model2modelsConfusion(summaryDF:pandas.DataFrame, modelRankResults, factor1, factor2, constraint, DFs, imgs):
    # one model compared to multi model, confusion matrix only
    modelDesc1 = factor1
    factor2 = list(factor2)
    tmpDF:dict[str, list] = {}
    stackDFs = []

    validIndexes = []
    for idx, row in summaryDF.iterrows():
        if (constraint and not constraint(row)):
            continue
        validIndexes.append(idx)


    for r in config.evaluationRanks:
        r1 = modelRankResults[modelDesc1][r]
        tmpDF[f"model1_{r}"] = [r1[i] for i in validIndexes]

        modelDesc = factor2[0]
        r2 = modelRankResults[modelDesc][r]
        tmpDF[f"model2_{r}"] = numpy.array([r2[i] for i in validIndexes])

        for modelDesc in factor2[1:]:
            r2 = modelRankResults[modelDesc][r]
            tmpDF[f"model2_{r}"] = numpy.where(numpy.array([r2[i] for i in validIndexes]) == tmpDF[f"model2_{r}"], tmpDF[f"model2_{r}"], "Other")
    tmpDF = pandas.DataFrame(tmpDF)
    
    cellText = f"{modelDesc1} \\ {','.join(factor2)}"
    categories = ["correct", "wrong", "No_pred has_GT", "has_pred No_GT", "No_pred No_GT"]
    for r in config.evaluationRanks:
        stackDFs.append(pandas.DataFrame([[r, "", "", "", "", "", ""]], columns=['confusion matrics'] + categories + ["Other"]))
        stackDFs.append(pandas.DataFrame([[cellText] + categories + ["Other"]], columns=['confusion matrics'] + categories + ["Other"]))
        tmpDF[cellText] = pandas.Categorical(tmpDF[f"model1_{r}"], categories=categories, ordered=True)
        tmpDF["tmp"] = pandas.Categorical(tmpDF[f"model2_{r}"], categories=categories + ["Other"], ordered=True)
        rankDF = pandas.crosstab(tmpDF[cellText], tmpDF[f"tmp"], dropna=False)
        rankDF["confusion matrics"] = categories
        stackDFs.append(rankDF)

    stackDF = pandas.concat(stackDFs)
    DFs[f'{modelDesc1} vs {",".join(factor2)}: confusion matrics'] = stackDF

def model2factorConfusion(summaryDF:pandas.DataFrame, modelRankResults, factor1, factor2, constraint, DFs, imgs):
    bins = factor2[1]
    factor2 = factor2[0]
    modelDesc = factor1
    if (f"{modelDesc} vs {factor2}" in DFs):
        dfIdx = 1
        dfName = f"{modelDesc} vs {factor2} {dfIdx}"
        while (dfName in DFs):
            dfIdx += 1
            dfName = f"{modelDesc} vs {factor2} {dfIdx}"
    else:
        dfIdx = 0
        dfName = f"{modelDesc} vs {factor2}"

    # need to manually bin the factor 2
    if (factor2 in ["length", "protein_length"]):
        bin2 = [-1] + list(range(0, 100000, 100))
        gap = 50
        # bins = numpy.linspace(summaryDF[factor2].min(), summaryDF[factor2].max(), 21)
    else:
        bin2 = list(range(-1, 200))
        gap = 10


    LCAs = []
    factors = []
    validIndexes = []
    for idx, row in summaryDF.iterrows():
        if (constraint and not constraint(row)):
            continue
        validIndexes.append(idx)
        factors.append(row[factor2])
        std = taxoTree.ICTVTree.nodes.get(row["ground_truth"])
        if (std is None):  # currently we only focus those with GT
            LCAs.append("N/A")
            continue
        pred1 = taxoTree.ICTVTree.nodes.get(row[f"{modelDesc}_pred"])

        if (pred1 is not None):
            LCANode = taxoTree.ICTVTree.findLCA([pred1, std])
            LCAs.append(LCANode.rank)
        else:
            LCAs.append("N/A")

    # df1: LCA vs factor2
    sumDF = pandas.DataFrame({
        "LCA": LCAs,
        "factors": factors
    })

    f2 = pandas.cut(sumDF["factors"], bins=bins, include_lowest=True)
    if f2.isna().any():
        f2 = f2.cat.add_categories("N/A").fillna("N/A")
    sumDF[f"factors_"] = f2
    sumDF['A_bin2'] = pandas.cut(sumDF["factors"], bins=bin2, include_lowest=True)
    sumDF["tmp"] = pandas.Categorical(sumDF["LCA"], categories=["N/A"] + config.evaluationRanks, ordered=True)
    analyseDF = pandas.crosstab(sumDF["factors_"], sumDF["tmp"], dropna=False)
    DFs[dfName] = analyseDF

    # img for df1
    ct = pandas.crosstab(sumDF['A_bin2'], sumDF["tmp"], dropna=False)
    ct_percent = ct.div(ct.sum(axis=1), axis=0).fillna(0)
    x = [str(interval) for interval in ct_percent.index]
    ys = ct_percent.T.values
    n_cate = len(config.evaluationRanks) + 1
    cmap = get_cmap('plasma')
    colors = [cmap(i / n_cate) for i in range(n_cate)]

    if (dfIdx > 0):  # only draw the image for the first analyse DF, if there are multiple
        plt.figure(figsize=(20, 6))
        plt.stackplot(x, ys, labels=["N/A"] + config.evaluationRanks, colors=colors)
        plt.xticks(ticks = range(0, len(x), gap), labels=[x[i] for i in range(0, len(x), gap)], rotation=90)
        plt.legend(loc='upper right')
        plt.title(f'{modelDesc} vs {factor2}')
        plt.xlabel(factor2)
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.savefig(f"{config.analysisFolder}/figure/{modelDesc} vs {factor2}.png")
        imgs[dfName] = f"{config.analysisFolder}/figure/{modelDesc} vs {factor2}.png"

    # df2: shows detailed statistics (correct, error, etc.) vs factor 2 for each rank
    stackDFs = list()
    categories = ["correct", "wrong", "No_pred has_GT", "has_pred No_GT", "No_pred No_GT"]
    for r in config.evaluationRanks:
        tmpDF = pandas.DataFrame({
            "tmp": modelRankResults[modelDesc][r]
        })
        tmpDF["tmp"] = pandas.Categorical(tmpDF["tmp"], categories=categories, ordered=True)
        tmpDF[factor2] = sumDF["factors_"]
        stackDFs.append(pandas.DataFrame([[r, "", "", "", "", ""]], columns=[factor2] + categories))
        stackDFs.append(pandas.DataFrame([[factor2] + categories], columns=[factor2] + categories))
        cf = pandas.crosstab(tmpDF[factor2], tmpDF["tmp"], dropna=False)
        cf[factor2] = sumDF["factors_"].cat.categories
        stackDFs.append(cf)
    stackDF = pandas.concat(stackDFs)
    DFs[f"{dfName} per rank confusion matrics"] = stackDF

    # df3: shows detailed statistics vs rank for each bin in factor 2
    rankRes = {}
    for r in config.evaluationRanks:
        rankRes[r] = [modelRankResults[modelDesc][r][i] for i in validIndexes]
    tmpDF = pandas.DataFrame({
        factor2: numpy.tile(sumDF[f"factors_"], len(config.evaluationRanks)),
        "rank": numpy.concatenate([numpy.tile(x, len(validIndexes)) for x in config.evaluationRanks]),
        "res": numpy.concatenate([rankRes[r] for r in config.evaluationRanks])
    })

    groups = tmpDF.groupby(factor2)
    stackDFs = list()
    for group, gdf in groups:
        gdf["rank"] = pandas.Categorical(gdf["rank"], categories=config.evaluationRanks, ordered=True)
        gdf["res"] = pandas.Categorical(gdf["res"], categories=categories, ordered=True)
        stackDFs.append(pandas.DataFrame([[group, "", "", "", "", ""]], columns=["rank"] + categories))
        stackDFs.append(pandas.DataFrame([["rank"] + categories], columns=["rank"] + categories))
        cf = pandas.crosstab(gdf["rank"], gdf["res"], dropna=False)
        cf["rank"] = config.evaluationRanks
        stackDFs.append(cf)
    stackDF = pandas.concat(stackDFs)
    DFs[f"{dfName} per bin confusion matrics"] = stackDF

    # for c in summaryDF[f"{factor2}_"].cat.categories:

def factor2factorConfusion(summaryDF:pandas.DataFrame, factor1, factor2, constraint, DFs, imgs):
    if (constraint):
        summaryDF = summaryDF[summaryDF.apply(constraint, axis=1)]

    bins = factor1[1]
    factor1 = factor1[0]
    bins2 = factor2[1]
    factor2 = factor2[0]
    cellText = f"{factor1} \\ {factor2}"
    summaryDF[cellText] = pandas.cut(summaryDF[factor1], bins=bins, include_lowest=True)

    summaryDF[f"{factor2}_"] = pandas.cut(summaryDF[factor2], bins=bins2, include_lowest=True)
    analyseDF = pandas.crosstab(summaryDF[cellText], summaryDF[f"{factor2}_"], dropna=False)

    DFs[f"{factor1} vs {factor2}"] = analyseDF


def mergeCachedResults():
    comingResultFolder = f"{config.cacheFolder}/resultsFromServer"
    files = os.listdir(comingResultFolder)
    for file in files:
        if (not file.endswith(".json")):
            IOUtils.showInfo(f"skiped {file}")
        else:
            filePath = f"{comingResultFolder}/{file}"
            originPath = f"{config.cacheResultFolder}/{file}"
            with open(filePath) as fp:
                newResults = json.load(fp)

            # check if the results are string, not a list
            validFile = True
            nonSimpleResultModels = ["blast", "minimap", "genomad", "kraken"]
            for m in nonSimpleResultModels:
                if isinstance(v, list):
                    validFile = False
                    break

            # validFile = "minimap" not in file and "blast" not in file
            
            if (not validFile):
                IOUtils.showInfo(f"Skip file {file} becase it seems not a standard result json")
                continue
            
            results = dict()
            if (os.path.exists(originPath)):
                with open(originPath) as fp:
                    results = json.load(fp)
            
            for k, v in newResults.items():
                if k in results:
                    if (results[k] != v):
                        IOUtils.showInfo(f"In file {file}, result for {k} are inconsistent: expect {results[k]}, but get {v}")
                        validFile = False
                else:
                    results[k] = v
            
            if (not validFile):
                IOUtils.showInfo(f"File {file} has some consistent results. Not updateed.")
                continue
                
            with open(originPath, 'wt') as fp:
                json.dump(results, fp, indent=2)
            
            IOUtils.showInfo(f"Updated {file}")
            os.remove(filePath)

def reAnalysis(models, dataset, evaluationMethod, subset='all', missingLabel="Unknown", idx=0, analyseList=[], rawResult=False):
    prefix = "samples" if rawResult else "sampleAnalysis" 
    fileName = f"{config.analysisFolder}/samplewise/{prefix}_{dataset}_{subset}_{evaluationMethod}_{missingLabel}_{idx}.xlsx"
    df = pandas.read_excel(fileName, sheet_name="raw results", keep_default_na=False)
    analysis(models, analyseList, df, dataset, subset, f"sampleAnalysis_{dataset}_{subset}_{evaluationMethod}_{missingLabel}")