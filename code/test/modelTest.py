import os
import sys
import json
import numpy
import pandas
import matplotlib.pyplot as plt
sys.path.append('./code')

from utils import IOUtils
from prototype.module import Module

from config import config

modelRoot = "/Data/VirusClassification/model"
outputRoot = "/Data/VirusClassification"
queryFilePath = ""
querySubsetFilePath = None
config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

from utils.NucleotideUtils import NucleotideUtils

from module.pipeline import Pipeline
from module.virusPredModule import VirusPred
from module.minimapThresholdModule import MinimapThresholdModule
from module.esm import ESM
from module.minimapMLMergeModule import MinimapMLMergeModule
from module.minimapThreshRankModule import MinimapThreshRankModule
from module.mlModule import MLModule
from module.vcontact import Vcontact

from module.phagcn import PhaGCN
from module.genomad import Genomad
from module.blast import Blast
from module.kraken import Kraken
from module.minimap import Minimap
from module.vitap import VITAP
from module.cat import CAT
from module.mergeModule import MergeModule
from module.metabuli import Metabuli
from module.diamond import Diamond

from moduleResult.plainResult import PlainResult
from moduleResult.mlResult import MLResult
from moduleResult.diamondAlignment import DiamondAlignment

from entity.taxoTree import taxoTree
from entity.sample import Sample

import matplotlib.markers as mmarkers


def testModel(models:dict[str, Module], dataset, evaluationMethod, subset=None, missingLabel="Unknown"):
    IOUtils.showInfo(f"Test {len(models)} models on {dataset}")
    queryFilePath = f"/Data/VirusClassification/dataset/{dataset}/{dataset}.fasta"
    if (subset is not None):
        querySubsetFilePath = f"/Data/VirusClassification/dataset/{dataset}/{subset}.txt"
    else:
        querySubsetFilePath = None
    config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

    from tools.evaluate import analyseStatistics
    samples = IOUtils.loadSamples(config.queryFilePath, config.querySubsetFilePath)

    for modelDesc, model in models.items():
        IOUtils.showInfo(f'getting {model.moduleName} results')
        model.getResults(samples)
    
    if (subset is not None):
        IOUtils.showInfo(f"dataset {dataset} subset {subset} finished. Skip the evaluation")
        return # currently we cannot evaluate a subset, because the answer is not complete
    IOUtils.showInfo('calculating statistics')
    # with open("/Data/VirusClassification/dataset/refseq_2024_test/answer.json") as fp:
    with open(f"/Data/VirusClassification/dataset/{dataset}/answer_{evaluationMethod}.json") as fp:
        stdResults = json.load(fp)

    for id, std in stdResults.items():
        if (std == 'no answer'):
            stdResults[id] = None
        else:
            stdResults[id] = taxoTree.getTaxoNodeFromICTV(ICTVName=std)

    for sample in samples:
        sample.info["stdResult"] = stdResults.get(sample.id)
    

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
    fileName = f"{config.cacheAnalysisFolder}/analysis_{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.cacheAnalysisFolder}/analysis_{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"


    withSubRank = False

    rankLevels = list()
    for r in config.resultRanks:
        if (withSubRank or not r.startswith('sub')):
            rankLevels.append(r)
    fig, ax = plt.subplots(figsize=(1.5*(1 + len(rankLevels)), 6))
    x = numpy.arange(len(rankLevels))
    width = 1.5 / (len(models) + 2)

    modelRecalls = {r: list() for r in rankLevels}
    modelPrecisions = {r: list() for r in rankLevels}
    
    availableColors = plt.get_cmap('tab20').colors

    with pandas.ExcelWriter(fileName) as writer:
    # for _ in range(1):
    #     writer = pandas.ExcelWriter(fileName)
        for idx, (modelDesc, model) in enumerate(list(models.items())):
            preds = dict()
            for sample in samples:
                res = sample.results[model.moduleName]
                if (res is not None):
                    preds[sample.id] = res.node
                else:
                    preds[sample.id] = None
            df, _ = analyseStatistics(preds, stdResults, missingLabel)
            # df.to_csv(f"{config.cacheAnalysisFolder}/kraken-refseq-performance.csv")
            # df.to_csv(f"{config.cacheAnalysisFolder}/model-{idx}-{dataset}-performance_{evaluationMethod}.csv")
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
            bars = ax.bar(x*1.5 + idx*width, recallList, width, alpha=0.3, color=availableColors[idx])
            # ax.bar(x + idx*width, accuracyList, width, color=bars[0].get_facecolor(), label=f"model {idx} accuracy", hatch='//')
            # ax.bar(x*1.5 + idx*width, accuracyList, width, color=bars[0].get_facecolor(), hatch='/', edgecolor='black')
            ax.bar(x*1.5 + idx*width, accuracyList, width, color=bars[0].get_facecolor(), alpha=0.9, label=f"{modelDesc}")

            # modelRecalls.append(totalRecall / totalRecallCount if totalRecallCount > 0 else 0)
            # modelPrecisions.append(totalPrecision / totalPrecisionCount if totalPrecisionCount > 0 else 0)

            analysisDict["model"].append(modelDesc)
            analysisDict["model"].append(modelDesc)
            analysisDict["model"].append(modelDesc)
            analysisDict["statistics"].append("coverage")
            analysisDict["statistics"].append("precision")
            analysisDict["statistics"].append("coverage-precision F1")
            for stat in [recalls, precisions, F1s]:
                if len(stat) != 4:
                    print("?")
                for r, s in zip(focusRanks, stat):
                    analysisDict[r].append(s)

        
        pandas.DataFrame(summaryDict).to_excel(writer, sheet_name='summary', index=False)
        pandas.DataFrame(analysisDict).to_excel(writer, sheet_name='analysis', index=False)
        
    ax.set_xlabel("rank")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Performance of {len(models)} models on {dataset}. Missing label are set as {missingLabel}")
    ax.set_xticks(x*1.5 + width * (len(models) - 1)/2)
    ax.set_xticklabels(rankLevels)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if (withSubRank):
        fileName = f"working/performance_withSub_{dataset}_{missingLabel}.png"
    else:
        fileName = f"working/performance_{dataset}_{missingLabel}.png"

    plt.savefig(fileName, bbox_inches="tight")
    plt.close()


    # save scatter plot
    if (withSubRank):
        fileName = f"working/performance_withSub_{dataset}_{missingLabel}_scatter.png"
    else:
        fileName = f"working/performance_{dataset}_{missingLabel}_scatter.png"

    markers = [m for m in mmarkers.MarkerStyle.markers.keys() if isinstance(m, str) and m not in (".", ",", " ", "")]
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
    fig.savefig(fileName, bbox_inches="tight")
    plt.close()

    

    IOUtils.showInfo('Done')
    return samples

def testModelVirusIdentity(models:dict[str, Module], dataset):
    queryFilePath = f"/Data/VirusClassification/dataset/{dataset}/{dataset}.fasta"
    querySubsetFilePath = None
    config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

    # from tools.evaluate import analyseStatistics
    samples = IOUtils.loadSamples(config.queryFilePath, config.querySubsetFilePath)

    for modelDesc, model in models.items():
        IOUtils.showInfo(f'getting {model.moduleName} results')
        model.getResults(samples)
    IOUtils.showInfo('calculating statistics')
    # with open(f"/Data/VirusClassification/dataset/{dataset}/answer_virusIdentify.json") as fp:
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
    fileName = f"{config.cacheAnalysisFolder}/analysis_{dataset}_virusIdentify_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.cacheAnalysisFolder}/analysis_{dataset}_virusIdentify_{analysisIndex}.xlsx"
    

    # with pandas.ExcelWriter(fileName) as writer:
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

        # pandas.DataFrame(summaryDict).to_excel(writer, sheet_name='summary', index=False)
        pandas.DataFrame(summaryDict).to_csv('working/result.csv', index=False)
        

    IOUtils.showInfo('Done')

def sampleWiseAnalysis(models:dict[str, Module], dataset, evaluationMethod, subset=None, missingLabel="Unknown", analyseList:list[tuple[str, str]]=list()):
    samples = testModel(models, dataset, evaluationMethod, subset, missingLabel)
    if (analyseList is None):
        IOUtils.showInfo("Nothing to analyse. Exit")
        return
    NucleotideUtils.extractProtein(samples)
    analysisIndex = 0
    fileName = f"{config.cacheAnalysisFolder}/sampleAnalysis_{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.cacheAnalysisFolder}/sampleAnalysis_{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    
    with pandas.ExcelWriter(fileName) as writer:
    # for _ in range(1):
    #     writer = pandas.ExcelWriter(fileName)
        # sheet 1: all information
        sampleIDs = list()
        sampleLengths = list()
        sampleProteinCounts = list()
        sampleProteinLengths = list()
        stdResults = list()
        stdResultRank = list()
        modelResults:dict[str, list[tuple[str, str, str, str]]] = {model.moduleName: list() for model in models.values()} # for each model, provide a list of (model_result, model_result_rank, LCA_rank)


        for sample in samples:
            sampleIDs.append(sample.id)
            sampleLengths.append(sample.length)
            sampleProteinCounts.append(len(sample.proteins))
            sampleProteinLengths.append(sum(p.length for p in sample.proteins))
            std = sample.info["stdResult"]
            if (std is not None and std.ICTVNode is not None):
                stdNode = std.ICTVNode
                stdResults.append(stdNode.name)
                stdResultRank.append(stdNode.rank)
            else:
                stdNode = None
                stdResults.append('N/A')
                stdResultRank.append('N/A')

            for model in models.values():
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

        summaryDict = {
            "id": sampleIDs,
            "length": sampleLengths,
            "protein_count": sampleProteinCounts,
            "protein_length": sampleProteinLengths,
            "ground_truth": stdResults,
            "ground_truth_rank": stdResultRank
        }

        for modelDesc, model in models.items():
            preds, predRanks, LCAs, LCARanks = zip(*modelResults[model.moduleName])
            summaryDict[f"{modelDesc}_pred"] = preds
            summaryDict[f"{modelDesc}_pred_rank"] = predRanks
            summaryDict[f"{modelDesc}_LCA"] = LCAs
            summaryDict[f"{modelDesc}_LCA_rank"] = LCARanks

        # for (k, v) in summaryDict.items():
        #     IOUtils.showInfo(f"{k}: {len(v)}")
        
        summaryDF = pandas.DataFrame(summaryDict)
        

        summaryDF.to_excel(writer, sheet_name='summary', index=False)

        # sheet 2: protein related info
        for factor1, factor2 in analyseList:
            if (factor2 in ["length", "protein_count", "protein_length"]):
                modelDesc = factor1
                # need to manually bin the factor 2
                if (factor2 in ["length", "protein_length"]):
                    bins = list(range(0, 10000, 1000)) + list(range(10000, 100000, 10000)) + [numpy.inf]
                    # bins = numpy.linspace(summaryDF[factor2].min(), summaryDF[factor2].max(), 21)
                else:
                    bins = list(range(10)) + list(range(10, 200, 10)) + [numpy.inf]
                summaryDF['A_bin'] = pandas.cut(summaryDF[factor2], bins=bins, include_lowest=True)
                summaryDF["tmp"] = pandas.Categorical(summaryDF[f"{modelDesc}_LCA_rank"], categories=["N/A"] + list(config.rankLevels.keys()), ordered=True)
                analyseDF = pandas.crosstab(summaryDF['A_bin'], summaryDF["tmp"])
                analyseDF.to_excel(writer, sheet_name=f'{modelDesc}_{factor2}'[:31])

            else:
                modelDesc1 = factor1
                modelDesc2 = factor2
                model1 = models[modelDesc1]
                model2 = models[modelDesc2]
                # df1: cross table of two models
                cellText = f"{modelDesc1} \\ {modelDesc2}"
                summaryDF[cellText] = pandas.Categorical(summaryDF[f"{modelDesc1}_LCA_rank"], categories=["N/A"] + list(config.rankLevels.keys()), ordered=True)
                summaryDF["tmp2"] = pandas.Categorical(summaryDF[f"{modelDesc2}_LCA_rank"], categories=["N/A"] + list(config.rankLevels.keys()), ordered=True)
                analyseDF = pandas.crosstab(summaryDF[cellText], summaryDF[f"tmp2"])
                analyseDF.to_excel(writer, sheet_name=f'{modelDesc1}_{modelDesc2}'[:31])


                # df2: similarity table of two models
                modelLCAs = {rank: 0 for rank in config.rankLevels.keys()}
                modelGTLCAs = {rank: 0 for rank in config.rankLevels.keys()}

                modelLCAs["N/A"] = 0
                modelLCAs["No GT"] = 0
                modelLCAs[f"{modelDesc1}_only"] = 0
                modelLCAs[f"{modelDesc2}_only"] = 0

                modelGTLCAs["N/A"] = 0
                modelGTLCAs["No GT"] = 0
                modelGTLCAs[f"{modelDesc1}_only"] = 0
                modelGTLCAs[f"{modelDesc2}_only"] = 0
                for sample in samples:
                    std = sample.info["stdResult"]
                    if (std is None or std.ICTVNode is None):  # currently we only focus those with GT
                        modelLCAs["No GT"] += 1
                        modelGTLCAs["No GT"] += 1
                        continue
                    stdNode = std.ICTVNode
                    pred1 = sample.results[model1.moduleName]
                    pred2 = sample.results[model2.moduleName]

                    pred1Avail = pred1 is not None and pred1.node is not None
                    pred2Avail = pred2 is not None and pred2.node is not None
                    if (pred1Avail):
                        pred1 = pred1.node.ICTVNode
                    if (pred2Avail):
                        pred2 = pred2.node.ICTVNode

                    if (pred1Avail and pred2Avail):
                        LCANode1 = taxoTree.ICTVTree.findLCA([pred1, pred2])
                        LCANode2 = taxoTree.ICTVTree.findLCA([pred1, pred2, stdNode])
                        modelLCAs[LCANode1.rank] += 1
                        modelGTLCAs[LCANode2.rank] += 1
                    elif (pred1Avail):
                        modelLCAs[f"{modelDesc1}_only"] += 1
                        modelGTLCAs[f"{modelDesc1}_only"] += 1
                    elif (pred2Avail):
                        modelLCAs[f"{modelDesc2}_only"] += 1
                        modelGTLCAs[f"{modelDesc2}_only"] += 1
                    else:
                        modelLCAs["N/A"] += 1
                        modelGTLCAs["N/A"] += 1

                analyseDF = pandas.DataFrame({
                    "LCA_rank": list(modelLCAs.keys()),
                    "model_LCA_count": list(modelLCAs.values()),
                    "model_GT_LCA_count": list(modelGTLCAs.values())
                })



                analyseDF.to_excel(writer, sheet_name=f'{modelDesc1}_{modelDesc2}'[:27] + '_LCA')
                
    # writer.close()


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


# def mergeMinimapAndModel(sample:Sample, modelNames):
#     minimapName, genomadName, minimapName2 = modelNames
#     if (minimapName2 in sample.results):
#         return sample.results[minimapName2]
#     elif (genomadName in sample.results):
#         return sample.results[genomadName]
#     else:  # only first model has run
#         return sample.results[minimapName]

# def mergeMLAndGenomad(sample:Sample, modelNames):
#     mlName, genomadName = modelNames
#     if (genomadName in sample.results):
#         return sample.results[genomadName]
#     else:
#         return sample.results[mlName]

def basicMerge(sample:Sample, modelNames, currentModelIndex):
    return sample.results[modelNames[currentModelIndex]]
    # for n in reversed(modelNames):
    #     if (n in sample.results):
    #         return sample.results[n]
    # return None

def taxoGenoMerge(sample:Sample, modelNames, currentModelIndex):
    if (currentModelIndex < 2):
        return None
    minimapResult = sample.results[modelNames[2]] if sample.results[modelNames[0]] is None else sample.results[modelNames[0]]
    mlResult = sample.results[modelNames[1]]
    if (currentModelIndex == 2):  # minimap again
        if (minimapResult is None): # case 1
            return mlResult
        if (mlResult is None): # case 2
            return minimapResult
        # case 3: both result exists, let's check if they are the same
        minimapResultNode = minimapResult.node.ICTVNode
        mlResultNode = mlResult.node.ICTVNode
        if (minimapResultNode == mlResultNode or minimapResultNode in mlResultNode.path or mlResultNode in minimapResultNode.path):
            return minimapResult
        else:  # if the ml and minimap result are not the same, we call genomad
            return None
    elif (currentModelIndex == 3): # genomad result
        genomadResult = sample.results[modelNames[3]]
        if (genomadResult is None):
            return minimapResult  # when we need to run genomad, it means that we have both minimap result and ml result. we use minimap by default
        else:
            return genomadResult

def errMerge(sample:Sample, modelNames, currentModelIndex):
    if (currentModelIndex == 0):
        return None
    result1 = sample.results[modelNames[0]]
    result2 = sample.results[modelNames[1]]
    if (result1 is None):
        return result2
    if (result2 is None):
        return result1
    # if they are the same, it doesn't matter which one to return; other wise, we use the second result.
    # Therefore, we can directly return the second result
    return result2

def diamondMerge(sample:Sample, modelNames, currentModelIndex):
    if (modelNames[currentModelIndex].startswith("diamond") and currentModelIndex != len(modelNames) - 1):
        res:PlainResult = sample.results[modelNames[currentModelIndex]]
        if (res is not None and res.score > 0.5):
            return res
        else:
            return None
    else:
        return sample.results[modelNames[currentModelIndex]]

def virtaxDiamondMerge1(sample: Sample, modelNames, currentModelIndex):
    if (currentModelIndex == 0 or currentModelIndex == 3):
        return sample.results[modelNames[currentModelIndex]]
    if (currentModelIndex == 1):
        return None
    if (currentModelIndex == 2):
        result1:PlainResult = sample.results[modelNames[1]]
        result2:MLResult = sample.results[modelNames[2]]
        if (result1 is None):
            return result2
        elif (result2 is None):
            if (diamondScore > 0.45):
                return result1
            else:
                return None
        else:
            diamondScore = result1.score
            # IOUtils.showInfo(f"diamond score: {diamondScore}, ML score: {list(result2.scores.values())}")
            if diamondScore > 0.45:
                return result1
            else:
                return result2

def virtaxDiamondMerge2(sample: Sample, modelNames, currentModelIndex):
    if (currentModelIndex == 0):
        return sample.results[modelNames[currentModelIndex]]
    if (currentModelIndex == 1 or currentModelIndex == 2):
        return None
    if (currentModelIndex == 3):
        result1:PlainResult = sample.results[modelNames[1]]
        result2:MLResult = sample.results[modelNames[2]]
        result3 = sample.results[modelNames[3]]  # minimap result
        if (result1 is None or result1.node is None):
            return result2
        elif (result2 is None or result2.node is None):
            if (diamondScore > 0.45):
                return result1
            else:
                return None
        else:
            node1 = result1.node.ICTVNode
            node2 = result2.node.ICTVNode
            diamondScore = result1.score
            if (result3 is not None and result3.node is not None):
                node3 = result3.node.ICTVNode
            else:
                node3 = None
            
            if (node3 is not None):
                # case 1: minimap align with diamond, select minimap result
                for n in node3.path:
                    if node1.name == n.name:
                        return result3

                # case 2: ML align with minimap, select ML result
                for n in node3.path:
                    if node2.name == n.name:
                        return result2
                    
            # case 3:ML align with diamond, select ML result
            for n in node1.path:
                if n.name == node2.name:
                    return result2
            
            # case 4： results are all different, and minimap is available
            if (node3 is not None):
                alignLCA = taxoTree.ICTVTree.findLCA([node1, node3])  # find LCA of minimap and diamond
                LCAResult = PlainResult(alignLCA.name)

                return LCAResult
            
            # case 5: all different, and minimap is unavailable
            return PlainResult(taxoTree.ICTVTree.findLCA([node1, node2]).name)
            
            # # case 4.1: ML result on LCA path, return LCA result
            # for n in alignLCA.path:
            #     if node2.name == n.name:
            #         return LCAResult

            # # case 4.2: LCA result on ML path, return LCA result
            # for n in node2.path:
            #     if alignLCA.name == n.name:
            #         return LCAResult
            
            # # case 4.3: LCA and ML are of different path, return 

def virtaxDiamondMerge3(sample: Sample, modelNames, currentModelIndex):
    if (currentModelIndex == 0):
        return sample.results[modelNames[currentModelIndex]]
    if (currentModelIndex == 1 or currentModelIndex == 2):
        return None
    if (currentModelIndex == 3):
        result1:PlainResult = sample.results[modelNames[1]]
        result2:MLResult = sample.results[modelNames[2]]
        result3 = sample.results[modelNames[3]]  # minimap result
        if (result1 is None or result1.node is None):
            return result2
        # elif (result2 is None or result2.node is None):
        #     if (diamondScore > 0.45):
        #         return result1
        #     else:
        #         return None
        else:
            node1 = result1.node.ICTVNode
            node2 = result2.node.ICTVNode
            diamondScore = result1.score
            minimapScore = 0
            if (result3 is not None and result3.node is not None):
                node3 = result3.node.ICTVNode
                minimapScore = list(result3.scores.values())[0]
            else:
                node3 = None

            if (diamondScore > minimapScore):
                alignNode = node1
                alignScore = diamondScore
            else:
                alignNode = node3
                alignScore = minimapScore

            if (alignScore > 0.45):
                return PlainResult(alignNode.name, alignScore)
            return result2
        

def getModels():
    # VirTaxonomer
    thRank = {
        "": "f",
        "pos": "g",
        "60": "g",
        "cm": "g",
        "sa": "g",
        "sa_cm": "g",
        "pos_sa": "g",
        "60_sa": "g",
        "pos_cm": "g",
        "pos_cm_sa": "g",
        # "60_cm": 'g',
        # "60_cm_sa": 'g'
    }

    thRank2 = {
        "": "f",
        "pos": "g",
        "60": "g",
        "cm": "g",
        "sa": "g",
        "sa_cm": "g",
        "pos_sa": "g",
        "60_sa": "g",
        "pos_cm": "g",
        "pos_cm_sa": "g",
        "60_cm": 'g',
        "60_cm_sa": 'g'
    }
    virTaxonomer_bottomup = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
    MinimapMLMergeModule(
        MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('topdown', 0.45, '1111000')
        MLModule('bottomup', 0.45, '1011000')
        # MLModule('bottomup', 0.45, '1011000')
        )
    )
    virTaxonomer_minimap = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
    )

    virTaxonomer_minimap_train = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4_ML_train', factors=['60', 'completeMatch']), 
        ESM()]),
        MinimapThreshRankModule('VMRv4_ML_train', limitOutputDict=thRank),
    )

    virTaxonomer_esm = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        MLModule('bottomup', 0.45, '1011000')
    )
    virTaxonomer_esm_train = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4_ML_train', factors=['60', 'completeMatch']), 
        ESM()]),
        MLModule('bottomup', 0.45, '1011000')
    )
    ml = MLModule('bottomup', 0.45, '1011000')
    ml_highest = MLModule('highest', 0.45, '1011000')
    virTaxonomer_bottomup_train = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4_ML_train', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
    MinimapMLMergeModule(
        MinimapThreshRankModule('VMRv4_ML_train', limitOutputDict=thRank),
        # MLModule('topdown', 0.45, '1111000')
        MLModule('bottomup', 0.45, '1011000')
        # MLModule('bottomup', 0.45, '1011000')
        )
    )

    virTaxonomer_bottomup_genus = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
    MinimapMLMergeModule(
        MinimapThreshRankModule('VMRv4', limitOutputDict=thRank2),
        # MLModule('topdown', 0.45, '1111000')
        MLModule('bottomup', 0.45, '1011000')
        # MLModule('bottomup', 0.45, '1011000')
        )
    )

    virTaxonomer_highest_genus = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
    MinimapMLMergeModule(
        MinimapThreshRankModule('VMRv4', limitOutputDict=thRank2),
        # MLModule('topdown', 0.45, '1111000')
        MLModule('highest', 0.45, '1011000')
        # MLModule('bottomup', 0.45, '1011000')
        )
    )

    virTaxonomer_virus_identify = VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()])
    
    virTaxonomer_virus_identify_train = VirusPred([
        MinimapThresholdModule('VMRv4_ML_train', factors=['60', 'completeMatch']), 
        ESM()])

    phagcn2_1000 = PhaGCN('2', lenThresh=1700, n=1000)
    phagcn2_10000 = PhaGCN('2', lenThresh=1700, n=10000)
    phagcn3_10000 = PhaGCN('3', lenThresh=1700, n=10000)
    phagcn3_100000 = PhaGCN('3', lenThresh=1700, n=100000)
    phagcn3_10000_merge = PhaGCN('3-merged', lenThresh=1700, n=10000)
    # phagcn3_100000 = PhaGCN('3', lenThresh=1700, n=100000)
    vcontact = Vcontact('ProkaryoticViralRefSeq211-Merged')
    vcontact_ML_train = Vcontact('ESM-train')
    vcontact_VMRv4 = Vcontact('VMRv4')
    genomad = Genomad()
    blast = Blast(reference="VMRv4")
    blast_train = Blast(reference="VMRv4_ML_train")
    kraken = Kraken()
    metabuli = Metabuli("VMRv4")
    metabuli_train = Metabuli("VMRv4_ML_train")

    minimap = Minimap(reference='VMRv4')
    minimap_train = Minimap(reference='VMRv4_ML_train')
    # minimap_thrank = MinimapThreshRankModule(reference="VMRv4", limitOutputDict=thRank)
    # minimap_thrank_train = MinimapThreshRankModule(reference="VMRv4_ML_train", limitOutputDict=thRank)
    # minimap_thrank_genusonly = MinimapThreshRankModule(reference="VMRv4", limitOutputDict=thRank2)
    # minimap_thrank_genusonly_train = MinimapThreshRankModule(reference="VMRv4_ML_train", limitOutputDict=thRank2)

    minimap_thresh = MinimapThresholdModule(reference="VMRv4", factors=["60", "completeMatch"])
    minimap_thresh_train = MinimapThresholdModule(reference="VMRv4_ML_train", factors=["60", "completeMatch"])

    diamond_sum = Diamond("VMRv4", "sum")
    diamond_sum_train = Diamond("VMRv4_ML_train", "sum")
    diamond_top3 = Diamond("VMRv4", "top3")
    diamond_top3_train = Diamond("VMRv4_ML_train", "top3")
    diamond_vote = Diamond("VMRv4", "vote")
    diamond_vote_genus = Diamond("VMRv4", "vote", threshRank='genus')
    diamond_vote_train = Diamond("VMRv4_ML_train", "vote")
    diamond_vote_genus_train = Diamond("VMRv4_ML_train", "vote", threshRank='genus')

    minimap_diamond_vote = MergeModule([minimap, diamond_vote], basicMerge, "minimap_diamond_vote")
    minimap_diamond_vote_genus = MergeModule([minimap, diamond_vote_genus], basicMerge, "minimap_diamond_vote_genus")
    minimap_diamond_vote_train = MergeModule([minimap_train, diamond_vote_train], basicMerge, "minimap_diamond_vote_train")
    minimap_diamond_vote_genus_train = MergeModule([minimap_train, diamond_vote_genus_train], basicMerge, "minimap_diamond_vote_genus_train")
    minimap_diamond_top3 = MergeModule([minimap, diamond_top3], basicMerge, "minimap_diamond_top3")
    minimap_diamond_vote_double = MergeModule([minimap_thresh, diamond_vote, minimap, diamond_vote], diamondMerge, "minimap_diamond_vote_double")
    minimap_diamond_top3_double = MergeModule([minimap_thresh, diamond_top3, minimap, diamond_top3], diamondMerge, "minimap_diamond_top3_double")

    vitap = VITAP()

    cat = CAT()

    # virTaxonomerTaxoOnly = MergeModule([minimap_thrank, ml, minimap], basicMerge, "minimap_ml")
    virTaxonomerTaxoOnly = MergeModule([minimap_thresh, ml, minimap], basicMerge, "minimap_ml")
    virTaxonomerTaxoOnly_train = MergeModule([minimap_thresh_train, ml, minimap_train], basicMerge, "minimap_ml_train")

    virTaxonomerStandard = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
        virTaxonomerTaxoOnly
    )

    virTaxonomerStandard_train = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4_ML_train', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
        virTaxonomerTaxoOnly_train
    )

    

    minimap_genomad = MergeModule([minimap_thresh, genomad, minimap], basicMerge, "minimap_genomad")
    minimap_genomad_train = MergeModule([minimap_thresh_train, genomad, minimap_train], basicMerge, "minimap_genomad_train")
    ml_genomad = MergeModule([ml, genomad], basicMerge, "ml_genomad")
    virTaxonomer_genomad = MergeModule([minimap_thresh, ml, minimap, genomad], taxoGenoMerge, "minimap_ml_genomad")
    virTaxonomer_genomad_train = MergeModule([minimap_thresh_train, ml, minimap_train, genomad], taxoGenoMerge, "minimap_ml_genomad_train")

    minimap_genomad_err = MergeModule([minimap_thresh, genomad], errMerge, "minimap_genomad_err")
    minimap_genomad_err_train = MergeModule([minimap_thresh_train, genomad], errMerge, "minimap_genomad_err_train")
    ml_genomad_err = MergeModule([ml, genomad], errMerge, "ml_genomad_err")
    virTaxonomer_genomad_err = MergeModule([virTaxonomerTaxoOnly, genomad], errMerge, "minimap_ml_genomad_err")
    virTaxonomer_genomad_err_train = MergeModule([virTaxonomerTaxoOnly_train, genomad], errMerge, "minimap_ml_genomad_err_train")

    virTaxonomer_diamond_taxoOnly = MergeModule([minimap_thresh, diamond_top3, ml, minimap], virtaxDiamondMerge1, "virTaxoDiamond")
    virTaxonomer_diamond_taxoOnly2 = MergeModule([minimap_thresh, diamond_top3, ml, minimap], virtaxDiamondMerge2, "virTaxoDiamond2")
    virTaxonomer_diamond_taxoOnly3 = MergeModule([minimap_thresh, diamond_top3, ml, minimap], virtaxDiamondMerge3, "virTaxoDiamond3")

    virTaxonomer_diamond = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESM()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
        virTaxonomer_diamond_taxoOnly
    )

    # return [minimap, blast, kraken, genomad, vcontact, phagcn, virTaxonomer_bottomup_genus, virTaxonomer_bottomup, virTaxonomer_highest_genus]
    # return [minimap, blast, genomad, phagcn, vcontact, virTaxonomer_virus_identify]
    # return [minimap, blast, genomad, phagcn, vcontact]
    # return [minimap, virTaxonomer_bottomup_genus]
    # return [minimap, blast, kraken, genomad, vcontact, phagcn, virTaxonomer_bottomup, vitap, cat]
    # return [minimap_train, blast, kraken, genomad, vcontact, phagcn, virTaxonomer_bottomup_train, vitap, cat]
    # return [minimap_train, blast, kraken, genomad, vcontact, phagcn, vitap, cat]
    # return [minimap, blast, genomad, vcontact, phagcn, virTaxonomer_bottomup, vitap, cat]
    # return [minimap, blast, genomad, vcontact, phagcn, virTaxonomer_bottomup, vitap]
    # return [minimap, kraken, blast, genomad, vcontact, phagcn, vitap, cat]
    # return [minimap, kraken, blast, genomad, vcontact, phagcn, virTaxonomer_bottomup, vitap]
    # return [minimap, blast, genomad, vcontact, phagcn, vitap, cat]
    # return [cat]
    # return [kraken]
    return {
        # "kraken_NCBI": kraken,
        # "genomAD-1.9_NCBI": genomad,
        # "CAT_NCBI": cat,
        "minimap_VMRv4": minimap, 
        # "minimap_VMRv4(ESMTrain)": minimap_train,
        # "minimap_VMRv4_threshold": minimap_thrank, 
        # "minimap_VMRv4_threshold(ESMTrain)": minimap_thrank_train,
        "blast_VMRv4": blast,
        # "blast_VMRv4(ESMTrain)": blast_train,
        # "Diamond_sum_VMRv4": diamond_sum,
        # "Diamond_sum_VMRv4(ESMTrain)": diamond_sum_train,
        "Diamond_vote_VMRv4": diamond_vote,
        # "Diamond_vote_VMRv4(ESMTrain)": diamond_vote_train,
        # "Diamond_vote_genus_VMRv4": diamond_vote_genus,
        # "Diamond_vote_genus_VMRv4(ESMTrain)": diamond_vote_genus_train,
        # "Diamond_top3_VMRv4": diamond_top3,
        # "Diamond_top3_VMRv4(ESMTrain)": diamond_top3_train,
        "Metabuli v1.0.9.2": metabuli,
        # "Metabuli v1.0.9.2 (ESM Train)": metabuli_train,
        # "vConTACT2_ProkaryoticViralRefSeq211": vcontact,
        # "vConTACT2_VMRv4": vcontact_VMRv4,
        # "vConTACT2_VMRv4(ESMTrain)": vcontact_ML_train,
        # "PhaGCN2_n=1000_VMRv1": phagcn2_1000,
        # "PhaGCN2_n=10000_VMRv1": phagcn2_10000,
        # "PhaGCN3_n=10000_VMRv1": phagcn3_10000,
        # "PhaGCN3_n=100000_VMRv1": phagcn3_100000,
        # "PhaGCN3-merge_n=10000_VMRv1": phagcn3_10000_merge,
        # (virTaxonomer_bottomup_genus, "VirTaxonomer-bottomup-genus_VMRv4"),
        # (virTaxonomer_highest_genus, "VirTaxonomer-highest-genus_VMRv4"),
        "VirTaxonomer-buttomup_VMRv4": virTaxonomer_bottomup,
        # (virTaxonomer_bottomup_train, "VirTaxonomer-buttomup_VMRv4_ESMTrain"),
        # (virTaxonomer_esm, "VirTaxonomer-Identify+ML_VMRv4"),
        # (virTaxonomer_esm_train, "VirTaxonomer-Identify(ESMTrain)+ML_VMRv4"),
        # (virTaxonomer_minimap, "VirTaxonomer-Minimap_VMRv4"),
        # (virTaxonomer_minimap_train, "VirTaxonomer-Identify(ESMTrain)+Minimap(ESMTrain)_VMRv4"),
        # (virTaxonomer_virus_identify, "VirTaxonomer-Identify"),
        # (virTaxonomer_virus_identify_train, "VirTaxonomer-Identify(ESMTrain)")
        "VirTaxonomer-ML_VMRv4": ml,
        # "VirTaxonomer-ML_VMRv4": ml_highest,
        "VirTaxonomer": virTaxonomerStandard,
        # "VirTaxonomer (ESMTrain)": virTaxonomerStandard_train,
        # "VirTaxonomer (no viral identify)": virTaxonomerTaxoOnly,
        # "VirTaxonomer (no viral identify) (ESMTrain)": virTaxonomerTaxoOnly_train,
        # "minimap_genomad (no viral identify)": minimap_genomad,
        # "minimap_genomad (no viral identify) (ESMTrain)": minimap_genomad_train,
        # "ml_genomad (no viral identify)": ml_genomad,
        # "minimap_ml_genomad (no viral identify)": virTaxonomer_genomad,
        # "minimap_ml_genomad (no viral identify) (ESMTrain)": virTaxonomer_genomad_train,

        # "minimap, genomad fix (no viral identify)": minimap_genomad_err,
        # "minimap, genomad fix (no viral identify) (ESMTrain)": minimap_genomad_err_train,
        # "ml genomad fix (no viral identify)": ml_genomad_err,
        # "minimap_ml genomad fix (no viral identify)": virTaxonomer_genomad_err,
        # "minimap_ml genomad fix (no viral identify) (ESMTrain)": virTaxonomer_genomad_err_train,

        "minimap_diamond_vote": minimap_diamond_vote,
        # "minimap_diamond_vote(ESMTrain)": minimap_diamond_vote_train,
        # "minimap_diamond_vote_genus": minimap_diamond_vote_genus,
        # "minimap_diamond_vote_genus(ESMTrain)": minimap_diamond_vote_genus_train,
        # "minimap_diamond_top3": minimap_diamond_top3,
        # "minimap_diamond_vote_double": minimap_diamond_vote_double,
        # "minimap_diamond_top3_double": minimap_diamond_top3_double,

        "virTax_diamond_vote": virTaxonomer_diamond,
        "virTax_diamond_vote_taxoOnly": virTaxonomer_diamond_taxoOnly,
        "virTax_diamond_vote_taxoOnly2": virTaxonomer_diamond_taxoOnly2,
        "virTax_diamond_vote_taxoOnly3": virTaxonomer_diamond_taxoOnly3,
    }

def main():
    # missingLabel = "Unknown"
    missingLabel = "Other"

    analyseList = [
        # ("minimap_VMRv4", "Diamond_vote_VMRv4"),
        # ("minimap_VMRv4(ESMTrain)", "Diamond_vote_VMRv4(ESMTrain)"),
        # ("Diamond_vote_VMRv4", "length"),
        # ("Diamond_vote_VMRv4(ESMTrain)", "length"),
        # ("Diamond_vote_VMRv4", "protein_count"),
        ("VirTaxonomer-buttomup_VMRv4", "VirTaxonomer")
        # ("Diamond_vote_VMRv4(ESMTrain)", "protein_count"),
        # ("Diamond_vote_VMRv4", "protein_length"),
        # ("Diamond_vote_VMRv4(ESMTrain)", "protein_length"),
        # ("minimap_diamond_vote", "minimapÞ_diamond_vote_genus"),
    ]

    # analyseList = None

    # mergeCachedResults()
    models = getModels()
    # # testModel(models, 'refseq_2024_test', 'textMatch')
    # # testModel(models, 'genbank_2024_test', 'textMatch')

    # testModel(models, 'vitap', 'std', missingLabel=missingLabel)
    # testModel(models, 'VMRv4_test_subseq', 'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'VMRv4_test', 'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'refseq_2024_test',  'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'genbank_2024_test', 'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'genbank_2025_2025Spring', 'accessionMatch', missingLabel=missingLabel)

    # sampleWiseAnalysis(models, 'VMRv4_test_subseq',  'accessionMatch', missingLabel=missingLabel, analyseList=analyseList)
    # sampleWiseAnalysis(models, 'VMRv4_test',  'accessionMatch', missingLabel=missingLabel, analyseList=analyseList)
    # sampleWiseAnalysis(models, 'refseq_2024_test',  'accessionMatch', missingLabel=missingLabel, analyseList=analyseList)
    # sampleWiseAnalysis(models, 'genbank_2024_test',  'accessionMatch', missingLabel=missingLabel, analyseList=analyseList)
    sampleWiseAnalysis(models, 'genbank_2024_2024',  'accessionMatch', missingLabel=missingLabel, analyseList=analyseList)
    # sampleWiseAnalysis(models, 'genbank_2025_2025Spring',  'accessionMatch', missingLabel=missingLabel, analyseList=analyseList)
    # testModelVirusIdentity(models, 'HGUT-Arch-Virus')
    # testModel(models, 'genbank_2024_test', 'accessionMatch')
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "species")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "genus1")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "genus2")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "familyclass")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "familyclass2")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "kingdomphylumorder")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "superkingdomrealmnonVirus")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "superkingdomrealmnonVirus1")
    # testModel(models, 'genbank_2024_test', 'accessionMatch', "superkingdomrealmnonVirus2")
    

if (__name__ == '__main__'):
    main()
