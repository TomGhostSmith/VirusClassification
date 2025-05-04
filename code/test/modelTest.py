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


from entity.taxoTree import taxoTree
from entity.sample import Sample

import matplotlib.markers as mmarkers


def testModel(models:list[tuple[Module, str]], dataset, evaluationMethod, subset=None, missingLabel="Unknown"):
    queryFilePath = f"/Data/VirusClassification/dataset/{dataset}/{dataset}.fasta"
    if (subset is not None):
        querySubsetFilePath = f"/Data/VirusClassification/dataset/{dataset}/{subset}.txt"
    else:
        querySubsetFilePath = None
    config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

    from tools.evaluate import analyseStatistics
    samples = IOUtils.loadSamples(config.queryFilePath, config.querySubsetFilePath)

    for model, modelDesc in models:
        IOUtils.showInfo(f'getting {model.moduleName} results')
        model.getResults(samples)
    
    if (subset is not None):
        IOUtils.showInfo(f"dataset {dataset} subset {subset} finished. Skip the evaluation")
        return # currently we cannot evaluate a subset, because the answer is not complete
    IOUtils.showInfo('calculating statistics')
    # with open("/Data/VirusClassification/dataset/refseq_2024_test/answer.json") as fp:
    with open(f"/Data/VirusClassification/dataset/{dataset}/answer_{evaluationMethod}.json") as fp:
        stdResults = json.load(fp)
    

    summaryDict = dict()
    summaryDict["index"] = list()
    summaryDict["model"] = list()
    for r in config.resultRanks:
        summaryDict[r.capitalize()] = list()

    # check and get a valid name
    analysisIndex = 0
    fileName = f"{config.cacheAnalysisFolder}/analysis_{dataset}_{evaluationMethod}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.cacheAnalysisFolder}/analysis_{dataset}_{evaluationMethod}_{analysisIndex}.xlsx"


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
        for idx, (model, modelDesc) in enumerate(models):
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

            # bars = ax.bar(x + idx*width, recallList, width, label=f"model {idx} recall", alpha=0.7)
            bars = ax.bar(x*1.5 + idx*width, recallList, width, alpha=0.3, color=availableColors[idx])
            # ax.bar(x + idx*width, accuracyList, width, color=bars[0].get_facecolor(), label=f"model {idx} accuracy", hatch='//')
            # ax.bar(x*1.5 + idx*width, accuracyList, width, color=bars[0].get_facecolor(), hatch='/', edgecolor='black')
            ax.bar(x*1.5 + idx*width, accuracyList, width, color=bars[0].get_facecolor(), alpha=0.9, label=f"{modelDesc}")

            # modelRecalls.append(totalRecall / totalRecallCount if totalRecallCount > 0 else 0)
            # modelPrecisions.append(totalPrecision / totalPrecisionCount if totalPrecisionCount > 0 else 0)

        
        pandas.DataFrame(summaryDict).to_excel(writer, sheet_name='summary', index=False)
        
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
    for idx, ((model, modelDesc), recall, precision) in enumerate(zip(models, modelRecalls["order"], modelPrecisions["order"])):
        axs[0, 0].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100, label=modelDesc)
        axs[0, 0].set_title("order")
        axs[0, 0].grid(True)
        axs[0, 0].set_xlabel("Recall")
        axs[0, 0].set_ylabel("Precision")
    for idx, ((model, modelDesc), recall, precision) in enumerate(zip(models, modelRecalls["family"], modelPrecisions["family"])):
        axs[0, 1].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100)
        axs[0, 1].set_title("family")
        axs[0, 1].grid(True)
        axs[0, 1].set_xlabel("Recall")
        axs[0, 1].set_ylabel("Precision")
    for idx, ((model, modelDesc), recall, precision) in enumerate(zip(models, modelRecalls["genus"], modelPrecisions["genus"])):
        axs[1, 0].scatter(recall, precision, color=availableColors[idx], marker=markers[idx], s=100)
        axs[1, 0].set_title("genus")
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel("Recall")
        axs[1, 0].set_ylabel("Precision")
    for idx, ((model, modelDesc), recall, precision) in enumerate(zip(models, modelRecalls["species"], modelPrecisions["species"])):
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

def testModelVirusIdentity(models:list[tuple[Module, str]], dataset):
    queryFilePath = f"/Data/VirusClassification/dataset/{dataset}/{dataset}.fasta"
    querySubsetFilePath = None
    config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

    # from tools.evaluate import analyseStatistics
    samples = IOUtils.loadSamples(config.queryFilePath, config.querySubsetFilePath)

    for model, modelDesc in models:
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
    for idx, (model, modelDesc) in enumerate(models):
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
    minimap_thrank = MinimapThreshRankModule(reference="VMRv4", limitOutputDict=thRank)
    minimap_thrank_train = MinimapThreshRankModule(reference="VMRv4_ML_train", limitOutputDict=thRank)
    minimap_thrank_genusonly = MinimapThreshRankModule(reference="VMRv4", limitOutputDict=thRank2)
    minimap_thrank_genusonly_train = MinimapThreshRankModule(reference="VMRv4_ML_train", limitOutputDict=thRank2)

    vitap = VITAP(threads=16)

    cat = CAT()

    virTaxonomerTaxoOnly = MergeModule([minimap_thrank, ml, minimap], basicMerge, "minimap_ml")
    virTaxonomerTaxoOnly_train = MergeModule([minimap_thrank_train, ml, minimap_train], basicMerge, "minimap_ml_train")

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

    

    minimap_genomad = MergeModule([minimap_thrank, genomad, minimap], basicMerge, "minimap_genomad")
    minimap_genomad_train = MergeModule([minimap_thrank_train, genomad, minimap_train], basicMerge, "minimap_genomad_train")
    ml_genomad = MergeModule([ml, genomad], basicMerge, "ml_genomad")
    virTaxonomer_genomad = MergeModule([minimap_thrank, ml, minimap, genomad], taxoGenoMerge, "minimap_ml_genomad")
    virTaxonomer_genomad_train = MergeModule([minimap_thrank_train, ml, minimap_train, genomad], taxoGenoMerge, "minimap_ml_genomad_train")

    minimap_genomad_err = MergeModule([minimap_thrank, genomad], errMerge, "minimap_genomad_err")
    minimap_genomad_err_train = MergeModule([minimap_thrank_train, genomad], errMerge, "minimap_genomad_err_train")
    ml_genomad_err = MergeModule([ml, genomad], errMerge, "ml_genomad_err")
    virTaxonomer_genomad_err = MergeModule([virTaxonomerTaxoOnly, genomad], errMerge, "minimap_ml_genomad_err")
    virTaxonomer_genomad_err_train = MergeModule([virTaxonomerTaxoOnly_train, genomad], errMerge, "minimap_ml_genomad_err_train")

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
    return [
        (kraken, "kraken_NCBI"),
        (genomad, "genomAD-1.9_NCBI"),
        (cat, "CAT_NCBI"),
        # (minimap, "minimap_VMRv4"), 
        (minimap_train, "minimap_VMRv4(ESMTrain)"),
        # (minimap_thrank, "minimap_VMRv4_threshold"), 
        # (minimap_thrank_train, "minimap_VMRv4_threshold(ESMTrain)"),
        # (blast, "blast_VMRv4"),
        (blast_train, "blast_VMRv4(ESMTrain)"),
        # (metabuli, "Metabuli v1.0.9.2"),
        # (metabuli_train, "Metabuli v1.0.9.2 (ESM Train)"),
        (vcontact, "vConTACT2_ProkaryoticViralRefSeq211"),
        (vcontact_VMRv4, "vConTACT2_VMRv4"),
        # (vcontact_ML_train, "vConTACT2_VMRv4(ESMTrain)"),
        # (phagcn2_1000, "PhaGCN2_n=1000_VMRv1"),
        # (phagcn2_10000, "PhaGCN2_n=10000_VMRv1"),
        # (phagcn3_10000, "PhaGCN3_n=10000_VMRv1"),
        # (phagcn3_100000, "PhaGCN3_n=100000_VMRv1"),
        # (phagcn3_10000_merge, "PhaGCN3-merge_n=10000_VMRv1"),
        # (virTaxonomer_bottomup_genus, "VirTaxonomer-bottomup-genus_VMRv4"),
        # (virTaxonomer_highest_genus, "VirTaxonomer-highest-genus_VMRv4"),
        # (virTaxonomer_bottomup, "VirTaxonomer-buttomup_VMRv4"),
        # (virTaxonomer_bottomup_train, "VirTaxonomer-buttomup_VMRv4_ESMTrain"),
        # (virTaxonomer_esm, "VirTaxonomer-Identify+ML_VMRv4"),
        # (virTaxonomer_esm_train, "VirTaxonomer-Identify(ESMTrain)+ML_VMRv4"),
        # (virTaxonomer_minimap, "VirTaxonomer-Minimap_VMRv4"),
        # (virTaxonomer_minimap_train, "VirTaxonomer-Identify(ESMTrain)+Minimap(ESMTrain)_VMRv4"),
        # (ml, "VirTaxonomer-ML_VMRv4"),
        # (virTaxonomer_virus_identify, "VirTaxonomer-Identify"),
        # (virTaxonomer_virus_identify_train, "VirTaxonomer-Identify(ESMTrain)")
        # (virTaxonomerStandard, "VirTaxonomer"),
        # (virTaxonomerStandard_train, "VirTaxonomer (ESMTrain)"),
        # (virTaxonomerTaxoOnly, "VirTaxonomer (no viral identify)"),
        # (virTaxonomerTaxoOnly_train, "VirTaxonomer (no viral identify) (ESMTrain)"),
        # (minimap_genomad, "minimap_genomad (no viral identify)"),
        # (minimap_genomad_train, "minimap_genomad (no viral identify) (ESMTrain)"),
        # (ml_genomad, "ml_genomad (no viral identify)"),
        # (virTaxonomer_genomad, "minimap_ml_genomad (no viral identify)"),
        # (virTaxonomer_genomad_train, "minimap_ml_genomad (no viral identify) (ESMTrain)"),

        # (minimap_genomad_err, "minimap, genomad fix (no viral identify)"),
        # (minimap_genomad_err_train, "minimap, genomad fix (no viral identify) (ESMTrain)"),
        # (ml_genomad_err, "ml genomad fix (no viral identify)"),
        # (virTaxonomer_genomad_err, "minimap_ml genomad fix (no viral identify)"),
        # (virTaxonomer_genomad_err_train, "minimap_ml genomad fix (no viral identify) (ESMTrain)")
    ]

def main():
    missingLabel = "Unknown"
    # missingLabel = "Other"

    # mergeCachedResults()
    models = getModels()
    # # testModel(models, 'refseq_2024_test', 'textMatch')
    # # testModel(models, 'genbank_2024_test', 'textMatch')

    # testModel(models, 'vitap', 'std', missingLabel=missingLabel)
    testModel(models, 'VMRv4_test_subseq', 'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'VMRv4_test', 'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'refseq_2024_test',  'accessionMatch', missingLabel=missingLabel)
    # testModel(models, 'genbank_2024_test', 'accessionMatch', missingLabel=missingLabel)
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
