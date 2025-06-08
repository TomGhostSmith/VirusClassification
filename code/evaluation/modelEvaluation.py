import os
# import sys
import json
import numpy
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
# sys.path.append('./code')

from utils import IOUtils
from prototype.module import Module

from config import config

datasetRoot = "/Data/VirusClassification/dataset"

from utils.NucleotideUtils import NucleotideUtils

from entity.taxoTree import taxoTree

import matplotlib.markers as mmarkers


def testModel(models:dict[str, Module], dataset, evaluationMethod, subset=None, missingLabel="Unknown"):
    IOUtils.showInfo(f"Test {len(models)} models on {dataset}")
    queryFilePath = f"{datasetRoot}/{dataset}/{dataset}.fasta"
    if (subset is not None):
        querySubsetFilePath = f"{datasetRoot}/{dataset}/{subset}.txt"
    else:
        querySubsetFilePath = None
    modelRoot = config.modelRoot
    outputRoot = config.outputRoot
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
    with open(f"{datasetRoot}/{dataset}/answer_{evaluationMethod}.json") as fp:
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
    fileName = f"{config.analysisFolder}/table/{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.analysisFolder}/table/{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"


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
        for idx, (modelDesc, model) in tqdm(list(enumerate(list(models.items()))), desc="evaluate", unit="model"):
            preds = dict()
            for sample in samples:
                res = sample.results[model.moduleName]
                if (res is not None):
                    preds[sample.id] = res.node
                else:
                    preds[sample.id] = None
            df, _ = analyseStatistics(preds, stdResults, missingLabel)
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
            ax.bar(x*1.5 + idx*width, recallList, width, color='white', hatch='/', edgecolor=availableColors[idx])
            # ax.bar(x + idx*width, accuracyList, width, color=bars[0].get_facecolor(), label=f"model {idx} accuracy", hatch='//')
            # ax.bar(x*1.5 + idx*width, accuracyList, width, color='white', hatch='/', edgecolor=bars[0].get_facecolor(), label=f"{modelDesc}")
            # ax.bar(x*1.5 + idx*width, accuracyList, width, color=bars[0].get_facecolor(), alpha=0.9, label=f"{modelDesc}")
            ax.bar(x*1.5 + idx*width, accuracyList, width, color=availableColors[idx], alpha=1, label=f"{modelDesc}")

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
        fileName = f"{config.analysisFolder}/figure/performance_withSub_{dataset}_{missingLabel}.png"
    else:
        fileName = f"{config.analysisFolder}/figure/performance_{dataset}_{missingLabel}.png"

    plt.savefig(fileName, bbox_inches="tight")
    plt.close()


    # save scatter plot
    if (withSubRank):
        fileName = f"{config.analysisFolder}/figure/performance_withSub_{dataset}_{missingLabel}_scatter.png"
    else:
        fileName = f"{config.analysisFolder}/figure/performance_{dataset}_{missingLabel}_scatter.png"

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

def sampleWiseAnalysis(models:dict[str, Module], dataset, evaluationMethod, subset=None, missingLabel="Unknown", analyseList:list[tuple[str, str]]=list()):
    samples = testModel(models, dataset, evaluationMethod, subset, missingLabel)
    if (analyseList is None):
        IOUtils.showInfo("Nothing to analyse. Exit")
        return
    NucleotideUtils.extractProtein(samples)

    DFs:dict[str, pandas.DataFrame] = dict()  # desc: DF
    
    # sheet 1: all information
    sampleIDs = list()
    sampleLengths = list()
    sampleProteinCounts = list()
    sampleProteinLengths = list()
    stdResults = list()
    stdResultRank = list()
    modelResults:dict[str, list[tuple[str, str, str, str]]] = {model.moduleName: list() for model in models.values()} # for each model, provide a list of (model_result, model_result_rank, LCA_rank)

    modelRankResults:dict[str, dict[str, list[str]]] = {modelDesc: {r: list() for r in config.evaluationRanks} for modelDesc in models.keys()}

    for sample in tqdm(samples, desc="sample wise analysis"):
        sampleIDs.append(sample.id)
        sampleLengths.append(sample.length)
        sampleProteinCounts.append(len(sample.proteins))
        sampleProteinLengths.append(sum(p.length for p in sample.proteins))
        std = sample.info["stdResult"]

        stdRanks = {r: None for r in config.evaluationRanks}
        if (std is not None and std.ICTVNode is not None):
            stdNode = std.ICTVNode
            stdResults.append(stdNode.name)
            stdResultRank.append(stdNode.rank)
            for n in stdNode.path:
                stdRanks[n.rank] = n.name  # non-standard rank will be discarded later
        else:
            stdNode = None
            stdResults.append('N/A')
            stdResultRank.append('N/A')

        for modelDesc, model in models.items():
            pred = sample.results[model.moduleName]
            predRanks = {r: None for r in config.evaluationRanks}
            if (pred is not None and pred.node is not None):
                pred = pred.node.ICTVNode
                if (stdNode is not None):
                    for n in pred.path:
                        predRanks[n.rank] = n.name
                    LCANode = taxoTree.ICTVTree.findLCA([pred, stdNode])
                    modelResults[model.moduleName].append((pred.name, pred.rank, LCANode.name, LCANode.rank))
                else:
                    modelResults[model.moduleName].append((pred.name, pred.rank, "N/A", "N/A"))
            else:
                modelResults[model.moduleName].append(("N/A", "N/A", "N/A", "N/A"))
            
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
    DFs["raw results"] = summaryDF

    # sheet 2: performance related analysis
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
            summaryDF["tmp"] = pandas.Categorical(summaryDF[f"{modelDesc}_LCA_rank"], categories=["N/A"] + config.evaluationRanks, ordered=True)
            analyseDF = pandas.crosstab(summaryDF['A_bin'], summaryDF["tmp"], dropna=False)
            DFs[f'{modelDesc} vs {factor2}'] = analyseDF

        elif (isinstance(factor2, list) and len(factor2) > 1):
            # one model compared to multi model, confusion matrix only
            tmpDF = {}
            stackDFs = list()
            for r in config.evaluationRanks:
                tmpDF[f"model1_{r}"] = modelRankResults[modelDesc1][r]
                modelDesc = factor2[0]
                tmpDF[f"model2_{r}"] = numpy.array(modelRankResults[modelDesc][r])
                for modelDesc in factor2[1:]:
                    tmpDF[f"model2_{r}"] = numpy.where(numpy.array(modelRankResults[modelDesc][r]) == tmpDF[f"model2_{r}"], tmpDF[f"model2_{r}"], "Other")
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

        else:
            modelDesc1 = factor1
            modelDesc2 = factor2[0] if isinstance(factor2, list) else factor2
            model1 = models[modelDesc1]
            model2 = models[modelDesc2]
            # df1: cross table of two models
            cellText = f"{modelDesc1} \\ {modelDesc2}"
            summaryDF[cellText] = pandas.Categorical(summaryDF[f"{modelDesc1}_LCA_rank"], categories=["N/A"] + config.evaluationRanks, ordered=True)
            summaryDF["tmp2"] = pandas.Categorical(summaryDF[f"{modelDesc2}_LCA_rank"], categories=["N/A"] + config.evaluationRanks, ordered=True)
            analyseDF = pandas.crosstab(summaryDF[cellText], summaryDF[f"tmp2"], dropna=False)
            DFs[f'{modelDesc1} vs {modelDesc2}: difference'] = analyseDF


            # df2: similarity table of two models
            modelLCAs = list()
            modelGTLCAs = list()
            for sample in samples:
                std = sample.info["stdResult"]
                if (std is None or std.ICTVNode is None):  # currently we only focus those with GT
                    modelLCAs.append("No GT")
                    modelGTLCAs.append("No GT")
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
                    modelLCAs.append(LCANode1.rank)
                    modelGTLCAs.append(LCANode2.rank)
                elif (pred1Avail):
                    modelLCAs.append(f"{modelDesc1}_only")
                    modelGTLCAs.append(f"{modelDesc1}_only")
                elif (pred2Avail):
                    modelLCAs.append(f"{modelDesc2}_only")
                    modelGTLCAs.append(f"{modelDesc2}_only")
                else:
                    modelLCAs.append("N/A")
                    modelGTLCAs.append("N/A")
            
            tmpDF = pandas.DataFrame({
                "id": [sample.id for sample in samples],
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
                tmpDF[f"model1_{r}"] = modelRankResults[modelDesc1][r]
                tmpDF[f"model2_{r}"] = modelRankResults[modelDesc2][r]
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
    
    analysisIndex = 0
    fileName = f"{config.analysisFolder}/samplewise/sampleAnalysis_{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    while (os.path.exists(fileName)):
        analysisIndex += 1
        fileName = f"{config.analysisFolder}/samplewise/sampleAnalysis_{dataset}_{evaluationMethod}_{missingLabel}_{analysisIndex}.xlsx"
    
    with pandas.ExcelWriter(fileName) as writer:
        contentDF = {
            "sheet": list(range(1, len(DFs))),
            "content": list(DFs.keys())[1:]
        }
        pandas.DataFrame(contentDF).to_excel(writer, sheet_name="contents", index=False)
        for idx, (desc, df) in enumerate(DFs.items()):
            if (desc == "raw results"):
                df.to_excel(writer, sheet_name='raw results', index=False)
            else:
                df.to_excel(writer, sheet_name=f"sheet {idx}", index=(not desc.endswith("confusion matrics")))


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