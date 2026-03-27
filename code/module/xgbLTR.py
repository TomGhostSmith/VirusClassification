import os
import json
import numpy
import pandas
import xgboost
import multiprocessing
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import mean_squared_error

from config import config
from utils import IOUtils
from entity.sample import Sample
from prototype.module import Module
from entity.taxoNode import TaxoNode
from moduleResult.plainResult import PlainResult

if multiprocessing.current_process().name == "MainProcess":
    from utils import trainUtils
    from entity.taxoTree import taxoTree
    from utils.NucleotideUtils import NucleotideUtils


class XGBoostLTR(Module):
    def __init__(self, trainset, evalMethod, modules:list[Module], featureModules:list[Module], features:list[str], tops:int, candidateFeatures:list[str]=[], limitOutput=True, complete="no", loss="mse"):
        moduleNames = "+".join([module.moduleName for module in modules])
        featureNames = "+".join(features)
        candidateFeatureNames = "+".join(candidateFeatures)
        self.candidateFeatures = candidateFeatures
        losses = {
            "mse": ("reg:squarederror", "rmse", xgboost.XGBRegressor, getNegMSE, False),
            "ndcg": ("rank:ndcg", "ndcg@1", xgboost.XGBRanker, getTop1Accuracy, True),
            "pairwise": ("rank:pairwise", "ndcg@1", xgboost.XGBRanker, getTop1Accuracy, True)
        }
        if loss not in losses:
            raise ValueError("Unsupported loss function")
        self.loss, self.eval, self.clz, self.scoreFunc, self.keepGroup = losses[loss]
        self.baseName = f"XGBoostLTR-train={trainset};eval={evalMethod};modules={moduleNames};features={featureNames};tops={tops};candidateFeatures={candidateFeatureNames};complete={complete};loss={loss}"
        super().__init__(f"{self.baseName};limitOutput={limitOutput}")
        self.trainset = trainset
        self.evalMethod = evalMethod
        self.modules = modules
        self.featureModules = featureModules
        self.modules = modules
        self.features = features
        self.tops = tops
        self.limitOutput = limitOutput

        if (complete not in ["no", "genus", "topdown"]):
            raise ValueError("Unsupported completion method")
    
        self.complete = complete

        self.modelListFile = f"{config.modelRoot}/XGBoostLTR/names.json"
        self.moduleListMap = {}

    def getFeatures(self, samples:list[Sample]):
        NucleotideUtils.extractProtein(samples)
        for module in self.featureModules:
            IOUtils.showInfo(f"get features from {module.moduleName}")
            module.getResults(samples, keepVotes=True, withMeta=True)
        for module in self.modules:
            IOUtils.showInfo(f"get candidates from {module.moduleName}")
            module.getResults(samples, keepVotes=True, withCandidateMeta=True)

        IOUtils.showInfo("summarize features")
        # basic feature independent on all the model
        for sample in samples:
            sample.info["length"] = sample.length
            sample.info["proteinCount"] = len(sample.proteins)
            partialCounts = {"00": 0, "01": 0, "10": 0, "11": 0}
            for protein in sample.proteins:
                partialCounts[protein.partial] += 1
            for k, v in partialCounts.items():
                sample.info[f"{k}_protein_ratio"] = v / len(sample.proteins) if len(sample.proteins) > 0 else 0

        features = {}
        candidateFeatures = []
        candidateRanges:list[tuple[int, int]] = []
        candidateList:list[TaxoNode] = []

        genusLevel = config.rankLevels["genus"]

        titles = self.features.copy()  # to avoid change the list "self.features"
        for i in range(len(self.modules)):
            titles.append(f"model_{i} rank")
            titles.append(f"model_{i} score")
        titles += self.candidateFeatures
        # for f in self.features:
        #     results = [sample.info.get(f) for sample in samples]
        #     features[f] = results

        candidateFeatureSet = set(self.candidateFeatures)


        for sample in samples:
            rankMaps:list[dict[str, int]] = []
            scoreMaps:list[dict[str, float]] = []
            fullScoreMaps:list[dict[str, float]] = []
            candidateInfos:dict[str, dict] = {}
            basicFeatures = [sample.info.get(f) for f in self.features]

            # get all top x results from all modules
            # calculate the union set of all the potential predictions
            candidates:dict[str, TaxoNode] = {}
            for module in self.modules:
                rankMap:dict[str, int] = {}
                scoreMap:dict[str, float] = {}
                res = sample.results[module.moduleName]
                if (res is not None):
                    idx = 0
                    for r in res:
                        if idx == self.tops:
                            break
                        if r.node is None:
                            continue
                        if r.node.ICTVName in rankMap:
                            continue
                        idx += 1
                        candidates[r.node.ICTVName] = r.node
                        rankMap[r.node.ICTVName] = idx
                        scoreMap[r.node.ICTVName] = r.score
                        if r.node.ICTVName not in candidateInfos:
                            candidateInfos[r.node.ICTVName] = {k: None for k in self.candidateFeatures}
                        for k, v in r.info.items():
                            if k in candidateFeatureSet:
                                candidateInfos[r.node.ICTVName][k] = v
                rankMaps.append(rankMap)
                scoreMaps.append(scoreMap)
                fullScoreMaps.append(sample.info.get(f"{module.moduleName}_votes"))


            # record the start/end candidate into a list
            start = len(candidateList)
            end = len(candidateList) + len(candidates)
            candidateRanges.append((start, end))

            # for each candidate option, generate its features
            for name, node in candidates.items():
                candidateFeature = basicFeatures.copy()
                genusName = name
                if self.complete == "topdown" and config.rankLevels[node.ICTVNode.rank] > genusLevel:
                    for n in node.ICTVNode.path:
                        if n.rank == "genus":
                            genusName = n.name
                            break
                for rankMap, scoreMap, fullScoreMap in zip(rankMaps, scoreMaps, fullScoreMaps):
                    if (name in rankMap):
                        candidateFeature.append(rankMap[name])
                        candidateFeature.append(scoreMap.get(name))
                    elif self.complete != "no" and fullScoreMap is not None and genusName in fullScoreMap: # check the whole score map
                        score = fullScoreMap[genusName]
                        rank = sum(x > score for x in fullScoreMap.values())
                        candidateFeature.append(rank)
                        candidateFeature.append(score)
                    else:
                        candidateFeature.append(None)
                        candidateFeature.append(None)

                        
                candidateFeature += list(candidateInfos[name].values())
                candidateFeatures.append(candidateFeature)
                candidateList.append(node)
                
        features = {k: v for k, v in zip(titles, zip(*candidateFeatures))}

        return pandas.DataFrame(features).astype(dtype=float), candidateList, candidateRanges

    def train(self):
        samples = trainUtils.loadTrainsetSamples(self.trainset, self.evalMethod)
        samplesWithGT = []
        for sample in samples:
            stdNode:TaxoNode = sample.info["stdResult"]
            if (stdNode is None or stdNode.ICTVNode is None):
                continue
            samplesWithGT.append(sample)

        features, candidates, candidateRanges = self.getFeatures(samplesWithGT)

        labels = []
        for sample, (start, end) in zip(samplesWithGT, candidateRanges):
            stdNode = sample.info["stdResult"].ICTVNode
            for candidate in candidates[start:end]:
                lca = taxoTree.ICTVTree.findLCA([stdNode, candidate.ICTVNode])
                labels.append(config.rankLevels[lca.rank])

        # train XGBoost1: learn which module to use
        mainModel = self.trainMainXGBoost(features, numpy.array(labels), candidateRanges)

        return mainModel

    def loadModel(self):
        if (os.path.exists(self.modelListFile)):
            with open(self.modelListFile) as fp:
                self.moduleListMap = json.load(fp)
        if self.baseName not in self.moduleListMap:
            mainModel = self.train()
        else:
            mainModelName = self.moduleListMap[self.baseName]
            if (not os.path.exists(f"{config.modelRoot}/XGBoostLTR/{mainModelName}")):
                mainModel = self.train()
            else:
                mainModel = self.clz()
                mainModel.load_model(f"{config.modelRoot}/XGBoostLTR/{mainModelName}")
        return mainModel            

    def run(self, samples, **kwargs):
        mainModel = self.loadModel()
        features, candidateList, candidateRanges = self.getFeatures(samples)
        scores:list[float] = mainModel.predict(features, )

        # for debug
        f = features.copy()
        f["result"] = scores
        f["candidate"] = [n.ICTVName for n in candidateList]
        f.to_csv("working/XGBLTR2.csv")

        results = [self.getResult(sample, candidateList[start:end], scores[start:end]) for sample, (start, end) in zip(samples, candidateRanges)]

        return results
    
    def getResult(self, sample, candidates:list[TaxoNode], scores:list[float]):
        speciesRank = config.rankLevels["species"]
        results = []
        pairs = sorted(zip(candidates, scores), key=lambda x:x[1], reverse=True)
        for candidate, score in pairs:
            if self.limitOutput:
                targetRank = round(score)
                for n in reversed(candidate.ICTVNode.path):
                    if (config.rankLevels[n.rank] <= targetRank):
                        results.append(PlainResult(n.name, score/speciesRank))
                        break
            else:
                results.append(PlainResult(candidate.ICTVName, score/speciesRank))

        if (len(results) == 0):
            results = None
        
        return results

    def trainMainXGBoost(self, features, targets, candidateRanges):
        IOUtils.showInfo("train XGBoost")
        groupIDs = numpy.zeros(len(targets))
        for idx, (s, e) in enumerate(candidateRanges):
            groupIDs[s:e] = idx


        kfold = GroupKFold(n_splits=5)
        splits = list(kfold.split(features, targets, groupIDs))

        bestScore = -numpy.inf
        bestModel = None
        bestParam = None

        param_grid = {
            "max_depth": [3, 4, 5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "n_estimators": [50, 100, 200]
        }

        for param in list(ParameterGrid(param_grid)):
            for fold_index, (train_idx, val_idx) in enumerate(splits):
                x_train, x_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = targets[train_idx], targets[val_idx]
                gid_train, gid_val = groupIDs[train_idx], groupIDs[val_idx]

                model = self.clz(
                    objective=self.loss,
                    random_state=42,
                    learning_rate=0.05,
                    eval_metric=self.eval,
                    **param
                )

                if (self.keepGroup):
                    model.fit(
                        x_train, y_train, qid=gid_train,
                        verbose=False,
                        # eval_set=[(x_val, y_val)], eval_qid=[gid_val],
                        # callbacks=[EarlyStopping(rounds=50, save_best=True)],
                    )
                else:
                    model.fit(
                        x_train, y_train,
                        verbose=False,
                        # eval_set=[(x_val, y_val)],
                        # callbacks=[EarlyStopping(rounds=50, save_best=True)],
                    )
                    

                # score = self.scoreFunc(model.best_score)
                # score = self.scoreFunc(model.get_score)
                y_pred = model.predict(x_val)
                score = self.scoreFunc(y_val, y_pred, gid_val)

                # IOUtils.showInfo(f"XGB {param} fold {fold_index}: score={score}")
                if score > bestScore:
                    bestModel = model
                    bestParam = param

        # Save to file
        IOUtils.showInfo(f"best model in training: {bestParam}")
        os.makedirs(f"{config.modelRoot}/XGBoostLTR", exist_ok=True)
        
        i = 0
        while True:
            mainXGBoostName = f"{config.modelRoot}/XGBoostLTR/model_{i}-main.json"
            if not os.path.exists(mainXGBoostName):
                break
            i += 1
        bestModel.save_model(mainXGBoostName)   # JSON is human-readable

        if (os.path.exists(self.modelListFile)):
            with open(self.modelListFile) as fp:
                self.moduleListMap = json.load(fp)
        self.moduleListMap[self.baseName] = os.path.basename(mainXGBoostName)
        with open(self.modelListFile, 'wt') as fp:
            json.dump(self.moduleListMap, fp, indent=2)

        return bestModel

def groupID2candidateRanges(groupIDs):
    ranges = []
    start = 0

    for i in range(1, len(groupIDs)):
        if groupIDs[i] != groupIDs[i - 1]:
            ranges.append((start, i))
            start = i

    ranges.append((start, len(groupIDs)))
    return ranges


# def getTop1Accuracy(score):
#     return score

# def getNegMSE(score):
#     return - score

def getTop1Accuracy(y_true, y_pred, groupIDs):
    score = 0
    candidateRanges = groupID2candidateRanges(groupIDs)
    for s, e in candidateRanges:
        highestIndex = numpy.argmax(y_pred[s:e])
        if (y_true[s:e][highestIndex] == max(y_true[s:e])):
            score += 1
    return score / len(candidateRanges)

def getNegMSE(y_true, y_pred, groupIDs):
    return - mean_squared_error(y_true, y_pred)