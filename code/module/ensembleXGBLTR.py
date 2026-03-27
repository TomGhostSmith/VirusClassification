import os
import json
import numpy
import pandas
import xgboost
import multiprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, ParameterGrid

from config import config
from utils import IOUtils
from entity.sample import Sample
from prototype.result import Result
from prototype.module import Module
from entity.taxoNode import TaxoNode
from entity.proteinSample import ProteinSample
from moduleResult.plainResult import PlainResult

if multiprocessing.current_process().name == "MainProcess":
    from utils import trainUtils
    from entity.taxoTree import taxoTree
    from utils.NucleotideUtils import NucleotideUtils

class EnsembleXGBoostLTR(Module):
    def __init__(self, trainset, evalMethod, modules:list[Module], featureModules:list[Module], proteinModules:list[Module], poolingModule:Module,
                contigFeatures:list[str], candidateFeatures:list[str], poolingFeature:str, poolingFeatureRange:list, tops:int, limitOutput=True, complete="no", loss="mse", excludeNoHit=False):
        moduleNames = "+".join([module.moduleName for module in modules])
        proteinModuleNames = "+".join([module.moduleName for module in proteinModules])
        contigFeatureNames = "+".join(contigFeatures)
        candidateFeatureNames = "+".join(candidateFeatures)
        self.baseName = f"EnsembleXGBoostLTR-train={trainset};eval={evalMethod};modules={moduleNames};proteinModules={proteinModuleNames};poolingModule={poolingModule.moduleName};contigFeatures={contigFeatureNames};candidateFeatures={candidateFeatureNames};poolingFeature={poolingFeature};tops={tops};complete={complete};excludeNoHit={excludeNoHit},loss={loss}"
        self.excludeNoHit = excludeNoHit
        super().__init__(f"{self.baseName};limitOutput={limitOutput}")
        self.trainset = trainset
        self.evalMethod = evalMethod

        losses = {
            "mse": ("reg:squarederror", "rmse", xgboost.XGBRegressor, getNegMSE, False),
            "ndcg": ("rank:ndcg", "ndcg@1", xgboost.XGBRanker, getTop1Accuracy, True),
            "pairwise": ("rank:pairwise", "ndcg@1", xgboost.XGBRanker, getTop1Accuracy, True)
        }
        if loss not in losses:
            raise ValueError("Unsupported loss function")
        self.loss, self.eval, self.clz, self.scoreFunc, self.keepGroup = losses[loss]

        self.modules = modules
        self.featureModules = featureModules
        self.poolingModule = poolingModule
        self.proteinModules = proteinModules

        self.contigFeatures = contigFeatures
        self.candidateFeatures = candidateFeatures
        self.poolingFeature = poolingFeature
        # if "N/A" not in poolingFeatureRange:
        #     poolingFeatureRange += ["N/A"]  # do not use append to avoid modifying the origin list
        self.poolingFeatureRange = poolingFeatureRange
        self.tops = tops
        self.limitOutput = limitOutput

        if (complete not in ["no", "genus", "topdown"]):
            raise ValueError("Unsupported completion method")
    
        self.complete = complete

        self.modelListFile = f"{config.modelRoot}/EnsembleXGBoostLTR/names.json"
        self.moduleListMap = {}

    def getFeatures(self, samples:list[Sample]):
        NucleotideUtils.extractProtein(samples)
        for module in self.featureModules:
            IOUtils.showInfo(f"get features from {module.moduleName}")
            module.getResults(samples, withMeta=True, withProteinMeta=True)
        
        for module in self.modules:
            IOUtils.showInfo(f"get contig candidates from {module.moduleName}")
            module.getResults(samples, keepVotes=True, withCandidateMeta=True)

        for module in self.proteinModules:
            IOUtils.showInfo(f"get protein candidates from {module.moduleName}")
            module.getResults(samples, keepProteinRes=True, wthPoolingMeta=True)

        IOUtils.showInfo(f"get pooling features from {self.poolingModule.moduleName}")
        self.poolingModule.getResults(samples, withProteinMeta=True)

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

        # contig basic feature + pooling basic feature + candidate scores + candidate features + pooling candidate features
        titles = self.contigFeatures.copy()  # to avoid change the list "self.features"
        for g in self.poolingFeatureRange:
            titles.append(f"group_{g} count")
            titles.append(f"group_{g} totalLength")
            # titles.append(f"group_{g} no hit count")
            # titles.append(f"group_{g} ambiguous count")  # top1 alignment - top2 alignment < threshold
        for i in range(len(self.modules)):
            titles.append(f"model_{i} rank")
            titles.append(f"model_{i} score")
        titles += self.candidateFeatures
        for g in self.poolingFeatureRange:
            for i in range(len(self.proteinModules)):
                titles.append(f"group_{g} model_{i} top1Ratio")
                titles.append(f"group_{g} model_{i} top{self.tops}Ratio")
                titles.append(f"group_{g} model_{i} MRR")
                titles.append(f"group_{g} model_{i} maxScore")  # max of protein.info["maxScore"] or protein.score
                titles.append(f"group_{g} model_{i} maxScoreMargin")  # maxScore - maxScore of non-candidate
                titles.append(f"group_{g} model_{i} sumScore")  # sum of protein.info["sumScore"] or protein.score
                titles.append(f"group_{g} model_{i} sumScoreMargin")  # sumScore - sumScore of non-candidate
                titles.append(f"group_{g} model_{i} avgScore")  # avg of protein.info["maxScore"] or protein.score
                titles.append(f"group_{g} model_{i} avgScoreMargin")  # avgScore - avgScore of non-candidate
                titles.append(f"group_{g} model_{i} minScore")  # min of protein.info["minScore"] or protein.score
                titles.append(f"group_{g} model_{i} avgMargin")  # avg([p.candidate[0].score - p.candidate[1].score for p in proteins]). not related to candidate

                # pooling of proteinCandidateFeatures

        for sample in samples:
            # record the start/end candidate into a list
            start = len(candidateList)
            candidateListSec, candidateFeaturesSec = self.getSampleCandidates(sample)
            end = len(candidateList) + len(candidateListSec)
            candidateRanges.append((start, end))
            candidateList += candidateListSec
            candidateFeatures += candidateFeaturesSec
                
        features = {k: v for k, v in zip(titles, zip(*candidateFeatures))}

        return pandas.DataFrame(features).astype(dtype=float), candidateList, candidateRanges
    
    def getSampleCandidates(self, sample:Sample):
        genusLevel = config.rankLevels["genus"]
        candidateFeatureSet = set(self.candidateFeatures)
        pooling:dict[str, list[ProteinSample]] = {g: [] for g in self.poolingFeatureRange}
        for protein in sample.proteins:
            group = protein.info.get(self.poolingFeature, "N/A")
            if self.excludeNoHit and protein.info["proteinAlignments"] == 0:
                continue
            if group != "N/A":
                pooling[group].append(protein)
        
        # get candidate and its contig feature first
        rankMaps:list[dict[str, int]] = []
        scoreMaps:list[dict[str, float]] = []
        fullScoreMaps:list[dict[str, float]] = []
        candidateInfos:dict[str, dict] = {}
        contigBasicFeatures = [sample.info.get(f) for f in self.contigFeatures]

        poolingBasicFeatures = []
        for proteins in pooling.values():
            # poolingBasicFeatures
            lengths = [protein.length for protein in proteins]
            poolingBasicFeatures.append(len(proteins))
            poolingBasicFeatures.append(sum(lengths))

        poolingStatistics:dict[str, list[ModuleGroupResult|None]] = {}
        for g, proteins in pooling.items():
            if len(proteins) == 0:
                poolingStatistics[g] = [None] * len(self.proteinModules)
                continue
            groupResults = []
            for module in self.proteinModules:
                hasResult = False
                for protein in proteins:
                    if protein.results[module.moduleName] is not None:
                        hasResult = True
                        break
                if (hasResult):
                    groupResults.append(ModuleGroupResult(module, proteins, self.tops))
                else:
                    groupResults.append(None)
            poolingStatistics[g] = groupResults

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
                    rankMap[r.node.ICTVName] = idx
                    scoreMap[r.node.ICTVName] = r.score
                    if r.node.ICTVName not in candidateInfos:
                        candidates[r.node.ICTVName] = r.node
                        candidateInfos[r.node.ICTVName] = {k: None for k in self.candidateFeatures}
                    for k, v in r.info.items():
                        if k in candidateFeatureSet:
                            candidateInfos[r.node.ICTVName][k] = v
            rankMaps.append(rankMap)
            scoreMaps.append(scoreMap)
            fullScoreMaps.append(sample.info.get(f"{module.moduleName}_votes"))

        # also get top-k candidate for each protein, and merge to candidates
        for module in self.proteinModules:
            for protein in sample.proteins:
                res = protein.results[module.moduleName]
                if (res is not None):
                    idx = 0
                    for r in res:
                        if idx == self.tops:
                            break
                        if r.node is None:
                            continue
                        idx += 1
                        if r.node.ICTVName not in candidateInfos:
                            candidates[r.node.ICTVName] = r.node
                            candidateInfos[r.node.ICTVName] = {k: None for k in self.candidateFeatures}
                        # don't add candidateInfos because we haven't aggregate proteins to contigs


        # for each candidate option, generate its features
        candidateFeatures = []
        for name, node in candidates.items():
            candidateFeature = contigBasicFeatures + poolingBasicFeatures
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

            # add pooling candidate features
            for g, v in poolingStatistics.items():
                for r in v:
                    if r is None:
                        candidateFeature += [None] * 11
                        continue
                    poolingCandidateFeature = [r.getTop1Ratio(name), r.getTopkRatio(name), r.getMRR(name), *r.getMaxScoreAndMargin(name), *r.getSumScoreAndMargin(name), *r.getAvgScoreAndMargin(name), r.getMinScore(name), r.avgMargin]
                    candidateFeature += poolingCandidateFeature

            candidateFeatures.append(candidateFeature)

        return list(candidates.values()), candidateFeatures            


    def train(self):
        samples = trainUtils.loadTrainsetSamples(self.trainset, self.evalMethod)
        samplesWithGT:list[Sample] = []
        for sample in samples:
            stdNode:TaxoNode = sample.info["stdResult"]
            if (stdNode is None or stdNode.ICTVNode is None):
                continue
            samplesWithGT.append(sample)

        features, candidates, candidateRanges = self.getFeatures(samplesWithGT)


        labels = []
        # proteinsWithGT = []
        GTs = []
        for sample in samplesWithGT:
            # proteinsWithGT += sample.proteins
            GTs += [sample.info["stdResult"].ICTVNode]


        for stdNode, (start, end) in zip(GTs, candidateRanges):
            for candidate in candidates[start:end]:
                lca = taxoTree.ICTVTree.findLCA([stdNode, candidate.ICTVNode])
                labels.append(config.rankLevels[lca.rank])

        # f = features.copy()
        # f["GT"] = labels
        # f["candidate"] = [n.ICTVName for n in candidates]
        # f.to_csv("working/EnsembleXGBLTR_train.csv")

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
            if (not os.path.exists(f"{config.modelRoot}/EnsembleXGBoostLTR/{mainModelName}")):
                mainModel = self.train()
            else:
                mainModel = self.clz()
                mainModel.load_model(f"{config.modelRoot}/EnsembleXGBoostLTR/{mainModelName}")
        return mainModel            

    def run(self, samples:list[Sample], keepProteinRes=False, **kwargs):
        mainModel = self.loadModel()
        features, candidateList, candidateRanges = self.getFeatures(samples)
        scores:list[float] = mainModel.predict(features)

        # for debug
        # f = features.copy()
        # f["result"] = scores
        # f["candidate"] = [n.ICTVName for n in candidateList]
        # f.to_csv("working/EnsembleXGBLTR.csv")


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
        os.makedirs(f"{config.modelRoot}/EnsembleXGBoostLTR", exist_ok=True)

        i = 0
        while True:
            mainXGBoostName = f"{config.modelRoot}/EnsembleXGBoostLTR/model_{i}-main.json"
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


class ModuleGroupResult:
    def __init__(self, module, proteins, k):
        self.candidates:dict[str, Candidate] = {}
        self.n_proteins = len(proteins)
        self.k = k

        margins = []

        for pi, protein in enumerate(proteins):
            res = protein.results[module.moduleName]
            if res is None:
                continue
            if len(res) > 1:
                margins.append(res[0].score - res[1].score)
            else:
                margins.append(res[0].score)

            for i, r in enumerate(res):
                n = r.node.ICTVName
                if n not in self.candidates:
                    self.candidates[n] = Candidate(n, self.n_proteins)
                self.candidates[n].addResult(pi, i, r)
        
        for candidate in self.candidates.values():
            candidate.calcAvgScore()

        # get max top2 candidates
        self.maxTop1:Candidate = None
        self.maxTop2:Candidate = None
        for candidate in self.candidates.values():
            if self.maxTop1 is None or candidate.maxScore > self.maxTop1.maxScore:
                self.maxTop2 = self.maxTop1
                self.maxTop1 = candidate
            elif self.maxTop2 is None or candidate.maxScore > self.maxTop2.maxScore:
                self.maxTop2 = candidate

        # get sum top2 candidates
        self.sumTop1:Candidate = None
        self.sumTop2:Candidate = None
        for candidate in self.candidates.values():
            if self.sumTop1 is None or candidate.sumScore > self.sumTop1.sumScore:
                self.sumTop2 = self.sumTop1
                self.sumTop1 = candidate
            elif self.sumTop2 is None or candidate.sumScore > self.sumTop2.sumScore:
                self.sumTop2 = candidate

        # get avg top2 candidates
        self.avgTop1:Candidate = None
        self.avgTop2:Candidate = None
        for candidate in self.candidates.values():
            if self.avgTop1 is None or candidate.avgScore > self.avgTop1.avgScore:
                self.avgTop2 = self.avgTop1
                self.avgTop1 = candidate
            elif self.avgTop2 is None or candidate.avgScore > self.avgTop2.avgScore:
                self.avgTop2 = candidate

        self.avgMargin = sum(margins) / len(margins)

    def getTop1Ratio(self, candidateName):
        if candidateName not in self.candidates:
            return None
        candidate = self.candidates[candidateName]
        return numpy.sum(candidate.tops == 1) / self.n_proteins
    
    def getTopkRatio(self, candidateName):
        if candidateName not in self.candidates:
            return None
        candidate = self.candidates[candidateName]
        return numpy.sum(candidate.tops <= self.k) / self.n_proteins
    
    def getMRR(self, candidateName):
        if candidateName not in self.candidates:
            return None
        candidate = self.candidates[candidateName]
        return numpy.average(1 / candidate.tops)

    def getMaxScoreAndMargin(self, candidateName):
        if candidateName not in self.candidates:
            return None, None
        candidate = self.candidates[candidateName]
        if candidate.name == self.maxTop1.name:
            if self.maxTop2 is not None:
                margin = candidate.maxScore - self.maxTop2.maxScore
            else:
                margin = candidate.maxScore
        else:
            margin = candidate.maxScore - self.maxTop1.maxScore

        return candidate.maxScore, margin
    
    def getSumScoreAndMargin(self, candidateName):
        if candidateName not in self.candidates:
            return None, None
        candidate = self.candidates[candidateName]
        if candidate.name == self.sumTop1.name:
            if self.sumTop2 is not None:
                margin = candidate.sumScore - self.sumTop2.sumScore
            else:
                margin = candidate.sumScore
        else:
            margin = candidate.sumScore - self.sumTop1.sumScore

        return candidate.sumScore, margin
    
    def getAvgScoreAndMargin(self, candidateName):
        if candidateName not in self.candidates:
            return None, None
        candidate = self.candidates[candidateName]
        if candidate.name == self.avgTop1.name:
            if self.avgTop2 is not None:
                margin = candidate.avgScore - self.avgTop2.avgScore
            else:
                margin = candidate.avgScore
        else:
            margin = candidate.avgScore - self.avgTop1.avgScore

        return candidate.avgScore, margin
    
    def getMinScore(self, candidateName):
        if candidateName not in self.candidates:
            return None
        candidate = self.candidates[candidateName]
        return candidate.minScore

class Candidate:
    def __init__(self, name, n_proteins):
        self.name = name
        self.maxScore = 0
        self.avgScores = numpy.zeros(n_proteins)
        self.avgScore = 0
        self.minScore = numpy.inf
        self.sumScore = 0
        self.tops = numpy.full(n_proteins, numpy.inf)

    def addResult(self, proteinIndex:int, resultIndex:int, r:Result):
        maxScore = r.info.get("maxScore", r.score)
        sumScore = r.info.get("sumScore", r.score)
        minScore = r.info.get("minScore", r.score)
        self.tops[proteinIndex] = resultIndex + 1
        if maxScore > self.maxScore:
            self.maxScore = maxScore
        
        if minScore < self.minScore:
            self.minScore = minScore

        self.sumScore += sumScore
        self.avgScores[proteinIndex] = maxScore
    
    def calcAvgScore(self):
        self.avgScore = numpy.average(self.avgScores)


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


def groupID2candidateRanges(groupIDs):
    ranges = []
    start = 0

    for i in range(1, len(groupIDs)):
        if groupIDs[i] != groupIDs[i - 1]:
            ranges.append((start, i))
            start = i

    ranges.append((start, len(groupIDs)))
    return ranges