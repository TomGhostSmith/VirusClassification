from prototype.module import Module
from utils import trainUtils
from entity.taxoNode import TaxoNode
from entity.taxoTree import taxoTree
from moduleResult.plainResult import PlainResult
from config import config
from entity.sample import Sample
from utils.NucleotideUtils import NucleotideUtils

from sklearn.model_selection import GridSearchCV
import xgboost
import pandas
import json
import os

# this module:
# for each protein, using XGBoostLTR to get its prediction, and contig result is voted from the protein results

class ProteinXGBoostLTR(Module):
    def __init__(self, trainset, evalMethod, modules:list[Module], featureModules:list[Module], contigFeatures:list[str], proteinFeatures:list[str], tops:int, candidateFeatures:list[str]=[], pooling="vote", limitOutput=True, complete="no", excludeNoHit=False):
        moduleNames = "+".join([module.moduleName for module in modules])
        contigFeatureNames = "+".join(contigFeatures)
        proteinFeatureNames = "+".join(proteinFeatures)
        candidateFeatureNames = "+".join(candidateFeatures)
        self.candidateFeatures = candidateFeatures
        if (pooling not in ["sum", "vote"] and not pooling.startswith("top")):
            raise ValueError("Unsupported pooling method")
        self.pooling = pooling
        self.baseName = f"ProteinXGBoostLTR-train={trainset};eval={evalMethod};modules={moduleNames};contigFeatures={contigFeatureNames};proteinFeatures={proteinFeatureNames};tops={tops};candidateFeatures={candidateFeatureNames};complete={complete}"
        super().__init__(f"{self.baseName};limitOutput={limitOutput};pooling={pooling};excludeNoHit={excludeNoHit}")
        self.excludeNoHit = excludeNoHit
        if (self.excludeNoHit and "proteinAlignments" not in proteinFeatures):
            raise ValueError("Please include proteinAlignments to exclude no hit proteins")
        self.trainset = trainset
        self.evalMethod = evalMethod
        self.modules = modules
        self.featureModules = featureModules
        self.modules = modules
        self.contigFeatures = contigFeatures
        self.proteinFeatures = proteinFeatures
        self.tops = tops
        self.limitOutput = limitOutput

        if (complete not in ["no", "genus", "topdown"]):
            raise ValueError("Unsupported completion method")
    
        self.complete = complete

        self.modelListFile = f"{config.modelRoot}/ProteinXGBoostLTR/names.json"
        self.moduleListMap = {"nextOffset": 0}

    def getFeatures(self, samples:list[Sample]):
        # basic feature independent on all the model
        for sample in samples:
            sample.info["length"] = sample.length
            sample.info["proteinCount"] = len(sample.proteins)
            for protein in sample.proteins:
                protein.info["proteinLength"] = protein.length
                protein.info["proteinIndex"] = protein.index

        name2ID:dict[str, int] = {n: i for i, n in enumerate(taxoTree.taxaNames["genus"])}

        features = {}
        candidateFeatures = []
        candidateRanges:list[tuple[int, int]] = []
        candidateList:list[TaxoNode] = []

        genusLevel = config.rankLevels["genus"]

        titles = self.contigFeatures + self.proteinFeatures  # to avoid change the list "self.features"
        for i in range(len(self.modules)):
            titles.append(f"model_{i} rank")
            titles.append(f"model_{i} score")
        titles += self.candidateFeatures
        # for f in self.features:
        #     results = [sample.info.get(f) for sample in samples]
        #     features[f] = results

        candidateFeatureSet = set(self.candidateFeatures)


        for sample in samples:
            contigBasicFeatures = [sample.info.get(f) for f in self.contigFeatures]
            for protein in sample.proteins:
                rankMaps:list[dict[str, int]] = []
                scoreMaps:list[dict[str, float]] = []
                fullScoreLists:list[list[float]] = []
                candidateInfos:dict[str, dict] = {}
                proteinBasicFeatures = [protein.info.get(f) for f in self.proteinFeatures]

                # get all top x results from all modules
                # calculate the union set of all the potential predictions
                candidates:dict[str, TaxoNode] = {}
                for module in self.modules:
                    rankMap:dict[str, int] = {}
                    scoreMap:dict[str, float] = {}
                    res = protein.results[module.moduleName]
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
                    fullScoreLists.append(protein.info.get(f"{module.moduleName}_prob"))


                # record the start/end candidate into a list
                start = len(candidateList)
                end = len(candidateList) + len(candidates)
                candidateRanges.append((start, end))

                # for each candidate option, generate its features
                for name, node in candidates.items():
                    candidateFeature = contigBasicFeatures + proteinBasicFeatures
                    genusName = name
                    if self.complete == "topdown" and config.rankLevels[node.ICTVNode.rank] > genusLevel:
                        for n in node.ICTVNode.path:
                            if n.rank == "genus":
                                genusName = n.name
                                break
                    for rankMap, scoreMap, fullScoreList in zip(rankMaps, scoreMaps, fullScoreLists):
                        if (name in rankMap):
                            candidateFeature.append(rankMap[name])
                            candidateFeature.append(scoreMap.get(name))
                        elif self.complete != "no" and fullScoreList is not None and genusName in name2ID: # check the whole score map
                            score = fullScoreList[name2ID[genusName]]
                            rank = sum(x > score for x in fullScoreList)
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
        NucleotideUtils.extractProtein(samples)

        for module in self.featureModules:
            module.getResults(samples, keepProb=True, withMeta=True, withProteinMeta=True, withProteinCandidateMeta=True, keepProteinRes=True)
        
        for module in self.modules:
            module.getResults(samples, keepProb=True, withMeta=True, withProteinMeta=True, withProteinCandidateMeta=True, keepProteinRes=True)

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
            GTs += [sample.info["stdResult"].ICTVNode] * len(sample.proteins)

        for stdNode, (start, end) in zip(GTs, candidateRanges):
            for candidate in candidates[start:end]:
                lca = taxoTree.ICTVTree.findLCA([stdNode, candidate.ICTVNode])
                labels.append(config.rankLevels[lca.rank])

        mainXGBoostName = f"model{self.moduleListMap['nextOffset']}-main.json"
        
        mainModel = trainMainXGBoost(features, labels, mainXGBoostName)
        
        self.moduleListMap[self.baseName] = mainXGBoostName
        self.moduleListMap["nextOffset"] += 1

        with open(self.modelListFile, 'wt') as fp:
            json.dump(self.moduleListMap, fp, indent=2)

        return mainModel

    def loadModel(self):
        if self.baseName not in self.moduleListMap:
            mainModel = self.train()
        else:
            mainModelName = self.moduleListMap[self.baseName]
            if (not os.path.exists(f"{config.modelRoot}/ProteinXGBoostLTR/{mainModelName}")):
                mainModel = self.train()
            else:
                mainModel = xgboost.XGBRegressor()
                mainModel.load_model(f"{config.modelRoot}/ProteinXGBoostLTR/{mainModelName}")
        return mainModel            

    def run(self, samples:list[Sample], keepProteinRes=False, **kwargs):
        if (os.path.exists(self.modelListFile)):
            with open(self.modelListFile) as fp:
                self.moduleListMap = json.load(fp)

        NucleotideUtils.extractProtein(samples)
        for module in self.featureModules:
            module.getResults(samples, keepProb=True, withMeta=True, withProteinMeta=True, withProteinCandidateMeta=True, keepProteinRes=True)
        for module in self.modules:
            module.getResults(samples, keepProb=True, withMeta=True, withProteinMeta=True, withProteinCandidateMeta=True, keepProteinRes=True)
        mainModel = self.loadModel()
        features, candidateList, candidateRanges = self.getFeatures(samples)
        scores:list[float] = mainModel.predict(features)

        # for debug
        f = features.copy()
        s = 0
        proteinNames = []
        for sample in samples:
            for protein, (start, end) in zip(sample.proteins, candidateRanges[s:s+len(sample.proteins)]):
                proteinNames += [protein.id]*(end - start)
            
            s += len(sample.proteins)
            
        f["result"] = scores
        f["candidate"] = [n.ICTVName for n in candidateList]
        f["protein"] = proteinNames
        f.to_csv("working/ProteinXGBLTR.csv")

        results = []
        s = 0
        for sample in samples:
            results.append(self.getResult(sample, [candidateList[start:end] for start, end in candidateRanges[s:s+len(sample.proteins)]], [scores[start:end] for start, end in candidateRanges[s:s+len(sample.proteins)]], keepProteinRes))
            s += len(sample.proteins)

        return results
    
    def getResult(self, sample:Sample, candidateLists:list[list[TaxoNode]], scoreLists:list[list[float]], keepProteinRes):
        votes:dict[str, int] = {}
        if (self.pooling == "vote"):
            for protein, candidates, scores in zip(sample.proteins, candidateLists, scoreLists):
                if self.excludeNoHit and protein.info["proteinAlignments"] == 0:
                    continue
                res:list[PlainResult] = self.getProteinResult(protein, candidates, scores)
                if (res):
                    name = res[0].pred
                    if name in votes:
                        votes[name] += 1
                    else:
                        votes[name] = 1
                if (keepProteinRes):
                    protein.results[self.moduleName] = res
        elif (self.pooling == "sum" or self.pooling.startswith("top")):
            if (self.pooling.startswith("top")):
                thresh = int(self.pooling[3:])
            else:
                thresh = None
            for protein, candidates, scores in zip(sample.proteins, candidateLists, scoreLists):
                if self.excludeNoHit and protein.info["proteinAlignments"] == 0:
                    continue
                res:list[PlainResult] = self.getProteinResult(protein, candidates, scores)
                if (res):
                    for r in res[:thresh]:
                        name = r.pred
                        if name in votes:
                            votes[name] += 1
                        else:
                            votes[name] = 1
                if (keepProteinRes):
                    protein.addResult(self.moduleName, res)

        results = []
        if len(votes) > 0:
            totalVotes = sum(votes.values())
            vs = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            for name, v in vs:
                r = PlainResult(name, score=v/totalVotes)
                results.append(r)

        else:
            results = None
        
        return results

    
    def getProteinResult(self, protein, candidates:list[TaxoNode], scores:list[float]):
        speciesRank = config.rankLevels["species"]
        results:list[PlainResult] = []
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
        

def trainMainXGBoost(features, targets, saveFile):
    param_grid = {
        "max_depth": [3, 4, 6, 8, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "n_estimators": [50, 100, 200, 500]
    }

    grid = GridSearchCV(
        estimator=xgboost.XGBRegressor(
            random_state=42,
            learning_rate=0.05,
            objective="reg:squarederror",
            eval_metric="rmse"
        ),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(features, targets)

    best_model = grid.best_estimator_

    # Save to file
    os.makedirs(f"{config.modelRoot}/ProteinXGBoostLTR", exist_ok=True)
    best_model.save_model(f"{config.modelRoot}/ProteinXGBoostLTR/{saveFile}")   # JSON is human-readable
    return best_model
