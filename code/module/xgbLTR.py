from prototype.module import Module
from utils import trainUtils
from entity.taxoNode import TaxoNode
from entity.taxoTree import taxoTree
from moduleResult.plainResult import PlainResult
from config import config
from entity.sample import Sample

from sklearn.model_selection import GridSearchCV
import xgboost
import pandas
import json
import os

class XGBoostLTR(Module):
    def __init__(self, trainset, evalMethod, modules:list[Module], featureModules:list[Module], features:list[str], tops:int, limitOutput=True):
        moduleNames = "+".join([module.moduleName for module in modules])
        featureNames = "+".join(features)
        self.baseName = f"XGBoostLTR-train={trainset};modules={moduleNames};features={featureNames};tops={tops}"
        super().__init__(f"{self.baseName};limitOutput={limitOutput}")
        self.trainset = trainset
        self.evalMethod = evalMethod
        self.modules = modules
        self.featureModules = featureModules
        self.modules = modules
        self.features = features
        self.tops = tops
        self.limitOutput = limitOutput

        self.modelListFile = f"{config.modelRoot}/XGBoostLTR/names.json"
        self.moduleListMap = {"nextOffset": 0}
        if (os.path.exists(self.modelListFile)):
            with open(self.modelListFile) as fp:
                self.moduleListMap = json.load(fp)

    def getFeatures(self, samples:list[Sample]):
        features = {}
        candidateFeatures = []
        candidateRanges:list[tuple[int, int]] = []
        candidateList:list[TaxoNode] = []

        titles = self.features.copy()  # to avoid change the list "self.features"
        for i in range(len(self.modules)):
            titles.append(f"model_{i} rank")
            titles.append(f"model_{i} score")
        # for f in self.features:
        #     results = [sample.info.get(f) for sample in samples]
        #     features[f] = results


        for sample in samples:
            rankMaps = []
            scoreMaps = []
            basicFeatures = [sample.info.get(f) for f in self.features]

            # get all top x results from all modules
            # calculate the union set of all the potential predictions
            candidates:set[TaxoNode] = set()
            for module in self.modules:
                rankMap = {}
                scoreMap = {}
                res = sample.results[module.moduleName]
                if (res is not None):
                    for idx, r in enumerate(res[:self.tops]):
                        if r.node is None:
                            continue
                        candidates.add(r.node)
                        rankMap[r.node] = idx
                        scoreMap[r.node] = r.score
                rankMaps.append(rankMap)
                scoreMaps.append(scoreMap)


            # record the start/end candidate into a list
            start = len(candidateList)
            end = len(candidateList) + len(candidates)
            candidateRanges.append((start, end))

            # for each candidate option, generate its features
            for c in candidates:
                candidateFeature = basicFeatures.copy()
                for rankMap, scoreMap in zip(rankMaps, scoreMaps):
                    candidateFeature.append(rankMap.get(c))
                    candidateFeature.append(scoreMap.get(c))
                candidateFeatures.append(candidateFeature)
                candidateList.append(c)
                
        features = {k: v for k, v in zip(titles, zip(*candidateFeatures))}

        return pandas.DataFrame(features).astype(dtype=float), candidateList, candidateRanges

    def train(self):
        samples = trainUtils.loadTrainsetSamples(self.trainset, self.evalMethod)

        for module in self.featureModules:
            module.getResults(samples)
        
        for module in self.modules:
            module.getResults(samples)

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

        mainXGBoostName = f"model{self.moduleListMap['nextOffset']}-main.json"
        
        # train XGBoost1: learn which module to use
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
            if (not os.path.exists(f"{config.modelRoot}/XGBoostLTR/{mainModelName}")):
                mainModel = self.train()
            else:
                mainModel = xgboost.XGBRegressor()
                mainModel.load_model(f"{config.modelRoot}/XGBoostLTR/{mainModelName}")
        return mainModel            

    def run(self, samples):
        for module in self.featureModules:
            module.getResults(samples)
        for module in self.modules:
            module.getResults(samples)
        mainModel = self.loadModel()  
        features, candidateList, candidateRanges = self.getFeatures(samples)
        scores:list[float] = mainModel.predict(features)

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
    os.makedirs(f"{config.modelRoot}/XGBoostLTR", exist_ok=True)
    best_model.save_model(f"{config.modelRoot}/XGBoostLTR/{saveFile}")   # JSON is human-readable
    return best_model
