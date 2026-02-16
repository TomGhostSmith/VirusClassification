from prototype.module import Module
from utils import trainUtils
from entity.taxoNode import TaxoNode
from entity.taxoTree import taxoTree
from prototype.result import Result
from config import config

from sklearn.model_selection import GridSearchCV
import xgboost
import pandas
import json
import os

class XGBoostRouter(Module):
    def __init__(self, trainset, evalMethod, modules:list[Module], featureModules:list[Module], features:list[str]):
        moduleNames = "+".join([module.moduleName for module in modules])
        featureNames = "+".join(features)
        super().__init__(f"XGBoostRouter-train={trainset};eval={evalMethod};modules={moduleNames};features={featureNames}")
        self.trainset = trainset
        self.evalMethod = evalMethod
        self.modules = modules
        self.featureModules = featureModules
        self.modules = modules
        self.features = features

        self.modelListFile = f"{config.modelRoot}/XGBoostRouter/names.json"
        self.moduleListMap = {"nextOffset": 0}

    def getFeatures(self, samples):
        features = {}
        for f in self.features:
            results = [sample.info.get(f) for sample in samples]
            features[f] = results
        features = pandas.DataFrame(features).astype(dtype=float)
        return features

    def train(self):
        samples = trainUtils.loadTrainsetSamples(self.trainset, self.evalMethod)

        for module in self.featureModules:
            module.getResults(samples, withMeta=True)
        
        for module in self.modules:
            module.getResults(samples)
        
        # get best modules and corresponding best ranks
        bestModules = []
        targetRanks = []
        for sample in samples:
            bestModule = 0
            bestRank = 0
            stdNode:TaxoNode = sample.info["stdResult"]
            if (stdNode is None or stdNode.ICTVNode is None):
                continue
            for idx, module in enumerate(self.modules):
                pred = sample.results[module.moduleName]
                if (pred is None or pred[0].node.ICTVNode is None):
                    continue
                lca = taxoTree.ICTVTree.findLCA([stdNode.ICTVNode, pred[0].node.ICTVNode])
                rank = config.rankLevels[lca.rank]
                if (rank > bestRank):
                    bestRank = rank
                    bestModule = idx
            
            bestModules.append(bestModule)
            targetRanks.append(bestRank)

        # get features
        features = self.getFeatures(samples)

        mainXGBoostName = f"model{self.moduleListMap['nextOffset']}-main.json"
        rankXGBoostName = f"model{self.moduleListMap['nextOffset']}-rank.json"
        
        # train XGBoost1: learn which module to use
        mainModel = trainMainXGBoost(features, bestModules, len(self.modules), mainXGBoostName)
        
        # TODO: add a rank control XGBoost? should we also use the feature of the model?
        # rankModel = trainRankXGBoost(features, bestModules, config.rankLevels["species"] + 1, rankXGBoostName)

        self.moduleListMap[self.moduleName] = [mainXGBoostName, rankXGBoostName]
        self.moduleListMap["nextOffset"] += 1

        with open(self.modelListFile, 'wt') as fp:
            json.dump(self.moduleListMap, fp, indent=2)

        return mainModel

    def loadModel(self):
        if self.moduleName not in self.moduleListMap:
            mainModel = self.train()
        else:
            mainModelName, rankModelName = self.moduleListMap[self.moduleName]
            if (not os.path.exists(f"{config.modelRoot}/XGBoostRouter/{mainModelName}")):
                mainModel = self.train()
            # elif (not os.path.exists(f"{config.modelRoot}/XGBoostRouter/{rankModelName}")):
            #     mainModel = self.train()
            else:
                mainModel = xgboost.XGBClassifier()
                mainModel.load_model(f"{config.modelRoot}/XGBoostRouter/{mainModelName}")
        return mainModel            

    def run(self, samples, **kwargs):
        if (os.path.exists(self.modelListFile)):
            with open(self.modelListFile) as fp:
                self.moduleListMap = json.load(fp)
        for module in self.featureModules:
            module.getResults(samples, withMeta=True)
        mainModel = self.loadModel()
        features = self.getFeatures(samples)
        modelIndexes = runXGBoost(mainModel, features)

        samplesToRun = [[] for _ in range(len(self.modules))]
        for modelIndex, sample in zip(modelIndexes, samples):
            samplesToRun[modelIndex].append(sample)
        
        for module, sampleList in zip(self.modules, samplesToRun):
            if (sampleList):
                module.getResults(sampleList)

        results = []

        for modelIndex, sample in zip(modelIndexes, samples):
            results.append(sample.results[self.modules[modelIndex].moduleName])
        
        return results

def trainMainXGBoost(features, targets, n_class, saveFile):
    param_grid = {
        "max_depth": [3, 4, 6, 8, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "n_estimators": [50, 100, 200, 500]
    }

    grid = GridSearchCV(
        estimator=xgboost.XGBClassifier(
            random_state=42,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=n_class,
            eval_metric="mlogloss"
        ),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(features, targets)

    best_model = grid.best_estimator_

    # Save to file
    os.makedirs(f"{config.modelRoot}/XGBoostRouter", exist_ok=True)
    best_model.save_model(f"{config.modelRoot}/XGBoostRouter/{saveFile}")   # JSON is human-readable
    return best_model

def trainRankXGBoost(features, targets, n_class, saveFile):
    param_grid = {
        "max_depth": [3, 4, 6, 8, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "n_estimators": [50, 100, 200, 500]
    }

    grid = GridSearchCV(
        estimator=xgboost.XGBClassifier(
            random_state=42,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=n_class,
            eval_metric="mlogloss"
        ),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(features, targets)

    best_model = grid.best_estimator_

    # Save to file
    os.makedirs(f"{config.modelRoot}/XGBoostRouter", exist_ok=True)
    best_model.save_model(f"{config.modelRoot}/XGBoostRouter/{saveFile}")   # JSON is human-readable
    return best_model


def runXGBoost(model:xgboost.XGBClassifier, features:pandas.DataFrame):
    preds = model.predict_proba(features)
    index = preds.argmax(axis=1)

    # for debug
    # f = features.copy()
    # f["result"] = index
    # f.to_csv("working/XGBRouter.csv")
    
    return index