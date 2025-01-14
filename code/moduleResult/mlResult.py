from prototype.result import Result
from entity.taxoTree import taxoTree
from config import config

class MLResult(Result):
    def __init__(self, strategy, thresh):
        super().__init__()
        self.strategy = strategy
        self.thresh = thresh
        # self.nodes = [(taxoTree.ICTVTree.nodes["Viruses"], 0)]
        self.res = None
        self.scores = dict()
        # self.ictvNode = 
        self.suspendScore = 0
        self.suspendNode = None
        self.terminate = False


    def addResult(self, name, score):
        assert (name in taxoTree.ICTVTree.nodes)
        thisNode = taxoTree.ICTVTree.nodes[name]

        self.scores[thisNode.rank] = score
        if (self.strategy == 'topdown'):
            if (score >= self.thresh):
                self.res = thisNode
            elif self.res is not None:
                self.terminate=True
        elif (self.strategy == 'topdownExt'):
            if (score >= self.thresh):
                self.res = thisNode
            elif (self.res is not None and self.res in thisNode.path): # use the result if previous node is on the path of current node. Exception: realm
                self.res = thisNode
            elif (self.res is not None):
                self.terminate = True
        elif (self.strategy == 'topdownPreserve'):
            if (score >= self.thresh):
                self.res = thisNode
                self.suspendNode = None
            elif (self.suspendNode is None):
                self.suspendNode = thisNode
            elif (self.suspendNode in thisNode.path): # use the result if previous node is on the path of current node. Exception: realm
                self.res = self.suspendNode
                self.terminate = True  # we stop here because there are at least two rank that are below cutoff
            else:  # self.suspendNode is not None, but current node is not on the same path
                self.terminate=True
        elif (self.strategy == 'bottomup'):
            if score >= self.thresh:
                self.res = thisNode
                self.terminate = True
        elif (self.strategy == 'highest'):
            if score > self.score:
                self.res = thisNode

    def calcTaxoNode(self):
        if (self.node is None):
            if self.res is None:
                self.res = taxoTree.ICTVTree.nodes["Viruses"]
            
            self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=self.res)