from prototype.result import Result
from entity.taxoTree import taxoTree
from config import config

class MLResult(Result):
    def __init__(self, strategy, thresh):
        super().__init__()
        self.strategy = strategy
        self.thresh = thresh
        self.nodes = [(taxoTree.ICTVTree.nodes["Viruses"], 0)]
        # self.ictvNode = 
        # self.score = 0
        # self.suspendScore = 0
        # self.suspendNode = None


    def addResult(self, name, score):
        assert (name in taxoTree.ICTVTree.nodes)
        thisNode = taxoTree.ICTVTree.nodes[name]
        # assert (thisNode.rank == rank)
        # lastRank = self.ictvNode.rank

        self.nodes.append((thisNode, score))

    def calcTaxoNode(self):
        if (self.node is None):
            res = taxoTree.ICTVTree.nodes["Viruses"]
            if self.strategy == 'topdown':
                for node, score in self.nodes:
                    if (score >= self.thresh):
                        res = node
                        self.score = score

            elif self.strategy == 'topdownExt':
                lastNode = None
                for node, score in self.nodes:
                    if (score >= self.thresh):
                        res = node
                        self.score = score
                        lastNode = node
                    elif (lastNode is not None and lastNode in node.path): # use the result if previous node is on the path of current node. Exception: realm
                        res = node
                        self.score = score
                        lastNode = node


            elif self.strategy == 'topdownPreserve':
                suspendNode = None
                suspendScore = 0
                for node, score in self.nodes:
                    if (score >= self.thresh):
                        res = node
                        self.score = score
                        suspendNode = None
                    elif (suspendNode is not None and suspendNode in node.path): # use the result if previous node is on the path of current node. Exception: realm
                        res = suspendNode
                        self.score = suspendScore
                        suspendNode = None
                        suspendScore = 0
                    else:
                        suspendNode = node
                        suspendScore = score

            elif self.strategy == 'bottomup':
                for node, score in self.nodes:
                    if score >= self.thresh:
                        res = node
                        self.score = score
                        break
            elif self.strategy == 'highest':
                nodes = sorted(self.nodes, key=lambda t : t[1], reverse=True)
                res, self.score = nodes[0]
            
            self.node = taxoTree.getTaxoNodeFromNode(ICTVNode=res)


            # self.node = taxoTree.getTaxoNodeFromICTV(ICTVName=self.name)