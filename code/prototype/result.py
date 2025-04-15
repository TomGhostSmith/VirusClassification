# base class for result
from entity.taxoNode import TaxoNode
class Result():
    def __init__(self):
        self.node:TaxoNode = None

    # this will be called after the result submited to the sample
    def calcTaxoNode(self):
        pass

