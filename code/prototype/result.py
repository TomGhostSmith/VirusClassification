# base class for result
class Result():
    def __init__(self):
        self.node = None

    # this will be called after the result submited to the sample
    def calcTaxoNode(self):
        pass

