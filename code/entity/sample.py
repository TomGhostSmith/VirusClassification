from prototype.result import Result

class Sample():
    def __init__(self, query, isATCG, length):
        self.query = query
        self.isATCG = isATCG
        self.length = length
        self.results = dict()
        self.info = dict()

    def addResult(self, name:str, result:Result):
        self.results[name] = result

    def addInfo(self, key, value):
        self.info[key] = value