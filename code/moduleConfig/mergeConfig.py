class MergeConfig():
    def __init__(self):
        self.factors = {
            "positive",
            "60",
            "completeMatch",
            "singleAlignment"
        }

    def updateName(self):
        self.name = ";".join(self.factors)

mergeConfig = MergeConfig()