from prototype.moduleConfig import ModuleConfig

class MinimapThresholdConfig(ModuleConfig):
    def __init__(self):
        self.reference = 'VMRv4'
        # self.reference = 'VMRv4_fold2024'
        # self.reference = 'VMRv4_fold2020'

        # self.mode = 'simple'
        self.mode = 'ont'

        self.threads = 12

        self.skipComments = True

        self.factors = {
            "positive",
            "60",
            "completeMatch",
            "singleAlignment"
        }

    def update(self):
        self.name = f'minimapThresh-ref={self.reference};mode={self.mode};thresh-{"_".join(self.factors)}'
        self.baseName = f'minimap-ref={self.reference};mode={self.mode}'

    def getMinimapCommand(self, queryFile):
        referenceFasta = f"/Data/ICTVData/reference/{self.reference}/{self.reference}.fasta"
        minimapBase = "minimap2"   # if you cannot call minimap2 directly, use its path here
        if self.mode == 'ont':
            mode = "-ax map-ont"
        else:
            mode = "-a"
        thread = f"-t {self.threads}"
        if self.skipComments:
            postProcess = ' | grep -v "^@"'
        else:
            postProcess = ""
        command = f"{minimapBase} {mode} {referenceFasta} {queryFile} {thread} {postProcess}"
        return command