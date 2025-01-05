from prototype.mergeConfig import MergeConfig
from module.minimap import Minimap
from module.mlModule import MLModule

class MinimapMLMergeConfig(MergeConfig):
    def __init__(self, minimap:Minimap, mlModule:MLModule):
        self.minimapConfig = minimap.minimapConfig
        self.mlConfig = mlModule.mlConfig
        super().__init__([minimap, mlModule])
        self.factors = {
            "positive",
            "60",
            "completeMatch",
            "singleAlignment"
        }

    def update(self):
        self.name = f"{self.minimapConfig.name}.{self.mlConfig.name}.minimapML-{"_".join(self.factors)}"