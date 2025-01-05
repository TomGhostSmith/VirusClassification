from prototype.moduleConfig import ModuleConfig

class MLConfig(ModuleConfig):
    def __init__(self):
        self.shortname = "t33_512"
        # self.shortname = "family_finetune_t33_256"


    def update(self):
        self.name = f'ML-param=contig_most_frequent_{self.shortname}'