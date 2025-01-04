class MLConfig():
    def __init__(self):
        self.shortname = "t33_512"
        # self.shortname = "family_finetune_t33_256"

        self.updateName()

    def updateName(self):
        self.name = f'contig_most_frequent_{self.shortname}'

mlConfig = MLConfig()