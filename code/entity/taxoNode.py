class TaxoNode():
    def __init__(self) -> None:
        self.ICTVID = None
        self.NCBIID = None
        self.ICTVName = None
        self.NCBIName = None
        self.ICTVNode = None
        self.NCBINode = None

        # note:
        # if there is no complete corresponding node, then find the most similar one. e.g. ICTVID, ICTVNode are the species-level, but the NCBIID, NCBIName, etc. are genus level

    def getICTVTaxoCSVStr(self) -> str:
        ranks = ["superkingdom", "realm", "kingdom", "phylum", "subphylum", "class", "subclass", "order", "suborder", "family", "subfamily", "genus", "subgenus", "species"]
        strs = {rank: "" for rank in ranks}
        for node in self.ICTVNode.path:
            strs[node.rank] = f'"{node.name}"'  # in case there are commas in the name
        title = f"{','.join(list(strs.keys()))}"
        content = f"{','.join(list(strs.values()))}"
        return title, content
    
    def getICTVTaxoPath(self) -> str:
        res = list()
        for node in self.ICTVNode.path:
            res.append(f"{node.rank}={node.name}")
        return "|".join(res)