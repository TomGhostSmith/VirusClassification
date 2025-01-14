import os
import sys
import json
from tqdm import tqdm
from anytree import Node

sys.path.append("..")
from config import config

class ICTVTree():
    # scpre: all, bacteria, virus, archaea
    def __init__(self) -> None:
        self.taxoFile = f"{config.refFolder}/VMRv4/VMRv4_names.json"

        self.ranks = dict()    # key: label  value: rank (e.g. species, genus, etc.)
        self.nodes = dict()    # key: label  value: node on the tree
        self.species = dict()  # key: species ID  value: node on the tree
        self.root = Node("Viruses", rank="superkingdom")
        self.nodes["Viruses"] = self.root
        self.name2ID = dict()
        self.ID2name = dict()

        self.rankLevels = [
            "realm",     # the clade in NCBI tree is equal to the realm in the ICTV tree
            "subrealm",
            "kingdom",
            "subkingdom",
            "phylum",
            "subphylum",
            "class",
            "subclass",
            "order",
            "suborder",
            "family",
            "subfamily",
            "genus",
            "subgenus",
            "species"
        ]

    def loadNodes(self):
        with open(self.taxoFile) as fp:
            content = json.load(fp)
        # for ID, path in tqdm(content.items(), desc="loading ICTV Tree"):
        for ID, path in content.items():
            parentNode = self.root
            terms = path.split('|')
            pathNames = terms[:-2]  # last one is the ID, last two is the common name
            assert(len(pathNames) == len(self.rankLevels))
            for pathName, rank in zip(pathNames, self.rankLevels):
                if (pathName == "Unknown"):
                    continue
                if (pathName in self.nodes):
                    assert(self.nodes[pathName].parent == parentNode)
                    parentNode = self.nodes[pathName]
                else:
                    self.nodes[pathName] = Node(pathName, rank=rank, parent=parentNode)
                    parentNode = self.nodes[pathName]
            self.species[ID] = parentNode
            self.ID2name[ID] = parentNode.name
            self.name2ID[parentNode.name] = ID

            # add synonyms, which is at terms[-2]
            self.nodes[terms[-2]] = parentNode
            self.name2ID[terms[-2]] = ID

    def showNode(self, node):
        result = list()
        for n in node.path:
            if n.name in self.name2ID:
                id = self.name2ID[n.name]
            else:
                id = ' * '
            result.append((n.rank, id, n.name))
        return result
    
    def showChildren(self, node):
        result = list()
        for c in node.children:
            if (c in self.name2ID):
                id = self.name2ID[c]
            else:
                id =' * '
            result.append((id, c.name, c.rank))
        return  result