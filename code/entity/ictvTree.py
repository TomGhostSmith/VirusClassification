import os
import sys
import json
import pandas
from tqdm import tqdm
from anytree import Node

sys.path.append("..")
from config import config

class ICTVTree():
    # scpre: all, bacteria, virus, archaea
    def __init__(self) -> None:
        # self.taxoFile = f"{config.modelRoot}/VMRv4/VMRv4_names.json"
        self.taxoFile = f"{config.modelRoot}/VMRv4/VMR_MSL39.v4_20241106.csv"

        self.ranks = dict()    # key: label  value: rank (e.g. species, genus, etc.)
        self.nodes = dict()    # key: label  value: node on the tree
        self.species = dict()  # key: species ID  value: node on the tree
        self.root = Node("Viruses", rank="superkingdom")
        self.nodes["Viruses"] = self.root
        self.name2ID = dict()
        self.ID2name = dict()

        self.accession2ID = dict()  # key: accession  value: ID
        self.ID2accession = dict()  # key: ID  value: a list of accessions

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
        df = pandas.read_csv(self.taxoFile)
        synonyms = dict()   # key: name. value: ID
        for _, row in df.iterrows():
            accession = row["Virus GENBANK accession"]
            ID = row["Species Sort"]
            self.accession2ID[accession] = ID
            if ID not in self.ID2accession:
                self.ID2accession[ID] = [accession]
            else:
                self.ID2accession[ID].append(accession)
            if (ID not in self.species):
                parentNode = self.root
                for rank in self.rankLevels:
                    pathName = row[rank.capitalize()]
                
                    if (pandas.isna(pathName)):
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

                # extract the synonyms
                syn = row["Virus name(s)"]
                if syn not in synonyms:
                    synonyms[syn] = {ID}
                else:
                    synonyms[syn].add(ID)
        
        # update synonyms to name2ID when the synonym is unique; otherwise we use the LCA of them
        for syn, IDs in synonyms.items():
            if len(IDs) == 1:
                id = next(iter(IDs))
                self.nodes[syn] = self.species[id]
                self.name2ID[syn] = id
            else:
                nodes = [self.species[id] for id in IDs]
                lca = self.findLCA(nodes)
                self.nodes[syn] = lca

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
    
    def findLCA(self, nodes):  # nodes are a list of nodes, return the LCA node
        nodes =  list(nodes)
        lca = nodes[0]
        for n in nodes[1:]:
            ancestors = set(lca.path)
            for an in reversed(n.path):
                if an in ancestors:
                    lca = an
                    break
        return lca