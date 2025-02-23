import os
import sys
import json
from anytree import Node
from tqdm import tqdm

sys.path.append("..")
sys.path.append(".")
from config import config
from utils import IOUtils

class NCBITree():
    # scpre: all, bacteria, virus, archaea
    def __init__(self, scope) -> None:
        self.scope = scope
        self.nodesFile = f"{config.modelRoot}/NCBI/Assembly/taxdump/{scope}_nodes.dmp"
        self.nameFile = f"{config.modelRoot}/NCBI/Assembly/taxdump/{scope}_names.dmp"
        self.fnaNameFile = f"{config.modelRoot}/NCBI/Assembly/{scope}/names.json"
        self.fnaFolder = f"{config.modelRoot}/NCBI/Assembly/{scope}/fna"

        self.name2ID = dict()  # including synonym names
        self.ID2name = dict()  # only include scientific names

        self.children = dict()   # key: parent ID. value: a list of children. Useful for iteration the whole tree
        self.ranks = dict()  # key: ID   value: rank (e.g. species, genus, etc.)
        self.nodes = dict()  # key: ID   value: node on the tree
        # self.species = dict()  # key: NCBI genome file name   value: ID
        self.species = dict()  # key: ID  value: a list of (fileName, filePath)
        self.hosts = dict() # key: ID value: host IDs
        self.root = None

        # store lines from the dmp, to export subtree dmp file quickly
        self.nodeLines = dict()
        self.nameLines = dict()

        self.accession2ID = dict()  # key: accession  value: id
        self.ID2accession = dict()  # key: ID  value: a list of accessions


    def loadNodes(self):
        with open(self.nodesFile) as fp:
            rootLine = fp.readline()   # manually skip the root
            terms = rootLine[:-3].split("\t|\t")
            id = terms[0]
            rank = terms[2]
            self.rootID = id
            self.ranks[id] = rank
            self.nodeLines[id] = rootLine
            self.root = Node(self.rootID, attributes=None, rank=rank)
            self.nodes[self.rootID] = self.root

            for line in fp:
                terms = line[:-3].split('\t|\t')   # a line is ended with "\t | \n", so remove them first
                id = terms[0]
                parent = terms[1]
                rank = terms[2]
                self.ranks[id] = rank
                self.nodeLines[id] = line

                if (parent in self.children):
                    self.children[parent].append(id)
                else:
                    self.children[parent] = [id]
        
        # build a tree from the root
        self.loadSubTree(self.rootID)


    # assume that root is in the self.nodes
    def loadSubTree(self, rootID):
        if (rootID in self.children):
            children = self.children[rootID]
            for child in children:
                self.nodes[child] = Node(child, attributes=None, rank=self.ranks[child], parent=self.nodes[rootID])
                self.loadSubTree(child)

    def loadAnnotations(self):
        with open(self.nameFile) as fp:
            for line in fp:
                terms = line[:-3].split("\t|\t")
                id = terms[0]
                value = terms[1]
                key = terms[3]

                if (id in self.nameLines):
                    self.nameLines[id].append(line)
                else:
                    self.nameLines[id] = [line]

                if (key == "scientific name"):
                    self.name2ID[value] = id
                    self.ID2name[id] = value
                elif (key == "synonym" or key == "equivalent name"):
                    self.name2ID[value] = id

    def loadSpecies(self, mergeSubSpecies=True):
        with open(self.fnaNameFile) as fp:
            filenames = json.load(fp)
        for filename, id in filenames.items():
            path = f"{self.fnaFolder}/{filename}.fasta"
            if (mergeSubSpecies):
                if (id not in self.nodes):
                    IOUtils.showInfo(f"Sequence with taxID {id} has no taxo meta data. Skipped.")
                    continue
                node = self.nodes[id]
                speciesID = self.getSpeciesNode(node).name
            else:
                speciesID = id

            if speciesID in self.species:
                self.species[speciesID].append((filename, path))
            else:
                self.species[speciesID] = [(filename, path)]

    # actually it load data from allNucloData, which contains RefSeq + GenRank
    def loadAccession(self):
        id2AccessionFile = f"{config.cacheResultFolder}/NCBIID2Accesssion.json"
        accession2IDFile = f"{config.cacheResultFolder}/Accession2NCBIID.json"
        if (os.path.exists(id2AccessionFile) and os.path.exists(accession2IDFile)):
            with open(id2AccessionFile) as fp:
                self.ID2accession = json.load(fp)
            with open(accession2IDFile) as fp:
                self.accession2ID = json.load(fp)
        else:
            with open(f"{config.modelRoot}/NCBI/Nucleotide/genbank.accession") as fp:
                fp.readline()
                lines = fp.readlines()
            for line in tqdm(lines, desc="loading Accession"):
                comma = line.find(",")
                accession = line[:comma]
                species = line.strip()[comma + 1:]
                if (species in self.name2ID):
                    node = self.getSpeciesNode(self.nodes[self.name2ID[species]])
                    if (node is not None):
                        name = accession[:accession.index(".")]
                        self.accession2ID[name] = node.name
                        if (node.name not in self.ID2accession):
                            self.ID2accession[node.name] = [name]
                        else:
                            self.ID2accession[node.name].append(name)
            
            with open(id2AccessionFile, 'wt') as fp:
                json.dump(self.ID2accession, fp, indent=2, sort_keys=True)
            with open(accession2IDFile, 'wt') as fp:
                json.dump(self.accession2ID, fp, indent=2, sort_keys=True)
    
    def isValidTaxoNode(self, node):
        acceptableRanks = set(config.evaluationRanks)
        for n in node.path:
            if n.rank not in acceptableRanks:
                return False
        return True

    # assume root is exported
    def exportSubTree(self, rootID, nodeFP, nameFP):
        if (rootID in self.children):
            children = self.children[rootID]
            for child in children:
                nodeFP.write(self.nodeLines[child])
                nameFP.writelines(self.nameLines[child])
                self.exportSubTree(child, nodeFP, nameFP)
            
    def exportTree(self, scope):
        targetNodesFile = f"{config.ncbiAssemblyFolder}/taxdump/{scope}_nodes.dmp"
        targetNameFile = f"{config.ncbiAssemblyFolder}/taxdump/{scope}_names.dmp"

        rootID = self.name2ID[scope]
        nodeFP = open(targetNodesFile, 'wt')
        nameFP = open(targetNameFile, 'wt')

        nodeFP.write(self.nodeLines[rootID])
        nameFP.writelines(self.nameLines[rootID])
        self.exportSubTree(rootID, nodeFP, nameFP)

        nodeFP.close()
        nameFP.close()

    # usage: for nodes below species rank (e.g. subspecies, norank, etc.), use this function to get the corresponding species node
    # if the input node is above species rank, then return None
    def getSpeciesNode(self, node):
        result = None
        for n in reversed(node.path):
            if (n.rank == 'species'):
                result = n
                break
        return result

    def showNode(self, node):
        result = list()
        for n in node.path:
            name = self.ID2name[n.name]   # the name of NCBI node is its ID
            result.append((n.rank, n.name, name))
        return result
    
    def showChildren(self, node):
        result = list()
        for c in node.children:
            name = self.ID2name[c.name]
            result.append((c.name, name, c.rank))
        return  result


# ncbiTree = NCBITree("all")
# ncbiTree.loadNodes()
# ncbiTree.loadAnnotations()
# print("?")
# ncbiTree.exportTree("Viruses")
# ncbiTree.exportTree("Archaea")
# ncbiTree.exportTree("Bacteria")