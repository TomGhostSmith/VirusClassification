import os
import json
from tqdm import tqdm
from Bio import SeqIO
import random
from sortedcontainers import SortedList

from config import config
from utils import IOUtils
from entity.ictvTree import ICTVTree
from entity.ncbiTree import NCBITree
from entity.taxoNode import TaxoNode

class TaxoTree():
    def __init__(self) -> None:

        self.ICTVTree = ICTVTree()
        self.viralNCBITree = NCBITree('Viruses')
        # self.bacteriaNCBITree = NCBITree('Bacteria')
        # self.archaeaNCBITree = NCBITree('Archaea')

        IOUtils.showInfo('Loading ICTV Tree')
        self.ICTVTree.loadNodes()
        self.ICTVTree.loadAccession()

        IOUtils.showInfo('Loading Viral NCBI Tree')
        self.viralNCBITree.loadNodes()
        self.viralNCBITree.loadAnnotations()
        self.viralNCBITree.loadSpecies()
        self.viralNCBITree.loadAccession()
        # IOUtils.showInfo('Loading Bacteria NCBI Tree')
        # self.bacteriaNCBITree.loadNodes()
        # self.bacteriaNCBITree.loadAnnotations()
        # self.bacteriaNCBITree.loadSpecies()
        # IOUtils.showInfo('Loading Archaea NCBI Tree')
        # self.archaeaNCBITree.loadNodes()
        # self.archaeaNCBITree.loadAnnotations()
        # self.archaeaNCBITree.loadSpecies()

        self.ICTV2NCBI = dict()  # key: ICTV name   value: NCBI Node
        self.NCBI2ICTV = dict()  # key: NCBI ID   value: ICTV Node

        # self.loadHosts()
        self.matchTaxoTree()

        IOUtils.showInfo("Taxo tree loaded")


    def checkPath(self, ICTVNode, NCBINode):
        result = True
        for iNode, nNode in zip(ICTVNode.path, NCBINode.path):
            # note: NCBI node's name is its id
            nName = self.viralNCBITree.ID2name[nNode.name]
            if (iNode.name != nName or iNode.rank != nNode.rank):
                result = False
                break

        return result
    
    def loadHosts(self):
        # with open(f"{config.ncbiNucleotideFolder}/hosts.json") as fp:
        #     self.viralNCBITree.hosts = json.load(fp)

        filtedHosts = dict()
        with open(f"{config.ncbiNucleotideFolder}/rawHosts.json") as fp:
            hostDict = json.load(fp)
        for species, hosts in hostDict.items():
            if (species in self.viralNCBITree.name2ID):
                node = self.viralNCBITree.getSpeciesNode(self.viralNCBITree.nodes[self.viralNCBITree.name2ID[species]])
                if (node is not None):
                    id = node.name
                    for host in hosts:
                        if (host in self.bacteriaNCBITree.name2ID):
                            hostNode = self.bacteriaNCBITree.getSpeciesNode(self.bacteriaNCBITree.nodes[self.bacteriaNCBITree.name2ID[host]])
                            if (hostNode is not None):
                                if (id not in filtedHosts):
                                    filtedHosts[id] = set()
                                filtedHosts[id].add(hostNode.name)
                        elif (host in self.archaeaNCBITree.name2ID):
                            hostNode = self.archaeaNCBITree.getSpeciesNode(self.archaeaNCBITree.nodes[self.archaeaNCBITree.name2ID[host]])
                            if (hostNode is not None):
                                if (id not in filtedHosts):
                                    filtedHosts[id] = set()
                                filtedHosts[id].add(hostNode.name)
        
        for k, v in filtedHosts.items():
            self.viralNCBITree.hosts[k] = list(v)

    def matchTaxoTree(self):
        IOUtils.showInfo('Calculating ICTV and NCBI correspondence')

        # ICTV to NCBI
        for ICTVName, ICTVSpeciesNode in self.ICTVTree.nodes.items():
            targetNode = None
            ICTVSpeciesPath = ICTVSpeciesNode.path
            for ICTVNode in reversed(ICTVSpeciesPath):
                if (ICTVNode.name in self.viralNCBITree.name2ID):
                    NCBIID = self.viralNCBITree.name2ID[ICTVNode.name]
                    NCBINode = self.viralNCBITree.nodes[NCBIID]

                    # assert they have the same path
                    # however, after check, we found that there are many cases that have different taxo path in NCBI and ICTV
                    # assert self.checkPath(ICTVNode, NCBINode)
                    targetNode = NCBINode
                    break
            
            if (targetNode is None):
                ICTVID = self.ICTVTree.name2ID[ICTVName] if ICTVName in self.ICTVTree.name2ID else ""
                IOUtils.showInfo(f'ICTV Node {ICTVID} ({ICTVName}) has no correspondance node in NCBI tree!', 'ERROR')
            else:
                self.ICTV2NCBI[ICTVName] = targetNode
        
        # NCBI to ICTV
        # for NCBIID in self.viralNCBITree.species.keys():
        for NCBIID, NCBISpeciesNode in self.viralNCBITree.nodes.items():
            # NCBISpeciesNode = self.viralNCBITree.nodes[NCBIID]
            targetNode = None
            NCBISpeciesPath = NCBISpeciesNode.path
            for NCBINode in reversed(NCBISpeciesPath):
                ncbiName = self.viralNCBITree.ID2name[NCBINode.name]  # note: the name of NCBI node is its id
                if (ncbiName in self.ICTVTree.nodes):
                    ICTVNode = self.ICTVTree.nodes[ncbiName]

                    # assert they have the same path
                    # assert self.checkPath(ICTVNode, NCBINode)
                    targetNode = ICTVNode
                    break
            if (targetNode is None):
                IOUtils.showInfo(f"NCBI ID {NCBIID} ({NCBISpeciesNode.name}) has no correspondance node in ICTV tree!", "Error")
            else:
                self.NCBI2ICTV[NCBIID] = targetNode

    def getTaxoNodeFromNCBI(self, NCBIID=None, NCBIName=None):
        result = TaxoNode()

        # convert invalid ID or name to None
        if (NCBIName is not None and NCBIName not in self.viralNCBITree.name2ID):
            IOUtils.showInfo(f'NCBI Name {NCBIName} is invalid', 'WARN')
            NCBIName = None
        if (NCBIID is not None and NCBIID not in self.viralNCBITree.ID2name):
            IOUtils.showInfo(f'NCBI ID {NCBIID} is invalid', 'WARN')
            NCBIID = None

        if (NCBIID is None and NCBIName is None):
            return None
            # raise ValueError("NCBI ID and name are both invalid")
        
        if (NCBIID is not None and NCBIName is not None):
            # check ID and name are correspond
            if (self.viralNCBITree.name2ID[NCBIName] != NCBIID):
                raise ValueError(f"NCBI ID {NCBIID} and name {NCBIName} are not consistent!")
            result.NCBIID = NCBIID
            result.NCBIName = NCBIName
        elif (NCBIID is not None):
            result.NCBIID = NCBIID
            result.NCBIName = self.viralNCBITree.ID2name[NCBIID]
        else:  # NCBI Name is not None
            result.NCBIName = NCBIName
            result.NCBIID = self.viralNCBITree.name2ID[NCBIName]

        # make sure that NCBI ID is a id for species, otherwise it is not in the NCBI2ICTV dict
        if (result.NCBIID not in self.NCBI2ICTV):
            raise ValueError(f"NCBI ID {result.NCBIID} ({result.NCBIName}) is not a valid species that has corresponding ICTV Node!")
        
        # finish other information about the node
        result.NCBINode = self.viralNCBITree.nodes[result.NCBIID]
        result.ICTVNode = self.NCBI2ICTV[result.NCBIID]
        result.ICTVName = result.ICTVNode.name
        result.ICTVID = self.ICTVTree.name2ID.get(result.ICTVName)  # note: the ICTV name miight not be the species name, so there might be no ID for this

        return result

    def getTaxoNodeFromICTV(self, ICTVID=None, ICTVName=None):
        result = TaxoNode()

        # convert invalid ID or name to None
        # if (ICTVName is not None and ICTVName not in self.ICTVTree.name2ID):
        #     IOUtils.showInfo(f'ICTV name {ICTVName} is invalid or not a species name', 'WARN')
        #     ICTVName = None
        if (ICTVName is not None and ICTVName not in self.ICTVTree.nodes):
            IOUtils.showInfo(f'ICTV name {ICTVName} is invalid', 'WARN')
            ICTVName = None
        if (ICTVID is not None and ICTVID not in self.ICTVTree.ID2name):
            IOUtils.showInfo(f"ICTV ID {ICTVID} is invalid", 'WARN')
            ICTVID = None
        
        if (ICTVID is None and ICTVName is None):
            return None
            # raise ValueError(f"ICTV ID and name are both invalid")
        
        if (ICTVID is not None and ICTVName is not None):
            # check ID and name are correspond
            if (self.ICTVTree.name2ID[ICTVName] != ICTVID):
                raise ValueError(f"ICTV ID {ICTVID} and name {ICTVName} are not consistent!")
            result.ICTVID = ICTVID
            result.ICTVName = ICTVName
        elif (ICTVID is not None):
            result.ICTVID = ICTVID
            result.ICTVName = self.ICTVTree.ID2name[ICTVID]
        else: # ICTV name is not None
            result.ICTVName = ICTVName
            result.ICTVID = self.ICTVTree.name2ID[ICTVName] if ICTVName in self.ICTVTree.name2ID else None

        # no need to check if ICTVName is in ICTV2NCBI, because only species have IDs
        # finish other information about the node
        result.ICTVNode = self.ICTVTree.nodes[result.ICTVName]
        result.NCBINode = self.ICTV2NCBI[result.ICTVName]
        result.NCBIID = result.NCBINode.name   # yes, NCBI Node has ID as its name
        result.NCBIName = self.viralNCBITree.ID2name[result.NCBIID]

        return result
    
    def getTaxoNodeFromNode(self, ICTVNode=None, NCBINode=None):
        if (ICTVNode is None and NCBINode is None):
            return None
        
        if (ICTVNode is not None and NCBINode is not None):
            raise ValueError("Please do not use both node simultaneously")
        
        result = TaxoNode()
        if (ICTVNode is not None):
            result.ICTVNode = ICTVNode
            result.ICTVName = ICTVNode.name
            result.ICTVID = self.ICTVTree.name2ID.get(ICTVNode.name)# note: the ICTV name miight not be the species name, so there might be no ID for this
            result.NCBINode = self.ICTV2NCBI[result.ICTVName]
            result.NCBIID = result.NCBINode.name   # yes, NCBI Node has ID as its name
            result.NCBIName = self.viralNCBITree.ID2name[result.NCBIID]
        elif (NCBINode is not None):
            result.NCBINode = NCBINode
            result.NCBIID = NCBINode.name
            result.NCBIName = self.viralNCBITree.ID2name[NCBINode.name]
            result.ICTVNode = self.NCBI2ICTV[result.NCBIID]
            result.ICTVName = result.ICTVNode.name
            result.ICTVID = self.ICTVTree.name2ID.get(result.ICTVName)  # note: the ICTV name miight not be the species name, so there might be no ID for this
        
        return result

    def findLCA(self, node1:TaxoNode|None, node2:TaxoNode|None):
        if (node1 is None or node2 is None):
            return None, None
        NCBILCA = self.findLCAonNCBI(node1, node2)
        ICTVLCA = self.findLCAonICTV(node1, node2)

        return NCBILCA, ICTVLCA

    def findLCAonNCBI(self, node1:TaxoNode, node2:TaxoNode):
        ancestors = set(node1.NCBINode.path)
        for ancestor in reversed(node2.NCBINode.path):
            if (ancestor in ancestors):
                return ancestor
        return self.viralNCBITree.root

    def findLCAonICTV(self, node1:TaxoNode, node2:TaxoNode):
        ancestors = set(node1.ICTVNode.path)
        for ancestor in reversed(node2.ICTVNode.path):
            if (ancestor in ancestors):
                return ancestor
        return self.ICTVTree.root
    
    def generateTestset(self, withGenBank, seed, maxPerSpecies=float('inf'), version=""):
        if (withGenBank):
            self.viralNCBITree.loadGenBank()
        random.seed(seed)

        wrapLength = 100   # wrap per 100 bp in the output fasta file

        # create an output file
        datasetName = f"genbank_{seed}_{version}" if withGenBank else f"refseq_{seed}_{version}" 
        folderPath = f"/Data/ICTVData/dataset/{datasetName}"
        os.makedirs(folderPath)
        targetFP = open(f"{folderPath}/{datasetName}.fasta", 'wt')

        # load query lengths from the competition, so we can generate dataset with similar length distribution
        with open("/Data/ICTVData/temp/queryLengths.txt") as fp:
            lengths = [int(l.strip()) for l in fp]
        minLength = min(lengths)

        # step 1: select sequences
        selectedSequences = list()
        for speciesID, sequences in self.viralNCBITree.species.items():  # sequences is a list of tuple (name, path)
            if (len(sequences) <= maxPerSpecies):
                seqs = sequences
            else:
                seqs = random.sample(sequences, maxPerSpecies)
            for name, seq in seqs:
                selectedSequences.append(('Viruses', name, speciesID, seq))

        # step 2: randomly add some bacteria and archaea from the host
        nonVirusCount = round(len(selectedSequences) / 0.85 * 0.15)
        hosts = list()
        for hostList in self.viralNCBITree.hosts.values():
            hosts += hostList
        if nonVirusCount > len(hosts):
            selectedHosts = hosts
            # since host number are not sufficient, we add other bacteria and archaea
            # there are too few archaea, so we add all of them if possible
            if ((nonVirusCount - len(hosts)) // 2 > len(self.archaeaNCBITree.species)):
                selectedArchaeas = self.archaeaNCBITree.species.keys()
            else:
                selectedArchaeas = random.sample(list(self.archaeaNCBITree.species.keys()), (nonVirusCount - len(hosts)) // 2)
            for id in selectedArchaeas:
                name, seq = random.sample(self.archaeaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Archaea', name, id, seq))

            bacteriaCount = nonVirusCount - len(hosts) - len(selectedArchaeas)
            selectedBacteria = random.sample(list(self.bacteriaNCBITree.species.keys()), bacteriaCount)
            for id in selectedBacteria:
                name, seq = random.sample(self.bacteriaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Bacteria', name, id, seq))

        else:
            selectedHosts = random.sample(hosts, nonVirusCount)  # assume that 15% sequences are host and 85% sequences are virus

        for id in selectedHosts:
            if id in self.bacteriaNCBITree.species:
                name, seq = random.sample(self.bacteriaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Bacteria', name, id, seq))
            elif id in self.archaeaNCBITree.species:
                name, seq = random.sample(self.archaeaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Archaea', name, id, seq))

        # step 3: extract a subsequence
        # shuffle the selectedSequence first
        random.shuffle(selectedSequences)
        # when the length randomly selected longer than the sequence, store it there
        # if there is some length in this list, and it is smaller than the current sequence, then there is no need to generate a new length
        # in this way, we can generalyl keep the length distribution
        stackedLength = SortedList(list())
        for superkingdom, sequenceName, speciesID, filePath in tqdm(selectedSequences):
            seqs = list(SeqIO.parse(filePath, "fasta"))
            if (len(seqs) > 0):
                seq = str(seqs[0].seq)
            else:
                IOUtils.showInfo(f'{superkingdom} ID {speciesID} ({filePath}) has no sequence', 'ERROR')
                break
            l = 0
            maxBelowThresholdIndex = stackedLength.bisect_left(len(seq))
            if (maxBelowThresholdIndex > 0):  # there is some length in the stack that is smaller than the seq length
                l = stackedLength[maxBelowThresholdIndex - 1]
                stackedLength.pop(maxBelowThresholdIndex - 1)
            elif (len(seq) < 2*minLength):
                l = len(seq)
            else:
                while (l == 0):
                    thisLen = random.sample(lengths, 1)[0]
                    if (thisLen > len(seq)):
                        stackedLength.add(thisLen)
                    else:
                        l = thisLen
            
            start = random.randint(0, len(seq) - l)
            subseq = seq[start : start+l]

            targetFP.write(f">{superkingdom}|{sequenceName}|{speciesID}|{start}|{l}\n")
            for i in range(0, len(subseq), wrapLength):
                targetFP.write(f"{subseq[i:i+wrapLength]}\n")

            
        targetFP.close()
    
    # dataset: ICTV or NCBI
    def showNodeInfo(self, node, dataset, sortBy, showChild):
        IOUtils.showInfo(f'  - {dataset} taxo path:')
        if (dataset == 'ICTV'):
            res = self.ICTVTree.showNode(node)
            children = self.ICTVTree.showChildren(node)
        elif (dataset == 'NCBIViral'):
            res = self.viralNCBITree.showNode(node)
            children = self.viralNCBITree.showChildren(node)
        elif (dataset == 'NCBIArchaea'):
            res = self.archaeaNCBITree.showNode(node)
            children = self.archaeaNCBITree.showChildren(node)
        elif (dataset == 'NCBIBacteria'):
            res = self.bacteriaNCBITree.showNode(node)
            children = self.bacteriaNCBITree.showChildren(node)

        prefix = "    "
    
        for rank, id, name in res:
            IOUtils.showInfo(f"{prefix}- {rank}: {id} ({name})")
            prefix += "  "

        if (sortBy == 'rank'):
            children = sorted(children, key=lambda a : config.rankLevels[a[2]] if a[2] in config.rankLevels else float('inf'))
        elif (sortBy == 'name'):
            children = sorted(children, key=lambda a : a[1])

        if (showChild):

            for id, name, rank in children:
                IOUtils.showInfo(f"{prefix}+ [{rank}] {id} ({name})")
            if (len(children) == 0):
                IOUtils.showInfo(f"{prefix}(No child)")
        else:
            IOUtils.showInfo(f"{prefix}+ {len(children)} children")


        return children


    def checkNode(self, node, sortBy='rank', showChild=True):
        IOUtils.showInfo('========================')
        children1 = self.showNodeInfo(node.ICTVNode, 'ICTV', sortBy, showChild)
        IOUtils.showInfo('------------------------')
        children2 = self.showNodeInfo(node.NCBINode, 'NCBIViral', sortBy, showChild)

        if (showChild):
            child1Dict = {t[1]: t for t in children1}
            child1Keys = list(child1Dict.keys())

            child2Dict = {t[1]: t for t in children2}

            for key in child2Dict.keys():
                child1Dict.pop(key, None)
            for key in child1Keys:
                child2Dict.pop(key, None)

            IOUtils.showInfo('------------------------')
            IOUtils.showInfo('  - ICTV specific children:')
            for id, name, rank in child1Dict.values():
                IOUtils.showInfo(f"    + [{rank}] {id} ({name})")
            if (len(child1Dict) == 0):
                IOUtils.showInfo(f"    (None)")


            IOUtils.showInfo('------------------------')
            IOUtils.showInfo('  - NCBI specific children:')
            for id, name, rank in child2Dict.values():
                IOUtils.showInfo(f"    + [{rank}] {id} ({name})")
            if (len(child2Dict) == 0):
                IOUtils.showInfo(f"    (None)")

    def checkNCBINodeByName(self, name, sortBy='rank', showChild=True):
        IOUtils.showInfo('========================')
        if (name in self.viralNCBITree.name2ID):
            self.showNodeInfo(self.viralNCBITree.nodes[self.viralNCBITree.name2ID[name]], 'NCBIViral', sortBy, showChild)
        elif (name in self.archaeaNCBITree.name2ID):
            self.showNodeInfo(self.viralNCBITree.nodes[self.archaeaNCBITree.name2ID[name]], 'NCBIArchaea', sortBy, showChild)
        elif (name in self.bacteriaNCBITree.name2ID):
            self.showNodeInfo(self.viralNCBITree.nodes[self.bacteriaNCBITree.name2ID[name]], 'NCBIBacteria', sortBy, showChild)
        else:
            IOUtils.showInfo(f"{name} is not a valid name in viruses/bacteria/archaea NCBI tree", 'ERROR')

    def checkNCBINodeByID(self, id, sortBy='rank', showChild=True):
        IOUtils.showInfo('========================')
        if (id in self.viralNCBITree.nodes):
            self.showNodeInfo(self.viralNCBITree.nodes[id], 'NCBIViral', sortBy, showChild)
        elif (id in self.archaeaNCBITree.nodes):
            self.showNodeInfo(self.archaeaNCBITree.nodes[id], 'NCBIArchea', sortBy, showChild)
        elif (id in self.bacteriaNCBITree.nodes):
            self.showNodeInfo(self.bacteriaNCBITree.nodes[id], 'NCBIBacteria', sortBy, showChild)
        else:
            IOUtils.showInfo(f"{id} is not a valid ID in viruses/bacteria/archaea NCBI tree", 'ERROR')

    
    def checkICTVNodeByName(self, name, sortBy='rank', showChild=True):
        IOUtils.showInfo('========================')
        if (name in self.ICTVTree.nodes):
            self.showNodeInfo(self.ICTVTree.nodes[name], 'ICTV', sortBy, showChild)
        else:
            IOUtils.showInfo(f"{name} is not a valid name in ICTV tree", 'ERROR')

    def checkICTVNodeByID(self, id, sortBy='rank', showChild=True):
        IOUtils.showInfo('========================')
        if (id in self.ICTVTree.ID2name):
            self.showNodeInfo(self.ICTVTree.nodes[self.ICTVTree.ID2name[id]], 'ICTV', sortBy, showChild)
        else:
            IOUtils.showIfo(f"{id} is not a valid ID in ICTV tree", 'ERROR')

    def calcRankMatch(self):
        lines = ["ICTV name,ICTV rank,NCBI rank\n"]
        for name, node in self.ICTV2NCBI.items():
            n = self.ICTVTree.nodes[name]
            if (self.viralNCBITree.getSpeciesNode(node) is not None and node.rank != 'species'):
                lines.append(f'"{name}",{n.rank},below species\n')
            else:
                lines.append(f'"{name}",{n.rank},{node.rank}\n')
        with open("/Data/ICTVData/temp/ictv2ncbi.csv", 'wt') as fp:
            fp.writelines(lines)


        lines = ["NCBI id,NCBI rank,ICTV rank\n"]
        for id, node in self.NCBI2ICTV.items():
            n = self.viralNCBITree.nodes[id]
            if (self.viralNCBITree.getSpeciesNode(n) is not None and n.rank != 'species'):
                lines.append(f"{id},below species,{node.rank}\n")
            else:
                lines.append(f"{id},{n.rank},{node.rank}\n")
        with open("/Data/ICTVData/temp/ncbi2ictv.csv", 'wt') as fp:
            fp.writelines(lines)


taxoTree = TaxoTree()
# taxoTree.generateTestset(False, 2024, version='test')
# taxoTree.generateTestset(True, 2024, maxPerSpecies=2, version='test')