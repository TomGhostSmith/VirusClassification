from sortedcontainers import SortedList
from tqdm import tqdm
import random
import sys
import os
from Bio import SeqIO

sys.path.append("code")
from config import config
from utils import IOUtils

modelRoot = "/Data/VirusClassification/model"
outputRoot = "/Data/VirusClassification"
queryFilePath = ""
querySubsetFilePath = None
config.setPath(modelRoot=modelRoot, outputRoot=outputRoot, queryFile=queryFilePath, querySubsetFile=querySubsetFilePath)

from entity.taxoTree import taxoTree


def loadGenbankSpecies(version, fileName="genbank.accession", species:dict[str, list[tuple]] = dict()):
    # for row in tqdm(metaDF.itertuples(), desc="loading meta data"):
    #     if ((not pandas.isnull(row.Species)) and row.Species in self.name2ID):
    with open(f"{config.modelRoot}/NCBI/Nucleotide/{version}/{fileName}") as fp:
        fp.readline()
        lines = fp.readlines()
    for line in tqdm(lines, desc="loading GenBank"):
        comma = line.find(",")
        accession = line[:comma]
        spec = line.strip()[comma + 1:]
        if (spec in taxoTree.viralNCBITree.name2ID):
            node = taxoTree.viralNCBITree.getSpeciesNode(taxoTree.viralNCBITree.nodes[taxoTree.viralNCBITree.name2ID[spec]])
            if (node is not None):
                id = node.name

                # get filePath
                name = accession[:accession.index(".")]
                chunks = [name[i:i+3] for i in range(0, len(name), 3)]
                file = f"{chunks[-1]}.fasta"
                path = f"{config.modelRoot}/NCBI/Nucleotide/{version}/fna/{"/".join(chunks[:-1])}/{file}"

                if (not os.path.exists(path)):
                    continue

                # add path to species
                if (id in species):
                    species[id].append((name, path))
                else:
                    species[id] = [(name, path)]
    return species


def generateTestset(source, species, seed, addNonVirus=False, maxPerSpecies=float('inf'), version=""):
    if (source not in ["refseq", "genbank"]):
        raise ValueError("Unsupport data source")
    random.seed(seed)

    wrapLength = 100   # wrap per 100 bp in the output fasta file

    # create an output file
    datasetName = f"{source}_{seed}_{version}"
    folderPath = f"/Data/VirusClassification/dataset/{datasetName}"
    os.makedirs(folderPath)
    targetFP = open(f"{folderPath}/{datasetName}.fasta", 'wt')

    # load query lengths from the competition, so we can generate dataset with similar length distribution
    with open("/Data/ICTVData/temp/queryLengths.txt") as fp:
        lengths = [int(l.strip()) for l in fp]
    minLength = min(lengths)

    # step 1: select sequences
    selectedSequences = list()
    for speciesID, sequences in species.items():  # sequences is a list of tuple (name, path)
        if (len(sequences) <= maxPerSpecies):
            seqs = sequences
        else:
            seqs = random.sample(sequences, maxPerSpecies)
        for name, seq in seqs:
            selectedSequences.append(('Viruses', name, speciesID, seq))

    # step 2: randomly add some bacteria and archaea from the host if wanted
    if (addNonVirus):
        nonVirusCount = round(len(selectedSequences) / 0.85 * 0.15)
        hosts = list()
        for hostList in taxoTree.viralNCBITree.hosts.values():
            hosts += hostList
        if nonVirusCount > len(hosts):
            selectedHosts = hosts
            # since host number are not sufficient, we add other bacteria and archaea
            # there are too few archaea, so we add all of them if possible
            if ((nonVirusCount - len(hosts)) // 2 > len(taxoTree.archaeaNCBITree.species)):
                selectedArchaeas = taxoTree.archaeaNCBITree.species.keys()
            else:
                selectedArchaeas = random.sample(list(taxoTree.archaeaNCBITree.species.keys()), (nonVirusCount - len(hosts)) // 2)
            for id in selectedArchaeas:
                name, seq = random.sample(taxoTree.archaeaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Archaea', name, id, seq))

            bacteriaCount = nonVirusCount - len(hosts) - len(selectedArchaeas)
            selectedBacteria = random.sample(list(taxoTree.bacteriaNCBITree.species.keys()), bacteriaCount)
            for id in selectedBacteria:
                name, seq = random.sample(taxoTree.bacteriaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Bacteria', name, id, seq))

        else:
            selectedHosts = random.sample(hosts, nonVirusCount)  # assume that 15% sequences are host and 85% sequences are virus

        for id in selectedHosts:
            if id in taxoTree.bacteriaNCBITree.species:
                name, seq = random.sample(taxoTree.bacteriaNCBITree.species[id], 1)[0]
                selectedSequences.append(('Bacteria', name, id, seq))
            elif id in taxoTree.archaeaNCBITree.species:
                name, seq = random.sample(taxoTree.archaeaNCBITree.species[id], 1)[0]
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
            exit(-1)
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

def main():
    # species = loadGenbankSpecies("20241225")
    # species = loadGenbankSpecies("20250505", "genbank_2025Spring.accession")
    species = loadGenbankSpecies("20250505", "genbank_2024.accession")
    # generateTestset("genbank", species, 2024, True, 2, "test")
    generateTestset("genbank", species, 2024, False, 2, "2024")

if (__name__ == '__main__'):
    main()