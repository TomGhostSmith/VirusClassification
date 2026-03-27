from utils import IOUtils
from entity.sample import Sample
from moduleResult.plainResult import PlainResult

def runSingle(samples:list[Sample], keepVotes, keepProteinRes, pooling, class_names, name, indexes):
    # IOUtils.showInfo(f"start {indexes}")
    sampleResults = []
    for index, sample in zip(indexes, samples):
        proteinRes = []
        if (pooling == "vote"):
            votes = {n: 0 for n in class_names if "Unknown" not in n}
            for protein in sample.proteins:
                scores = protein.info[f"{name}_prob"]
                rawScores = [(taxo, score) for taxo, score in zip(class_names, scores)]
                tops = sorted(rawScores, key=lambda x:x[1], reverse=True)
                if (keepProteinRes):
                    proteinRes.append([(p, s) for p, s in rawScores[:20] if "Unknown" not in p])
                    # r = [PlainResult(p, s) for p, s in rawScores[:20] if "Unknown" not in p]
                    # if len(r) == 0:
                    #     r = None
                    # proteinRes.append(r)
                bestTaxo = tops[0][0]
                if ("Unknown" not in bestTaxo):
                    votes[bestTaxo] += 1
        elif (pooling == "sum" or pooling.startswith("top")):
            if pooling.startswith("top"):
                thresh = int(pooling[3:])
            else:
                thresh = None
            votes = {n: 0 for n in class_names if "Unknown" not in n}
            for protein in sample.proteins:
                scores = protein.info[f"{name}_prob"].tolist()
                rawScores = [(taxo, score) for taxo, score in zip(class_names, scores)]
                tops = sorted(rawScores, key=lambda x:x[1], reverse=True)
                if (keepProteinRes):
                    proteinRes.append([(p, s) for p, s in rawScores[:20] if "Unknown" not in p])
                    # r = [PlainResult(p, s) for p, s in rawScores[:20] if "Unknown" not in p]
                    # if len(r) == 0:
                    #     r = None
                    # proteinRes.append(r)
                for taxo, score in tops[:thresh]:
                    if ("Unknown" not in taxo):
                        votes[taxo] += score

        totalVotes = sum(votes.values())    # If pooling method == "sum", the totalVotes will be 1 * len(proteins) (not considering "Unknown" labels)
        if (totalVotes > 0):
            if (keepVotes):
                voteDict = {k: v/totalVotes for k, v in votes.items()}
            else:
                voteDict = None
            vs = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            results = [PlainResult(n, v/totalVotes) for n, v in vs[:20]]
            # results = [(n, v/totalVotes) for n, v in vs[:20]]
            sampleResults.append((results, voteDict, proteinRes, index))
        else:
            sampleResults.append((None, None, proteinRes, index))
    
    # IOUtils.showInfo(f"finish {indexes}")
    return sampleResults
