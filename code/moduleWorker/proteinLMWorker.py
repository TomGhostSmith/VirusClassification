import numpy
from tqdm import tqdm

from utils import IOUtils
from entity.sample import Sample
from moduleResult.plainResult import PlainResult

def applyStrategy(distances, confidenceScores, strategy):
    # here we use softmin weighted distance
    # size of [cluster,  sample]
    if (strategy == 'nearest'):
        return softmin(distances)
    if (strategy == 'nearest_bound'):
        distances = distances + numpy.where(confidenceScores == 0, numpy.inf, 0)
        return softmin(distances)
    if (strategy == 'confidence'):
        return confidenceScores
    if (strategy == 'product'):
        w = softmin(distances)
        return w * confidenceScores
    if (strategy.startswith('nearest_conf')):
        thresh = float(strategy[12:])
        distances = distances + numpy.where(confidenceScores < thresh, numpy.inf, 0)
        return softmin(distances)


def extractWeightMatrix(clusters:list[tuple[str, list[numpy.ndarray]]], embeddings, strategy, invStdVar, refEmbeddings, ref_sq):
    distances = numpy.zeros((len(clusters), embeddings.shape[0]), dtype=numpy.float32)
    if (strategy == "individual"):
        emb = embeddings * invStdVar
        emb_sq = numpy.sum(emb ** 2, axis=1, keepdims=True).T
        re = refEmbeddings @ emb.T
        sq = ref_sq - 2 * re + emb_sq
        sq = numpy.maximum(sq, 0.0)
        distances = numpy.sqrt(sq)

        weights = softmin(distances)
        return weights

    else:
        confidenceScores = numpy.zeros((len(clusters), embeddings.shape[0]), dtype=numpy.float16)
        for idx, (name, mu, var, dis_threshes) in enumerate(clusters):
            mu = mu.astype(numpy.float64)
            var = var.astype(numpy.float64)
            dis = numpy.sqrt(numpy.sum((embeddings - mu) ** 2 / var, axis=1)).astype(numpy.float32)
            ranks = numpy.searchsorted(dis_threshes, dis, side='left')   # ideally, we should calculate average between left and right. But here, we think the identical number is almost impossible
            scores = 1 - ranks / len(dis_threshes)

            distances[idx] = dis
            confidenceScores[idx] = scores

        weights = applyStrategy(distances, confidenceScores, strategy)
        return weights
    
def extractPrediction(scores, inverse_indicies, uniqueNames):
    votes = numpy.bincount(inverse_indicies, weights=scores)
    total = sum(votes)

    if (total > 0):
        predictions = [PlainResult(n, s/total) for n, s in sorted(zip(uniqueNames, votes), key=lambda x:x[1], reverse=True)]
    else:
        predictions = None
    return predictions


def runSingle(clusters, samples:list[Sample], indexs, pooling, embedding, strategy, model, isv, ref, refsq, ind, uniqueNames, t=0, queue=None):
    res = []
    incre = 0
    shms = []
    
    if (isinstance(isv, tuple)):
        shm, invStdVar = IOUtils.loadSharedMemory(*isv)
        shms.append(shm)
    else:
        invStdVar = isv

    if (isinstance(ref, tuple)):
        shm, refEmbeddings = IOUtils.loadSharedMemory(*ref)
        shms.append(shm)
    else:
        refEmbeddings = ref

    if (isinstance(refsq, tuple)):
        shm, ref_sq = IOUtils.loadSharedMemory(*refsq)
        shms.append(shm)
    else:
        ref_sq = refsq

    if (isinstance(ind, tuple)):
        shm, inverse_indicies = IOUtils.loadSharedMemory(*ind)
        shms.append(shm)
    else:
        inverse_indicies = ind

    if (queue is None):
        bar = tqdm(total=len(samples))

    for index, sample in zip(indexs, samples):
        if (len(sample.proteins) == 0):
            res.append((None, index))
            continue
        # if (self.embedding == 'CLS'):
        #     embeddings = numpy.array([protein.info[f"{self.model}_CLSemb"] for protein in sample.proteins])
        if (embedding == 'ave'):
            embeddings = numpy.array([protein.info[f"{model}_aveemb"] for protein in sample.proteins])

        embeddings = embeddings.astype(numpy.float64)
        if (pooling == "mean"):
            aveEmbedding = numpy.mean(embeddings, axis=0, keepdims=True)
            weights = extractWeightMatrix(clusters, aveEmbedding, strategy, invStdVar, refEmbeddings, ref_sq)
            weights = weights.squeeze(1)

            res.append((extractPrediction(weights, inverse_indicies, uniqueNames), index))

        else:
            weights = extractWeightMatrix(clusters, embeddings, strategy, invStdVar, refEmbeddings, ref_sq)
            
            if (pooling == 'sum'):
                votes = numpy.sum(weights, axis=1)
            elif (pooling.startswith('top')):
                # IOUtils.showInfo("s1")
                thresh = int(pooling[3:])
                # IOUtils.showInfo("s2")
                nonTopWeightsIdx = numpy.argpartition(-weights, thresh, axis=0)[thresh:]
                # IOUtils.showInfo("s3")
                weights[nonTopWeightsIdx, numpy.arange(weights.shape[1])] = 0
                # IOUtils.showInfo("s4")
                votes = numpy.sum(weights, axis=1)

            res.append((extractPrediction(votes, inverse_indicies, uniqueNames), index))
        
        if (queue is None):
            bar.update(1)
        else:
            try:
                queue.put((t, 1 + incre), block=False)
                incre = 0
            except queue.Full:
                IOUtils.showInfo("stuck")
                incre += 1
    if (queue is None):
        bar.close()
    
    for shm in shms:
        IOUtils.closeSharedMemory(shm)
    return res
    

def softmin(distances):
    w = numpy.exp(-distances)
    s = numpy.sum(w, axis=0)
    msk = s != 0
    w[:, msk] /= s[msk]
    return w
