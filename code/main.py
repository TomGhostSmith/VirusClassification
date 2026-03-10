from config import config

import argparse
import os
import shutil

def main(input, output):
    from module.pipeline import Pipeline
    from module.virusPredModule import VirusPred
    from module.minimapMLMergeModule import MinimapMLMergeModule
    from module.minimapThreshRankModule import MinimapThreshRankModule
    from module.minimapThresholdModule import MinimapThresholdModule
    from module.mlModule import MLModule
    from module.esmIdentify import ESMIdentify
    from entity.sample import Sample
    from entity.modelRunnder import ModelRunnder
    from module.mergeModule import MergeModule
    from module.markerML import MarkerML


    pipeline = Pipeline(
    VirusPred([
        MinimapThresholdModule('VMRv4', factors=['60', 'completeMatch']), 
        ESMIdentify()]),
        # MinimapThreshRankModule('VMRv4', limitOutputDict=thRank),
        # MLModule('bottomup', 0.45, '1011000')
        MergeModule([
            MinimapThresholdModule(reference="VMRv4", factors=["60", "completeMatch"]),
            MarkerML("VMRv4", "bottomup", 0.45, "1022124")], 
            basicMerge, "pipeline_0.45")
    )
    
    evaluator = ModelRunnder(pipeline)
    evaluator.run(input, f"{output}/result.tsv")


def basicMerge(sample, modelNames, currentModelIndex):
    res = sample.results[modelNames[currentModelIndex]]
    return res


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description="VirTaxonomer: A deep learning based model to taxomize virus")
    parser.add_argument('--input', help="the fasta file to process", required=True)
    parser.add_argument('--model', help='the model folder', required=True)
    parser.add_argument('--output', help='the result folder', required=True)
    # parser.add_argument('--model', help='the model folder')
    # parser.add_argument('--ML', help="machine learning model strategy", default='bottomup')
    # parser.add_argument('--restrict', help="restrict lowest prediction rank", default='species')
    parser.add_argument('--batchsize', help="batchsize for machine learning models", type=int, default=64)
    args = parser.parse_args()
    input = args.input
    model = args.model
    output = args.output
    batchsize = int(args.batchsize)
    # batchsize = args.batchsize
    
    if (not (input.endswith('fasta') or input.endswith('fa'))):
        raise ValueError('The file you input seems not a fasta file')
    
    if (not os.path.isdir(model)):
        raise ValueError('The model path should be a folder')
    
    

    config.modelRoot = model
    config.esmBatchSize = batchsize
    config.mlBatchSize = batchsize
    config.setPath(modelRoot=model, outputRoot=output, queryFile=input)

    main(input, output)

    # shutil.rmtree(config.cacheFolder)