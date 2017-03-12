from medpy import metric
import sys
sys.path.append("../")
from evaluation.surface import Surface
import glob
import nibabel as nb
import numpy as np
import os
import argparse


def parse():
    parser =argparse.ArgumentParser(description="Evaluate LITS segmentations.")
    parser.add_argument('--prediction_dir',
                        help="flat directory with prediction volumes in "
                        "*.nii.gz format",
                        required=True, type=str)
    parser.add_argument('--label_dir',
                        help="flat directory with ground truth labels in *.nii"
                             " format",
                        required=True, type=str)
    parser.add_argument('--results',
                        help="path to save results csv file",
                        required=False, type=str)
    return parser.parse_args()


def get_scores(pred,label,vxlspacing):
    volscores = {}

    if np.count_nonzero(pred)==0 or np.count_nonzero(label)==0:
        volscores['dice'] = 0
        volscores['jaccard'] = 0
        volscores['voe'] = 1.
    volscores['dice'] = metric.dc(pred,label)
    volscores['jaccard'] = volscores['dice']/(2.-volscores['dice'])
    volscores['voe'] = 1. - volscores['jaccard']
    
    print("DEBUG: ", np.count_nonzero(label), np.count_nonzero(pred))
    #if np.count_nonzero(label)==0:
        #print("DEBUG: ground truth has no lesions!")
    #if np.count_nonzero(pred)==0:
        #print("DEBUG: prediction has no lesions!")
    #volscores['rvd'] = metric.ravd(label,pred)

    #if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
        #volscores['assd'] = 0
        #volscores['msd'] = 0
    #else:
        #evalsurf = Surface(pred,label,
                           #physical_voxel_spacing = vxlspacing,
                           #mask_offset = [0.,0.,0.],
                           #reference_offset = [0.,0.,0.])
        #volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
        #volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)
    return volscores
    
    
if __name__=='__main__':
    args = parse()
    
    if not os.path.exists(args.prediction_dir):
        raise ValueError("prediction_dir doesn't exist: {}"
                         "".format(args.prediction_dir))
    if not os.path.exists(args.label_dir):
        raise ValueError("label_dir doesn't exist: {}"
                         "".format(args.label_dir))
    
    #labels = sorted(glob.glob(os.path.join(args.label_dir,
                                           #'segmentation*.nii')))
    probs = sorted(glob.glob(os.path.join(args.prediction_dir,
                                          'volume*.nii.gz')))
    labels = []
    for p in probs:
        p_fn = p.split('/')[-1]
        l_fn = 'segmentation'+p_fn[6:-7]+'.nii'
        labels.append(os.path.join(args.label_dir, l_fn))

    results = []
    outpath = args.results or "results.csv"

    for label, prob in zip(labels,probs):
        print(prob)
        loaded_label = nb.load(label)
        loaded_prob = nb.load(prob)

        #liver_scores = get_scores(loaded_prob.get_data()>=1,
                                  #loaded_label.get_data()>=1,
                                  #loaded_label.header.get_zooms()[:3])
        lesion_scores = get_scores(loaded_prob.get_data()>0,
                                   loaded_label.get_data()==2,
                                   loaded_label.header.get_zooms()[:3])
        #print("Liver dice: {}".format(liver_scores['dice']))
        print("Lesion dice: {}".format(lesion_scores['dice']))

        #results.append([label, liver_scores, lesion_scores])
        results.append([label, lesion_scores])

        #create line for csv file
        outstr = str(label) + ','
        #for l in [liver_scores, lesion_scores]:
            #for k,v in l.items():
                #outstr += str(v) + ','
                #outstr += '\n'
        for k,v in lesion_scores.items():
            outstr += str(v) + ','
            outstr += '\n'

        #create header for csv file if necessary
        if not os.path.isfile(outpath):
            headerstr = 'Volume,'
            #for k,v in liver_scores.items():
                #headerstr += 'Liver_' + k + ','
            for k,v in lesion_scores.items():
                headerstr += 'Lesion_' + k + ','
            headerstr += '\n'
            outstr = headerstr + outstr

        #write to file
        f = open(outpath, 'a+')
        f.write(outstr)
        f.close()
