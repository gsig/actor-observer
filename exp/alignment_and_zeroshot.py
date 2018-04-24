#!/usr/bin/env python
import sys
import os
import subprocess
import traceback
import pdb
from bdb import BdbQuit
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())
sys.path.insert(0, '.')
os.nice(19)
from main import main

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
    '--print-freq', '1',
    '--train-file', '../CharadesEgo_v1_train.csv',
    '--val-file', '../CharadesEgo_v1_test.csv',
    '--dataset', 'CharadesEgoPlusRGB',
    '--data', '/scratch/gsigurds/CharadesEgo_v1_rgb/',
    '--arch', 'ActorObserverWithClassifier',
    '--subarch', 'resnet152',
    '--pretrained-subweights', '/nfs.yoda/gsigurds/charades_pretrained/resnet_rgb.pth.tar',
    '--loss', 'ActorObserverLossAllWithClassifier',
    '--subloss', 'DistRatio',
    '--lr', '1e-5',
    '--clsweight', '2.0',
    '--lr-decay-rate', '8',
    '--batch-size', '15',
    '--train-size', str(0.2 / 6),
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--epochs', '50',
    '--alignment',
    #'--usersalignment',
    '--valvideo',
    '--valvideoego',
    # '--evaluate',
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print ''
    pdb.post_mortem()
    sys.exit(1)
