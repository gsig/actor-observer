## ActorObserverNet code in PyTorch from Actor and Observer: Joint Modeling of First and Third-Person Videos, CVPR 2018 

Contributor: Gunnar Atli Sigurdsson

* This code implements a triplet network in PyTorch

The code implements found in:
```
@inproceedings{sigurdsson2018actor,
author = {Gunnar A. Sigurdsson and Abhinav Gupta and Cordelia Schmid and Ali Farhadi and Karteek Alahari},
title = {Actor and Observer: Joint Modeling of First and Third-Person Videos},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2018},
code = {https://github.com/gsig/actor-observer},
}
```



## Technical Overview:
 
For testing. The network uses a batch size of 25, scores all images, and pools the output to make a classfication prediction or uses all 25 outputs for localization.

All outputs are stored in the cache-dir. This includes epoch*.txt which is the classification output, and localize*.txt which is the localization output (note the you need to specify that you want this in the options).
Those output files can be combined after training with the python scripts in this directory.
All output files can be scored with the official MATLAB evaluation script provided with the Charades dataset.

Requirements:
* Python 2.7
* PyTorch 


## Steps to train your own model on CharadesEgo:
 
1. Download the Charades Annotations (allenai.org/plato/charades/)
2. Download the Charades RGB frames (allenai.org/plato/charades/)
3. Duplicate and edit one of the experiment files under exp/ with appropriate parameters. For additional parameters, see opts.py
4. Run an experiment by calling python exp/rgbnet.py where rgbnet.py is your experiment file
5. The checkpoints/logfiles/outputs are stored in your specified cache directory. 
6. Build of the code, cite our papers, and say hi to us at CVPR.

Good luck!


## Pretrained networks:



Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields
