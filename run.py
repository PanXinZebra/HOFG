# -*- coding: utf-8 -*-
from skimage.io import *;
from SegHelperV2 import *;
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, join_segmentations
from skimage.segmentation import mark_boundaries
from DeepSegment import *;

import torch
print(torch.__version__);


#Example 1

#create "Grid"
thesegments= 800;
thecompactness=10;
imagefile=r".\\imgs\\test1.bmp";
gridresult=r".\\results\\grid.bmp";
image=imread(imagefile);
image=image[:,:,0:3];
nmimage=image/255;
segments=slic(image,n_segments=thesegments,compactness=thecompactness,start_label=0);
seghelper=SegHelperV2();
seghelper.readSegments(segments);
bd=mark_boundaries(image/255, segments);
imshow(bd);
imsave(gridresult,bd)

#Perform HOFG
thesavestepimgpath=".\\results\\"
mingrids=3;
result=runDeepSeg(seghelper,image,maxiter=4,savestepimg=True, savestepimgpath=thesavestepimgpath,theMinGrids=mingrids);
bd=mark_boundaries(nmimage, result,mode='thick');
imshow(bd);
resultpath=".\\results\\ZI-HOFG.bmp"
imsave(resultpath,bd)
resultnpz=".\\results\\ZS-HOFG";
np.save(resultnpz,result);
