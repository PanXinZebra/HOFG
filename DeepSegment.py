# -*- coding: utf-8 -*-

from skimage.io import *;
from SegHelperV2 import *;
from DeepNetwork import *;
from skimage.io import *;
from skimage.segmentation import flood;






def runARound(overallhelper,inputimage,currentmask,backgroundseg,iteraseed,savesubimg=False, savesubimgpath=None, aresizeBigImage=False,aresizeMaxwidth=600, minGrids=-1,theiterList=[150,50,20,5,5], theiterClassnum=[3,3,5,15,15]):
    ''' 
    //tranfer parameter to network
    aresizeBigImage=False,aresizeMaxwidth=600
    '''
    
    saveseed=1;
    
    currentresultmask=currentmask.copy();
    
    localhelper=SegHelperV2();
    localhelper.readSegments(currentmask, skipNeighbor=True);
    #incrementseed=1;
    incrementseed=np.max(currentresultmask)+1;
    
    for x in localhelper.imgseglist.keys():
        member=localhelper.imgseglist[x];
        subimg=localhelper.cutImage(inputimage, x,fillblack=True);
      
        if (savesubimg):
            pass;
            maskimg=localhelper.getMaskImg(x);
            
            imsave(savesubimgpath+("%02d-%04dA.bmp"%(iteraseed,x)),subimg);
            imsave(savesubimgpath+("%02d-%04dM.bmp"%(iteraseed,x)),maskimg);
            saveseed=saveseed+1;
            
        
        
       
        '''filter  too small, single segment , too small groups'''
        neednotprocess=False;
        issmall=False;
        
        [tthss,ttlss,ttbds]=np.shape(subimg);
        
        if (tthss<=5 or ttlss<=5):
            issmall=True;
        
        
        if (iteraseed!=0):
            backvaluelist=localhelper.valuesSelectByKey(backgroundseg,x);
            groupnum=backvaluelist[0];
            if ((0 in backvaluelist) or issmall or groupnum<minGrids):
                #print('too small segment');
                #subsegments=currentresultmask.copy()*0+1;
                [shss,slss,sbds]=np.shape(subimg);
                subsegments=np.zeros((shss,slss))+1;
                neednotprocess=True;
            
        
       
            
        
        
        ''' Obtain Sub segment result, '''
        #subsegments=slic(subimg,n_segments=16,compactness=1,start_label=0);
        thetracefile="I:\\Paper2022-F8\\trace\\%04d-%04d"%(iteraseed,x);
        
        if (neednotprocess==False):
            pass;
            subsegments=0;
            '''
            if (iteraseed==0):
                pass;
                subsegments=runDeepNetworkSegment(subimg,myminlables=3,resizeBigImage=aresizeBigImage,resizeMaxwidth=aresizeMaxwidth, tracefilename=thetracefile);
            
            if (iteraseed>0 and iteraseed<=1):
                pass;
                subsegments=runDeepNetworkSegment(subimg,myminlables=3,mymaxiter=50,resizeBigImage=aresizeBigImage,resizeMaxwidth=aresizeMaxwidth, tracefilename=thetracefile);
            
            if (iteraseed>1 and iteraseed<=3):
                pass;
                #mymaxiter=20
                subsegments=runDeepNetworkSegment(subimg,myminlables=5,mymaxiter=20,resizeBigImage=aresizeBigImage,resizeMaxwidth=aresizeMaxwidth, tracefilename=thetracefile);
            
            if (iteraseed>3):
                pass;
                subsegments=runDeepNetworkSegment(subimg,myminlables=15,mymaxiter=10,resizeBigImage=aresizeBigImage,resizeMaxwidth=aresizeMaxwidth, tracefilename=thetracefile);
            '''  
            #theiterList=[150,50,20,5,5], theiterClassnum=[3,3,5,15,15]
            itlistlen=len(theiterList);
            iterclasslen=len(theiterClassnum);
            myiternum=theiterList[itlistlen-1];
            myiterclass=theiterClassnum[iterclasslen-1];
            
            if (iteraseed<itlistlen):
                myiternum=theiterList[iteraseed];
            if (iteraseed<iterclasslen):
                myiterclass=theiterClassnum[iteraseed];
            subsegments=runDeepNetworkSegment(subimg,myminlables=myiterclass,mymaxiter=myiternum,resizeBigImage=aresizeBigImage,resizeMaxwidth=aresizeMaxwidth, tracefilename=thetracefile);
                
            
        
        
        #print("--%d---%d"%(len(np.unique(subsegments)),incrementseed));
        ''' merge into local '''
        localhelper.copySegimage(subsegments,currentresultmask,x,incrementseed);
        incrementseed=incrementseed+np.max(subsegments)+10;
    
       
    
    ''''merge into global'''    
    overallhelper.mergeLables(currentresultmask);    
    overallhelper.copyMergetoDecsionLabel();
    #if (iteraseed!=0):
    overallhelper.breakSameDecsionLable();
    newseg=overallhelper.createSegmentsImageByDecsionLabel();
    return newseg;


def runDeepSeg(overallhelper,inputimage,maxiter=4, savesubimg=False, savesubimgpath=None, savestepimg=False,savestepimgpath=None,oresizeBigImage=False,oresizeMaxwidth=600, theMinGrids=-1,iterList=[150,50,20,5,5], iterClassnum=[3,3,5,15,15]):
    '''
    savesubimg 
    savesubimgpath
    ===save small image patches during execute method.
    
    savestepimg
    savestepimgpath
    ===save each iterate result
    
    //tranfer parameter to network
    oresizeBigImage=False,oresizeMaxwidth=600
    
     theMingrids=-1
     //min number of grids to perform segment when==-1 means  this parameter not use
     
     //max number of iterations and calsses in each round
     iterList=[150,50,20,5,5], iterClassnum=[3,3,5,15,15];
    '''
    
    

    for kk in overallhelper.imgseglist.keys():
        member=overallhelper.imgseglist[kk];
        member["DecsionLabel"]=1;
        member["MergeCounter"]=1;
        
    currentmask=overallhelper.createSegmentsImageByDecsionLabel();
    
    backgroundseg=overallhelper.createSegmentsImageByMergeCounter();
  
    
    '''Perform Segment'''
    ite=0;
    while ite<maxiter:
        print("Run[ %03d ]"%(ite));
        currentmask=runARound(overallhelper,inputimage,currentmask,backgroundseg,iteraseed=ite,savesubimg=savesubimg,savesubimgpath=savesubimgpath,aresizeBigImage=oresizeBigImage,aresizeMaxwidth=oresizeMaxwidth,minGrids=theMinGrids,theiterList=iterList, theiterClassnum=iterClassnum);
        ite=ite+1;
        backgroundseg=overallhelper.createSegmentsImageByMergeCounter();
        
        if (savestepimg):
            pass;
            np.save(savestepimgpath+"%03d.npz"%ite,currentmask);
            normimage=inputimage;
            if (np.max(normimage)>10):
                normimage=inputimage/255;
            bd=mark_boundaries(normimage, currentmask,mode='thick');
            imsave(savestepimgpath+"%03d.bmp"%ite,bd);
        
        
        
               
        
    return currentmask;



    







