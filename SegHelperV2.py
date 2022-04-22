# -*- coding: utf-8 -*-

import numpy as np;

from skimage.transform import *;
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, join_segmentations
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import *;
import pickle;
import copy;
from skimage.measure import shannon_entropy
from skimage.morphology import flood;
from skimage import feature;
from skimage import exposure;
from skimage.exposure import is_low_contrast;

class SegHelperV2:
    '''           
    A imagesegment's property
    ID===ID Number;
    Pixels-Row and Pixels-Col == Pixels position list
    Position-Rect  outer frame of a rectange
    W/H  Width and Hight
    CenterPoint CenterPoint of a segment
    
    '''
    imgseglist:dict;
    
    '''segment image'''
    imgseg:np.array;
    
    '''segment number'''
    numsegs:int;
    
    def __init__(self):
        pass;
        
    '''Read image-segments result into this seghelper
    skipNeighbor=False (default) add segment neighor list (slow)
    skipNeighbor=True do not add segment neighor list (faster)
    '''    
    def readSegments(self, imagesegments, skipNeighbor=False):
        #maxvalue=np.max(imagesegments);
        #minvalue=np.min(imagesegments);
        #print(maxvalue);
        #print(minvalue);
        seglist=dict();
        uniquelist=np.unique(imagesegments);
        for ii in uniquelist:
            members=dict();
            members["ID"]=ii;
            members["Pixels-Row"]=[];
            members["Pixels-Col"]=[];
            members["Position-Rect"]=[];
            members["W/H"]=[];
            members["CenterPoint"]=[];
            members["NumofPixels"]=0;
            members["DecsionLabel"]=0;
            members["MergeLable"]=0;
            members["MergeCounter"]=0;
            
            '''add neighbour'''
            members["Neigbors"]=[];
            
            seglist[ii]=members;
        
        [hss,lss]=np.shape(imagesegments);
        
        '''Set Pixels-Row and Pixels-Col ='''
        for ii in range(hss):
            for jj in range(lss):
                idpos=imagesegments[ii,jj];
                seglist[idpos]["Pixels-Row"].append(ii);
                seglist[idpos]["Pixels-Col"].append(jj);
                
        '''add neighbors'''
        if (skipNeighbor==False):
            pass;
            for ii in range(hss-1):
                for jj in range(lss-1):
                    #p1<---->p2---In row
                    p1=imagesegments[ii,jj];
                    p2=imagesegments[ii,jj+1];
                    if (not(p1==p2)):
                        pass;
                        if (not(p2 in seglist[p1]["Neigbors"])):
                            seglist[p1]["Neigbors"].append(p2);
                        if (not(p1 in seglist[p2]["Neigbors"])):
                            seglist[p2]["Neigbors"].append(p1);
                    
                    #p1<---->p2---In Col
                    p1=imagesegments[ii,jj];
                    p2=imagesegments[ii+1,jj];
                    if (not(p1==p2)):
                        pass;
                        if (not(p2 in seglist[p1]["Neigbors"])):
                            seglist[p1]["Neigbors"].append(p2);
                        if (not(p1 in seglist[p2]["Neigbors"])):
                            seglist[p2]["Neigbors"].append(p1);

        
        '''Set Position-Rect, W/H, CenterPoint, NumofPixels'''  
        for ii in uniquelist:
            minrow=min(seglist[ii]["Pixels-Row"]);
            maxrow=max(seglist[ii]["Pixels-Row"]);
            mincol=min(seglist[ii]["Pixels-Col"]);
            maxcol=max(seglist[ii]["Pixels-Col"]);
            seglist[ii]["Position-Rect"]=[minrow, mincol, maxrow, maxcol];
            seglist[ii]["W/H"]=[maxcol-mincol+1,maxrow-minrow+1];
            crow=int((maxrow+minrow)/2);
            ccol=int((maxcol+mincol)/2);
            seglist[ii]["CenterPoint"]=[crow, ccol];
            seglist[ii]["NumofPixels"]=len(seglist[ii]["Pixels-Row"]);
            
        self.imgseglist=seglist;
        self.numsegs=len(uniquelist);
        self.imgseg=imagesegments;
    
    '''Display neighors'''
    def listNeighbors(self):
        pass;
        keys=self.imgseglist.keys();
        for x in keys:
            print("-%d"%x);
            print(self.imgseglist[x]["Neigbors"]);
        
    
    '''Use "DecsionLabel" Create segment image'''
    def createSegmentsImageByDecsionLabel(self):
        keys=self.imgseglist.keys();
        copyimg=self.imgseg.copy();
        for x in keys:
            member=self.imgseglist[x];
            copyimg[member["Pixels-Row"],member["Pixels-Col"]]=member["DecsionLabel"];
        return copyimg;
    
    '''Use "MergeCounter" Create segment image'''
    def createSegmentsImageByMergeCounter(self):
        keys=self.imgseglist.keys();
        copyimg=self.imgseg.copy();
        for x in keys:
            member=self.imgseglist[x];
            copyimg[member["Pixels-Row"],member["Pixels-Col"]]=member["MergeCounter"];
        return copyimg;
    
    '''Cut a subimage by key'''
    def cutImage(self, ainputimage, key, fillblack=False, enhance=True):
        [minrow, mincol, maxrow, maxcol]=self.imgseglist[key]["Position-Rect"];
        [cols,rows]=self.imgseglist[key]["W/H"];
        shplen=len(ainputimage.shape);
        resultimage=0;
        if shplen>=3:
            resultimage=ainputimage[minrow:maxrow+1,mincol:maxcol+1,:];
        else:
            resultimage=ainputimage[minrow:maxrow+1,mincol:maxcol+1];
        
        resultimage=resultimage.copy();
        
        '''
        themask=np.zeros((rows,cols));
        if (fillblack):
            rowlist=(self.imgseglist[key]["Pixels-Row"]).copy();
            collist=(self.imgseglist[key]["Pixels-Col"]).copy();
            
            rlen=len(rowlist);
            for ii in range(rlen):
                rowlist[ii]=rowlist[ii]-minrow;
                collist[ii]=collist[ii]-mincol;
                
            themask[rowlist,collist]=1;
            if shplen>=3:
                [hss,lss,bds]=np.shape(ainputimage);
                for ii in range(bds):
                    resultimage[:,:,ii]=resultimage[:,:,ii]*themask;    
            else:
                resultimage=resultimage*themask;
        '''
        if (fillblack):
            rowlist=(self.imgseglist[key]["Pixels-Row"]).copy();
            collist=(self.imgseglist[key]["Pixels-Col"]).copy();
            
            rlen=len(rowlist);
            for ii in range(rlen):
                rowlist[ii]=rowlist[ii]-minrow;
                collist[ii]=collist[ii]-mincol;
            [hss,lss,bds]=np.shape(ainputimage);
            themask=np.zeros((rows,cols))+1;
            themask[rowlist,collist]=0;
            [fhhh,flll]=np.where(themask==1);
            if shplen>=3:
                for ii in range(bds):
                    pass;
                    resultimage[fhhh,flll,ii]=np.mean(resultimage[rowlist,collist,ii]);
                       
            else:
                resultimage[fhhh,flll]=np.mean(resultimage[rowlist,collist]);
           
        
        if (enhance):
            islowcontrast=is_low_contrast(resultimage);
            islowcontrast=True;
            if (islowcontrast):
                resultimage2 = exposure.equalize_hist(resultimage)
                if (resultimage.dtype=='uint8'):
                    resultimage2=resultimage2*255;
                    resultimage2=np.uint8(resultimage2);
                resultimage=resultimage2;
            
        
        
        return resultimage;
    
    def getMaskImg(self,key):
        [minrow, mincol, maxrow, maxcol]=self.imgseglist[key]["Position-Rect"];
        [cols,rows]=self.imgseglist[key]["W/H"];
        themask=np.zeros((rows,cols,3));
        
        rowlist=(self.imgseglist[key]["Pixels-Row"]).copy();
        collist=(self.imgseglist[key]["Pixels-Col"]).copy();
            
        rlen=len(rowlist);
        for ii in range(rlen):
            rowlist[ii]=rowlist[ii]-minrow;
            collist[ii]=collist[ii]-mincol;
        
        themask[rowlist,collist,0]=1;
        themask[rowlist,collist,1]=1;
        themask[rowlist,collist,2]=1;
        return themask;
        
 
        
    '''Values selected by key in a segment image'''
    def valuesSelectByKey(self, ainputimage, key):
        collist=self.imgseglist[key]["Pixels-Col"];
        rowlist=self.imgseglist[key]["Pixels-Row"];
        return np.unique( ainputimage[rowlist,collist]);
        
        
    
    ''''copy a subsegimge into a resultimg according key'''
    def copySegimage(self,ainputseg,wholeseg,key,incrementseed):
        [minrow, mincol, maxrow, maxcol]=self.imgseglist[key]["Position-Rect"];
        [cols,rows]=self.imgseglist[key]["W/H"];
        numpixels=self.imgseglist[key]["NumofPixels"];
        collist=self.imgseglist[key]["Pixels-Col"];
        rowlist=self.imgseglist[key]["Pixels-Row"];
        for ii in range(numpixels):
            thecol=collist[ii];
            therow=rowlist[ii];
            wholeseg[therow,thecol]=ainputseg[therow-minrow,thecol-mincol]+incrementseed;
        return;
        
    '''Merge a segment in to "mergetlabels"'''
    def mergeLables(self, ainputset):
        keys=self.imgseglist.keys();
        for key in keys:
            rowlist=self.imgseglist[key]["Pixels-Row"];
            collist=self.imgseglist[key]["Pixels-Col"];
            subids=ainputset[rowlist,collist];
            (uids,unums)=np.unique(subids,return_counts=True);
            upos=np.argmax(unums);
            self.imgseglist[key]["MergeLable"]=uids[upos];
    
    def breakSameDecsionLable(self):
        pass;
        maxlable=0;
        keys=self.imgseglist.keys();
        decisionlist=dict();
        for key in keys:
            adecision=self.imgseglist[key]["DecsionLabel"];
            if (adecision>maxlable):
                maxlable=adecision;
            
            if (adecision in decisionlist):
                decisionlist[adecision].append(key);
            else:
                decisionlist[adecision]=[];
                decisionlist[adecision].append(key);
            
            self.imgseglist[key]["poped"]=0;
        
        decsionkeys=decisionlist.keys();
        
        for tdecsion in decsionkeys:
            dgroups=[];
            sdkeylist=decisionlist[tdecsion];
            
            while (True):
                pass;
                firstid=sdkeylist[0];
                allchild=self.popKeys2(firstid,tdecsion);
                #print("---------------%d"%len(allchild));
                for echild in allchild:
                    sdkeylist.remove(echild);
                dgroups.append(allchild);
                if (len(sdkeylist)==0):
                    break;
            if (len(dgroups)<=1):
                continue;
            
            for ii in range(len(dgroups)):
                if (ii==0):
                    continue;
                print("Gounp NUM=%d,Break decsion %d-->%d"%(len(dgroups),tdecsion,maxlable+1));
                maxlable=maxlable+1;
                for renewkey in dgroups[ii]:
                    pass;
                    self.imgseglist[renewkey]["DecsionLabel"]=maxlable;
                    
            
    '''get All Connection list, recursion can't use on biger data'''
    def popKeys(self, key, decsion):
        self.imgseglist[key]["poped"]=1;
        nbs=self.imgseglist[key]["Neigbors"];
        if (len(nbs)==0):
            return [key];
        
        childlist=[];
        resultlist=[key];
        for tmpkey in nbs:
            if (not(self.imgseglist[tmpkey]["DecsionLabel"]==decsion)):
                continue;
            if (not(self.imgseglist[tmpkey]["poped"]==0)):
                continue;
            self.imgseglist[tmpkey]["poped"]=1;
            childlist.append(tmpkey);
            
        for ekeys in childlist:
            rs=self.popKeys(ekeys,decsion);
            for zz in rs:
                resultlist.append(zz);
        return resultlist;
  
    '''get All Connection list,more powerful'''
    def popKeys2(self, key, decsion):
        pass;
        self.imgseglist[key]["poped"]=1;
        nbs=self.imgseglist[key]["Neigbors"];
        if (len(nbs)==0):
            return [key];
        
        resultlist=[key];
        mystack=[key];
        
        while (len(mystack)!=0):
            pass;
            thekey=mystack[0];
            mystack=mystack[1:];
            
            nbs=self.imgseglist[thekey]["Neigbors"];
            
            if (len(nbs)==0):
                continue;
            
            for tmpkey in nbs:
                if (not(self.imgseglist[tmpkey]["DecsionLabel"]==decsion)):
                    continue;
                if (not(self.imgseglist[tmpkey]["poped"]==0)):
                    continue;
                self.imgseglist[tmpkey]["poped"]=1;
                resultlist.append(tmpkey);
                mystack.append(tmpkey);      
                
        return resultlist;
    
    '''Copy MergeLabel to DecsionLabel, all the DecsionLabel was set to 0 to N '''
    def copyMergetoDecsionLabel(self):
        mydict=dict();
        countdict=dict();
        incrementlable=0;
                
        for key in self.imgseglist.keys():
            member=self.imgseglist[key];
            
            mlable=member["MergeLable"];
            dlable=0;
            if (mlable in mydict):
                dlable=mydict[mlable];
                '''If found multi segment assign to same merge label ==(Number of segments-1)'''
                #countdict[dlable]=1;
                countdict[dlable]=countdict[dlable]+1;
            else:
                mydict[mlable]=incrementlable;
                incrementlable=incrementlable+1;
                dlable=mydict[mlable];
                countdict[dlable]=0;
                
            member["DecsionLabel"]=dlable;
            
        for key in self.imgseglist.keys():
            member=self.imgseglist[key];
            dlable=member["DecsionLabel"];
            '''If found multi segment assign to same merge label==(Number of segments-1), just one ==0'''
            member["MergeCounter"]=countdict[dlable];
            
                
        
        
     
class DecsionImageHelper:
    '''
    Input a rgb image, and translate it into a int categorylabel image
    '''
    image:np.array;
    
    
    colorlist:[];
    colornum:int;
    npcolorlist:np.array;
    
    
    
    def __init__(self):
        pass;
        self.setColorList();
        
    '''inputimage and read it into image '''
    def readImage(self, decimage):
        pass;
        
        [hss,lss,bds]=np.shape(decimage);
        
        if bds==4:
            decimage=decimage[:,:,0:3];
            '''remove transperant bands'''
            
        
        self.image=np.zeros((hss,lss),dtype='int');
        
        for ii in range(hss):
            for jj in range(lss):
                tvalue=decimage[ii,jj,:];
                self.image[ii,jj]=self.getLabel(tvalue);
                
    def getLabel(self, inputvalue):
        pass;
        vv=np.sum(np.abs(self.npcolorlist-inputvalue),1);
        return np.argmin(vv);
        
    
    def setColorList(self, alist=None):
        if alist is not None:
            self.colorlist=alist;
            self.colornum=len(self.colorlist);  
            self.npcolorlist=np.asarray(self.colorlist);
            return;
        '''
        Set Plate==Vaihingen and Potsdam default value
        0 草地
        1 树木
        2 建筑
        3 道路  
        4 汽车
        5 裸地  比实验区1多的部分
        '''
        self.colorlist=[(0,255,255),\
                        (0,255,0),\
                        (0,0,255),\
                        (255,255,255),\
                        (255,255,0),\
                        (255,0,0)];
        self.colornum=len(self.colorlist);  
        self.npcolorlist=np.asarray(self.colorlist);
    
    
    def clone(self):
        return copy.deepcopy(self);
    
    
    '''Transform decision based on segments'''
    def transformIntoSegment(self, sh:SegHelperV2):
        allkeys=sh.imgseglist.keys();
        for thekey in allkeys:
            segment=sh.imgseglist[thekey];
            hlist=segment["Pixels-Row"];
            llist=segment["Pixels-Col"];
            items=self.image[hlist,llist];
            [unique,count]=np.unique(items,return_counts=True)
            pos=np.argmax(count);
            realvalue=int(unique[pos]);
            self.image[hlist,llist]=realvalue;
            

    '''save lable image into color image'''
    def saveColorImage(self, filename):
        pass;
        [hss,lss]=np.shape(self.image);
        resultimage=np.zeros((hss,lss,3),dtype='uint8');
        
        for ii in range(hss):
            for jj in range(lss):
                thelabel=self.image[ii,jj];
                thecolor=self.colorlist[thelabel];
                resultimage[ii,jj,:]=thecolor;
        
        imsave(filename,resultimage);
        
    def accuracyOA(self, otherdecsion):
        pass;
        
        comparemap=(self.image==otherdecsion.image)+0;
        [hss,lss]=np.shape(self.image);
        
        accuracy=np.sum(comparemap)/(hss*lss);
        print(accuracy);
        return accuracy;
        
                
        
        
            
            
            
            
        
        
    
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        