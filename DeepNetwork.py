# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from skimage.io import *;
from skimage.segmentation import flood;
from skimage.transform import resize;

'''Change classfyid to lable'''
def classyIDtoSeglable(imsegresult):
    myseg=imsegresult.copy();
    myseg2=imsegresult.copy();
    startlabel=0;
    [hss,lss]=np.shape(imsegresult);
    for ii in range(hss):
        for jj in range(lss):
            tempvalue=myseg[ii,jj];
            if (tempvalue==-1):
                continue;
            floodmask=flood(myseg,(ii,jj));
            fpos=np.where(floodmask==True);
            myseg2[fpos]=startlabel;
            startlabel=startlabel+1;
            #print(startlabel);
            myseg[fpos]=-1;
    return myseg2;

'''Resize (smaller) image by a maxwidth'''
def imgResizebyMaxWidth(image,maxwidth):
    pass;
    leng=len(image.shape);
    if (leng==2):
        [hss,lss]=np.shape(image);
        bds=1;
    else:
        [hss,lss,bds]=np.shape(image);
        
    if (maxwidth>=hss and maxwidth>=lss):
        return [image, False, hss, lss];
    
    if (hss>lss):
        nhss=maxwidth;
        nlss=int(maxwidth*lss/hss);
    else:
        nlss=maxwidth;
        nhss=int(maxwidth*hss/lss);
        
    nimg=resize(image,(nhss,nlss,bds));
    return [nimg, True, nhss, nlss];

'''Enlarge a decsion image to bigger '''
def reMapDecisonImage(resultmap,hss,lss):
    pass;
    [shss,slss]=np.shape(resultmap);
    
    '''
    if (shss>=hss):
        return [resultmap, False];
    
    if (slss>=lss):
        return [resultmap, False];
    '''
    
    bl=shss/(hss+0.0);
    
    bigermap=np.zeros((hss,lss),dtype=resultmap.dtype);
    
    for ii in range(hss):
        for jj in range(lss):
            rii=int(ii*bl+0.5);
            rjj=int(jj*bl+0.5);
            if (rii>=shss):
                rii=shss-1;
            if (rjj>=slss):
                rjj=slss-1;
            bigermap[ii,jj]= resultmap[rii,rjj];   
    return [bigermap, True];





def runDeepNetworkSegment(im,imagemaxvalue= 255.0,hascosslink=True,\
                          mysavetempimg=False,temfsteppath="I:\\paper2022-01\\",\
                          myminlables=3,mychannel=15,mynConvs=2,mymaxiter=100,mylearningrate=0.2,\
                          printtrainstep=False,outputSegmentList=False,tempsegmentlist=None,tempclassfy=None,
                          resizeBigImage=False,resizeMaxwidth=600, tracefilename=None):
    pass;

    '''Input image data'''
    #im= inputimage;
    
    '''Max Value of image default=255'''
    #imagemaxvalue= 255.0;
    
    '''Has cross link'''
    #hascosslink=True;
    
    '''Save Temporary result'''
    #mysavetempimg=False;
    
    '''Temporary Path'''
    #temfsteppath="I:\\paper2022-01\\"
    
    '''#min classfy lables'''
    #myminlables=2;
    
    '''#max classfy labels'''
    #mychannel=10;
    
    '''#Number of convs'''
    #mynConvs=2;
    
    '''#Max iterator'''
    #mymaxiter=20;
    
    '''lerning rate'''
    #mylearningrate=0.2;
    
    '''Wether on not print Setp'''
    #printtrainstep=True;
    
    '''Wether output each classificaiton result '''
    #outputSegmentList=False,tempsegmentlist=templist
    
    '''oupt of net work'''
    #Tempclassfy
    
    '''Smaller big image '''
    #resizeBigImage=False,
    #resizeMaxwidth=600
    
    [hss,lss,bds]=im.shape;
    
    
    '''resize into smaller image'''
    ohss=hss;
    olss=lss;
    obds=bds;
    imresized=False;
    if (resizeBigImage==True):
        [im,imresized,nhss,nlss]=imgResizebyMaxWidth(im,resizeMaxwidth);
        
    
    if (imresized==True):
        hss=nhss;
        lss=nlss;
        imagemaxvalue=1; #np.max(im);
        print('Resize-%d,%d to %d,%d'%(ohss,olss,hss,lss));
    '''''' 
        
    
    class MyNet(nn.Module):
        def __init__(self,input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, mychannel, kernel_size=3, stride=1, padding=1 )
            self.bn1 = nn.BatchNorm2d(mychannel)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(mynConvs-1):
                self.conv2.append( nn.Conv2d(mychannel, mychannel, kernel_size=3, stride=1, padding=1 ) )
                self.bn2.append( nn.BatchNorm2d(mychannel) )
            self.conv3 = nn.Conv2d(mychannel, mychannel, kernel_size=1, stride=1, padding=0 )
            self.bn3 = nn.BatchNorm2d(mychannel)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu( x )
            x = self.bn1(x)
            z1=x;
            for i in range(mynConvs-1):
                x = self.conv2[i](x)
                x = F.relu( x )
                x = self.bn2[i](x)
            if hascosslink:
                x=torch.cat([x,z1]);
            x = self.conv3(x)
            x = self.bn3(x)
            return x;
    
    
    
    mystepsize_sim=1;
    mystepsize_con=1;
    
    use_cuda = torch.cuda.is_available();
    
    
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/imagemaxvalue]) )
    
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    
    
    # train
    
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
    model.train()
    
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()
    
    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average = True)
    loss_hpz = torch.nn.L1Loss(size_average = True)
    
    
    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], mychannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, mychannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=mylearningrate, momentum=0.9)
    #label_colours = np.random.randint(255,size=(100,3))
    label_colours = stableColorList();
        
    for batch_idx in range(mymaxiter):
        pass;
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, mychannel )
    
        outputHP = output.reshape( (im.shape[0], im.shape[1], mychannel) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)
    
        
            
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
    
        if mysavetempimg:
            im_target_rgb = np.array([label_colours[ c % mychannel ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            tempsavefilename=temfsteppath+("Z%03d.bmp"%batch_idx);
            imsave(tempsavefilename,im_target_rgb);
        
        
        
        
        if (outputSegmentList):
            imsegresult=im_target.reshape(hss,lss);
            tempsegmentlist.append([nLabels,imsegresult]);
            if (tempclassfy!=None):
                tempclassfy.append(output);
            
        
        loss = mystepsize_sim * loss_fn(output, target) + mystepsize_con * (lhpy + lhpz);
        loss.backward();
        optimizer.step();
        
        if printtrainstep:
            print (batch_idx, '/', mymaxiter, '|', ' label num :', nLabels, ' | loss :', loss.item())
        
        if nLabels<myminlables:
            break;
    
    
    #[hss,lss,bds]=im.shape;
    imsegresult=im_target.reshape(hss,lss);
    myresult=classyIDtoSeglable(imsegresult);
    
    #save trace
    omyresult=myresult.copy();
    
    #print(myresult.dtype);
    '''restore into original size if resized'''
    if (imresized):
        [myresult,remapped]=reMapDecisonImage(myresult,ohss,olss);
        #print('---restore');
    '''--------------------------------------'''
    #print(myresult.dtype);
     
    #save trace
    '''
    np.save(tracefilename+"np",[imsegresult,omyresult,myresult]);
    traceimg=imsegresult/np.max(imsegresult);
    imsave(tracefilename+"decide.bmp",traceimg);
    
    traceimg=omyresult/np.max(omyresult);
    imsave(tracefilename+"lable-o.bmp",traceimg);
    
    traceimg=myresult/np.max(myresult);
    imsave(tracefilename+"lable.bmp",traceimg);
    '''
    
    
    
    return myresult;


#resultimg=runDeepNetworkSegment(image);
def stableColorList():
    pass;
    cl = np.zeros((100,3),dtype='float32');
    cl[0,:]=(0,216,189);
    cl[1,:]=(110,71,242);
    cl[2,:]=(67,154,65);
    cl[3,:]=(5,123,172);
    cl[4,:]=(22,32,6);
    cl[5,:]=(171,213,17);
    cl[6,:]=(130,187,240);
    cl[7,:]=(133,242,238);
    cl[8,:]=(80,68,175);
    cl[9,:]=(141,51,153);
    cl[10,:]=(113,142,177);
    cl[11,:]=(84,220,106);
    cl[12,:]=(231,219,238);
    cl[13,:]=(81,57,154);
    cl[14,:]=(51,139,90);
    cl[15,:]=(217,23,205);
    cl[16,:]=(230,60,238);
    cl[17,:]=(54,68,90);
    cl[18,:]=(253,22,2);
    cl[19,:]=(35,248,67);
    cl[20,:]=(125,216,214);
    cl[21,:]=(217,0,93);
    cl[22,:]=(25,3,25);
    cl[23,:]=(57,51,105);
    cl[24,:]=(172,26,26);
    cl[25,:]=(137,159,155);
    cl[26,:]=(214,168,156);
    cl[27,:]=(217,165,63);
    cl[28,:]=(31,53,53);
    cl[29,:]=(199,88,87);
    cl[30,:]=(134,189,247);
    cl[31,:]=(222,131,65);
    cl[32,:]=(138,180,233);
    cl[33,:]=(221,213,53);
    cl[34,:]=(113,168,120);
    cl[35,:]=(89,2,125);
    cl[36,:]=(240,229,87);
    cl[37,:]=(192,188,142);
    cl[38,:]=(236,223,234);
    cl[39,:]=(196,54,35);
    cl[40,:]=(100,48,250);
    cl[41,:]=(65,240,116);
    cl[42,:]=(18,166,239);
    cl[43,:]=(190,116,167);
    cl[44,:]=(230,150,104);
    cl[45,:]=(136,64,22);
    cl[46,:]=(147,225,63);
    cl[47,:]=(183,123,109);
    cl[48,:]=(250,99,34);
    cl[49,:]=(253,16,112);
    cl[50,:]=(71,15,8);
    cl[51,:]=(206,216,165);
    cl[52,:]=(51,91,96);
    cl[53,:]=(233,115,114);
    cl[54,:]=(14,232,54);
    cl[55,:]=(12,193,3);
    cl[56,:]=(18,236,253);
    cl[57,:]=(89,224,101);
    cl[58,:]=(109,140,167);
    cl[59,:]=(242,141,11);
    cl[60,:]=(146,91,133);
    cl[61,:]=(16,93,2);
    cl[62,:]=(19,170,58);
    cl[63,:]=(128,212,165);
    cl[64,:]=(9,84,14);
    cl[65,:]=(45,220,110);
    cl[66,:]=(238,86,179);
    cl[67,:]=(56,242,44);
    cl[68,:]=(145,55,140);
    cl[69,:]=(27,212,55);
    cl[70,:]=(137,186,71);
    cl[71,:]=(7,73,12);
    cl[72,:]=(17,54,68);
    cl[73,:]=(54,30,188);
    cl[74,:]=(110,85,103);
    cl[75,:]=(243,171,111);
    cl[76,:]=(193,127,3);
    cl[77,:]=(249,112,89);
    cl[78,:]=(97,19,28);
    cl[79,:]=(9,232,121);
    cl[80,:]=(223,235,98);
    cl[81,:]=(128,249,232);
    cl[82,:]=(187,112,23);
    cl[83,:]=(116,48,239);
    cl[84,:]=(31,137,27);
    cl[85,:]=(164,46,193);
    cl[86,:]=(196,223,198);
    cl[87,:]=(211,106,92);
    cl[88,:]=(215,37,194);
    cl[89,:]=(40,42,119);
    cl[90,:]=(1,173,139);
    cl[91,:]=(188,139,25);
    cl[92,:]=(135,46,221);
    cl[93,:]=(54,108,104);
    cl[94,:]=(112,127,127);
    cl[95,:]=(197,233,13);
    cl[96,:]=(253,245,155);
    cl[97,:]=(144,43,239);
    cl[98,:]=(20,226,62);
    cl[99,:]=(161,143,68);
    return cl;




