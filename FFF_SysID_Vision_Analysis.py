# import required libraries
import numpy as np
import cv2
print('OpenCV version: '+cv2.__version__)
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from collections import Counter

# Set source folder
SRC_FOLDER = "C:/Users/raksh/OneDrive - The Pennsylvania State University/PhD Research/Paper-4/SysID Experiment/OL Test 3/"

# open and read file containing start and end timestamps of the videos
df_vidTimes = pd.read_excel(SRC_FOLDER + "Video_Timestamps_1.xlsx")
df_vidTimes.drop(df_vidTimes.columns[0],axis=1,inplace=True)

################ ALL FUNCTIONS DEFINITIONS ################

def perspCorrection(img,pt1,pt2,pt3,pt4,scale_width,scale_height):
    
    # Create a copy of the image
    img_copy = np.copy(img)

    # Convert to RGB so as to display via matplotlib
    # Using Matplotlib we can easily find the coordinates of the 4 points that is essential for finding then transformation matrix
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       
    # to calculate the transformation matrix
    input_pts = np.float32([pt1,pt2,pt3,pt4])
    output_pts = np.float32([[0,0],[scale_width-1,0],[0,scale_height-1],[scale_width-1,scale_height-1]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    
    # Apply the perspective transformation to the image
    imgPersp = cv2.warpPerspective(img,M,(scale_width, scale_height)) #,flags=cv2.INTER_LINEAR) cv2.INTER_CUBIC is also an option
    imgGrayPersp = cv2.cvtColor(imgPersp, cv2.COLOR_BGR2GRAY)            
    
    # visulaize corners using cv2 circles
    for x in range (0,4):
        cv2.circle(img_copy,(round(input_pts[x][0]),round(input_pts[x][1])),5,(0,0,255),cv2.FILLED)    
    
    return [img_copy,imgPersp,imgGrayPersp]
    
def extractTopBottom(img,tStart,tEnd,bStart,bEnd):
    img_top = img[tStart[1]:tEnd[1],tStart[0]:tEnd[0]]
    img_bottom = img[bStart[1]:bEnd[1],bStart[0]:bEnd[0]]      
    
    return [img_top,img_bottom]
    
def gaussianBlur(img,fsize):
    
    # gaussian blur
    gblur = cv2.GaussianBlur(img,(fsize,fsize),0)
    
    return gblur

def medianBlur(img,fsize=3):
    
    # median blur - effective at removing salt and pepper noise
    mblur = cv2.medianBlur(img,fsize)
    
    return mblur
    
def bilateralFilter(img):
    
    # Bilateral filter preserves edges while removing noise
    bfblur = cv2.bilateralFilter(img,9,75,75)
    
    return bfblur
    
def gAdaptiveThresholding(img):
    
    # median filtering
    adaptive_gaussian = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    return adaptive_gaussian

def morphOps(img,kernel1,kernel2,k1_num_passes=2):    
    
    # Closing = Dilation + Erosion
    # dilation
    mask_dil = cv2.dilate(img,kernel1,iterations = k1_num_passes)
    
    # erosion
    mask_erode = cv2.erode(mask_dil,kernel2,iterations = 1)
    
    return mask_erode    

def computeW_Rev(img,img_debug): 
    
    avg_num_pixels = 159
    scaling_factor = 1.0
    mm_per_pixel = ((1/32)*25.4)/(scaling_factor*avg_num_pixels)
    edge_length_threshold = 55
    min_L_edge_threshold = False
    min_R_edge_threshold = False
    
    # Predefine arrays for data storage
    approx_edges = 10
    num_edges = np.zeros(img.shape[0]) #,dtype=np.uint16) 
    edge_start = np.zeros([img.shape[0],approx_edges])#,dtype=np.uint16)
    edge_end = np.zeros([img.shape[0],approx_edges])#,dtype=np.uint16)
    
    edge_count = 0
    k=0

    sse = False
    tse = False

    # start scanning from (0,0) until black pixel is found 
    # go across columns first

    for i in range(img.shape[0]):

        found_edge = False
        temp_edge_count = 0
        k=0    

        for j in range(img.shape[1]):

            if(img[i,j]<=50):
                # Black pixel found - edge
                if(found_edge==False):
                    found_edge = True
                    temp_edge_count += 1
                    num_edges[i] = temp_edge_count
                    edge_start[i][k] = j
                    k += 1

            else:
                if(found_edge):
                    edge_end[i][k-1] = j-1
                    found_edge = False         
    
    x = Counter(num_edges)
    y = {z:count for z, count in x.items() if count >= edge_length_threshold and z > 1}
    #print(y)
    if(len(y)!=0):
        edge_condition = sorted(y,key=y.get)[0]
        
    else:
        print('num_edges > 1 and length(num_edges) >= threshold not satisfied . . . Lowering threshold to identify matches')
        w = {z:count for z, count in x.items() if count < edge_length_threshold and z > 1}
        if(len(w)!=0):
            print('Found num_edges > 1 and length(num_edges) < threshold!')
            edge_condition = sorted(w,key=w.get)[0]
        else:
            print('Unable to find edge condition . . . check image')
            edge_condition = -1

    if img_debug:
        print('edge condition: ' + str(edge_condition))
    
    if edge_condition == 2: #max(num_edges)==2:

        # max num_edges = 2
        
        L1_edge_start = edge_start[:,0][np.argwhere(num_edges==2)][np.logical_and(edge_start[:,0][np.argwhere(num_edges==2)]>60,edge_start[:,0][np.argwhere(num_edges==2)]<300)]
        L1_edge_end = edge_end[:,0][np.argwhere(num_edges==2)][np.logical_and(edge_end[:,0][np.argwhere(num_edges==2)]>60,edge_end[:,0][np.argwhere(num_edges==2)]<300)]

        if(np.max(L1_edge_start)-np.min(L1_edge_start)>13):
            L1_edge_start = L1_edge_start[L1_edge_start >= (np.max(L1_edge_start)-10)]

        if(np.max(L1_edge_end)-np.min(L1_edge_end)>15):
            L1_edge_end = L1_edge_end[L1_edge_end >= (np.max(L1_edge_end)-10)]

        trueLedge_start = L1_edge_start
        trueLedge_end = L1_edge_end

        R1_edge_start = edge_start[:,1][np.argwhere(num_edges==2)][edge_start[:,1][np.argwhere(num_edges==2)]>350]
        R1_edge_end = edge_end[:,1][np.argwhere(num_edges==2)][edge_end[:,1][np.argwhere(num_edges==2)]>350]

        if(np.max(R1_edge_start)-np.min(R1_edge_start)>13):
            R1_edge_start = R1_edge_start[R1_edge_start <= (np.min(R1_edge_start)+10)]

        if(np.max(R1_edge_end)-np.min(R1_edge_end)>13):
            R1_edge_end = R1_edge_end[R1_edge_end <= (np.min(R1_edge_end)+10)]

        trueRedge_start = R1_edge_start
        trueRedge_end = R1_edge_end

        if(len(trueLedge_start)>len(trueLedge_end)):
            trueLedge_start = np.array([trueLedge_start[i] for i in range(len(trueLedge_end))])
        
        if(len(trueLedge_start)<len(trueLedge_end)):
            trueLedge_end = np.array([trueLedge_end[i] for i in range(len(trueLedge_start))])

        if(len(trueRedge_start)>len(trueRedge_end)):
            trueRedge_start = np.array([trueRedge_start[i] for i in range(len(trueRedge_end))])
        
        if(len(trueRedge_start)<len(trueRedge_end)):
            trueRedge_end = np.array([trueRedge_end[i] for i in range(len(trueRedge_start))])

        line1_start = (round(np.mean((trueLedge_start+trueLedge_end)/2)),0) 
        line1_end = (round(np.mean((trueLedge_start+trueLedge_end)/2)),img.shape[0])

        line2_start = (round(np.mean((trueRedge_start+trueRedge_end)/2)),0)
        line2_end = (round(np.mean((trueRedge_start+trueRedge_end)/2)),img.shape[0])

        edge_count = 2
        case_cond = 1
        
    elif edge_condition == 3: #max(num_edges)==3: 
        
        # max num_edges = 3
               
        # logic for finding true left edge                      
        L2_edge_start = edge_start[:,1][np.argwhere(num_edges==3)][edge_start[:,1][np.argwhere(num_edges==3)]<250]
        if(len(L2_edge_start)>=edge_length_threshold):
            trueLedge_start = L2_edge_start
            trueLedge_end = edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]
        else:
            if(len(edge_start[:,0][np.argwhere(num_edges==3)][np.logical_and(edge_start[:,0][np.argwhere(num_edges==3)]<250,edge_start[:,0][np.argwhere(num_edges==3)]>60)])!=0):                                              
                L1_edge_start = edge_start[:,0][np.argwhere(num_edges==3)][np.logical_and(edge_start[:,0][np.argwhere(num_edges==3)]<250,edge_start[:,0][np.argwhere(num_edges==3)]>60)]                                          
                
                if(len(L2_edge_start)!=0):
                    L1_edge_start = np.hstack((L1_edge_start,L2_edge_start))

                if(np.max(L1_edge_start)-np.min(L1_edge_start)>13):
                    L1_edge_start = L1_edge_start[L1_edge_start >= (np.max(L1_edge_start)-10)]

            else:
                L1_edge_start = edge_start[:,0][np.argwhere(num_edges==2)][edge_start[:,0][np.argwhere(num_edges==2)]<250]

            if(len(L1_edge_start)>=edge_length_threshold):
                trueLedge_start = L1_edge_start

                if(len(edge_start[:,0][np.argwhere(num_edges==3)][np.logical_and(edge_start[:,0][np.argwhere(num_edges==3)]<250,edge_start[:,0][np.argwhere(num_edges==3)]>60)])!=0):
                    trueLedge_end = edge_end[:,0][np.argwhere(num_edges==3)][np.logical_and(edge_end[:,0][np.argwhere(num_edges==3)]<250,edge_end[:,0][np.argwhere(num_edges==3)]>60)]                    
                    
                    if(len(L2_edge_start)!=0):
                        trueLedge_end = np.hstack((trueLedge_end,edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]))

                    if(np.max(trueLedge_end)-np.min(trueLedge_end)>13):
                        trueLedge_end = trueLedge_end[trueLedge_end >= (np.max(trueLedge_end)-10)]

                else:
                    trueLedge_end = edge_end[:,0][np.argwhere(num_edges==2)][edge_end[:,0][np.argwhere(num_edges==2)]<250]
            
            elif(len(L1_edge_start)!=0 and len(L1_edge_start)<edge_length_threshold):
                
                trueLedge_start = L1_edge_start

                trueLedge_end = edge_end[:,0][np.argwhere(num_edges==3)][edge_end[:,0][np.argwhere(num_edges==3)]<250]
                trueLedge_end = np.hstack((trueLedge_end,edge_end[:,0][np.argwhere(num_edges==2)][edge_end[:,0][np.argwhere(num_edges==2)]<250]))
                
                min_L_edge_threshold = True
            else:
                print('max(num_edges)=3 invalid true left edge condition encountered . . . check code')

        # logic for finding true right edge
        R2_edge_start = edge_start[:,1][np.argwhere(num_edges==3)][edge_start[:,1][np.argwhere(num_edges==3)]>350]
        if(len(R2_edge_start)>=edge_length_threshold):
            trueRedge_start = R2_edge_start
            trueRedge_end = edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]>350]
        else:
            R1_edge_start = edge_start[:,1][np.argwhere(num_edges==2)][edge_start[:,1][np.argwhere(num_edges==2)]>350]

            if(len(R1_edge_start)==0):
                # three definite edges
                trueRedge_start = edge_start[:,2][np.argwhere(num_edges==3)][edge_start[:,2][np.argwhere(num_edges==3)]>350]
                trueRedge_end = edge_end[:,2][np.argwhere(num_edges==3)][edge_end[:,2][np.argwhere(num_edges==3)]>350]
                            
            elif(len(R1_edge_start)>=edge_length_threshold):
                trueRedge_start = R1_edge_start
                trueRedge_end = edge_end[:,1][np.argwhere(num_edges==2)][edge_end[:,1][np.argwhere(num_edges==2)]>350]                

            elif(len(R1_edge_start)!=0 and len(R1_edge_start)<edge_length_threshold):
                # there are some elements but edge length is minimal
                trueRedge_start = R1_edge_start
                trueRedge_end = edge_end[:,1][np.argwhere(num_edges==2)][edge_end[:,1][np.argwhere(num_edges==2)]>350]

                min_R_edge_threshold = True

            else:
                print('max(num_edges)=3 invalid true right edge condition encountered . . . check code')   

        
        if(np.max(trueRedge_start)-np.min(trueRedge_start)>13):
            trueRedge_start = trueRedge_start[trueRedge_start <= (np.min(trueRedge_start)+10)]

        if(np.max(trueRedge_end)-np.min(trueRedge_end)>13):
            trueRedge_end = trueRedge_end[trueRedge_end <= (np.min(trueRedge_end)+10)]

        
        if(len(trueLedge_start)>len(trueLedge_end)):
            trueLedge_start = np.array([trueLedge_start[i] for i in range(len(trueLedge_end))])
        
        if(len(trueLedge_start)<len(trueLedge_end)):
            trueLedge_end = np.array([trueLedge_end[i] for i in range(len(trueLedge_start))])

        if(len(trueRedge_start)>len(trueRedge_end)):
            trueRedge_start = np.array([trueRedge_start[i] for i in range(len(trueRedge_end))])
        
        if(len(trueRedge_start)<len(trueRedge_end)):
            trueRedge_end = np.array([trueRedge_end[i] for i in range(len(trueRedge_start))])

        if(len(trueLedge_start)<edge_length_threshold):
            min_L_edge_threshold = True

        if(len(trueRedge_start)<edge_length_threshold):
            min_R_edge_threshold = True

        if(min_L_edge_threshold or min_R_edge_threshold):
            line1_start = (round(np.mean((trueLedge_start + trueLedge_end)/2)),0) 
            line1_end = (round(np.mean((trueLedge_start + trueLedge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((trueRedge_start + trueRedge_end)/2)),0)
            line2_end = (round(np.mean((trueRedge_start + trueRedge_end)/2)),img.shape[0])
                
            edge_count = 3
            case_cond = 2 

        elif(np.logical_and(len(trueLedge_start)>=edge_length_threshold,len(trueRedge_start)>=edge_length_threshold)):
                    
            line1_start = (round(np.mean((trueLedge_start + trueLedge_end)/2)),0) 
            line1_end = (round(np.mean((trueLedge_start + trueLedge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((trueRedge_start + trueRedge_end)/2)),0)
            line2_end = (round(np.mean((trueRedge_start + trueRedge_end)/2)),img.shape[0])
                
            edge_count = 3
            case_cond = 3        
            
        else:
            print('max(num_edges)=3 with no matching condition reached . . . check code')     
        
    
    elif edge_condition == 4: #max(num_edges)==4: 
        
        # max num_edges = 4
        # logic for finding true left edge                      
        L3_edge_start = edge_start[:,2][np.argwhere(num_edges==4)][edge_start[:,2][np.argwhere(num_edges==4)]<250]
        if(len(L3_edge_start)>=edge_length_threshold):
            trueLedge_start = L3_edge_start
            trueLedge_end = edge_end[:,2][np.argwhere(num_edges==4)][edge_end[:,2][np.argwhere(num_edges==4)]<250]
        else:
            L2_edge_start = edge_start[:,1][np.argwhere(num_edges==4)][np.logical_and(edge_start[:,1][np.argwhere(num_edges==4)]<250,edge_start[:,1][np.argwhere(num_edges==4)]>60)]
            L2_edge_start = np.hstack((L2_edge_start,edge_start[:,1][np.argwhere(num_edges==3)][edge_start[:,1][np.argwhere(num_edges==3)]<250]))
            if(len(L2_edge_start)>=edge_length_threshold):
                trueLedge_start = L2_edge_start

                trueLedge_end = edge_end[:,1][np.argwhere(num_edges==4)][np.logical_and(edge_end[:,1][np.argwhere(num_edges==4)]<250,edge_end[:,1][np.argwhere(num_edges==4)]>60)]
                trueLedge_end = np.hstack((trueLedge_end,edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]))
            
            else:
                L1_edge_start = edge_start[:,0][np.argwhere(num_edges==2)][edge_start[:,0][np.argwhere(num_edges==2)]<250]
                L1_edge_start = np.hstack((L1_edge_start,edge_start[:,0][np.argwhere(num_edges==3)][edge_start[:,0][np.argwhere(num_edges==3)]<250]))
                L1_edge_start = np.hstack((L1_edge_start,edge_start[:,0][np.argwhere(num_edges==4)][edge_start[:,0][np.argwhere(num_edges==4)]<250]))

                if(len(L1_edge_start)>= edge_length_threshold):
                    trueLedge_start = L1_edge_start

                    trueLedge_end = edge_end[:,0][np.argwhere(num_edges==2)][edge_end[:,0][np.argwhere(num_edges==2)]<250]
                    trueLedge_end = np.hstack((trueLedge_end,edge_end[:,0][np.argwhere(num_edges==3)][edge_end[:,0][np.argwhere(num_edges==3)]<250]))
                    trueLedge_end = np.hstack((trueLedge_end,edge_end[:,0][np.argwhere(num_edges==4)][edge_end[:,0][np.argwhere(num_edges==4)]<250]))
                else:
                    print('max(num_edges)=4 invalid true left edge condition encountered . . . check code')

        # logic for finding true right edge
        R3_edge_start = edge_start[:,1][np.argwhere(num_edges==4)][edge_start[:,1][np.argwhere(num_edges==4)]>350]
        if(len(R3_edge_start)>=edge_length_threshold):
            trueRedge_start = R3_edge_start
            trueRedge_end = edge_end[:,1][np.argwhere(num_edges==4)][edge_end[:,1][np.argwhere(num_edges==4)]>350]
        else:
            R2_edge_start = edge_start[:,2][np.argwhere(num_edges==4)][edge_start[:,2][np.argwhere(num_edges==4)]>350]
            R2_edge_start = np.hstack((R2_edge_start,edge_start[:,1][np.argwhere(num_edges==3)][edge_start[:,1][np.argwhere(num_edges==3)]>350]))
            if(len(R2_edge_start)>=edge_length_threshold):
                trueRedge_start = R2_edge_start

                trueRedge_end = edge_end[:,2][np.argwhere(num_edges==4)][edge_end[:,2][np.argwhere(num_edges==4)]>350]
                trueRedge_end = np.hstack((trueRedge_end,edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]>350]))

            else:
                R1_edge_start = edge_start[:,1][np.argwhere(num_edges==2)][edge_start[:,1][np.argwhere(num_edges==2)]>350]
                R1_edge_start = np.hstack((R1_edge_start,edge_start[:,2][np.argwhere(num_edges==3)][edge_start[:,2][np.argwhere(num_edges==3)]>350]))
                R1_edge_start = np.hstack((R1_edge_start,edge_start[:,3][np.argwhere(num_edges==4)][edge_start[:,3][np.argwhere(num_edges==4)]>350]))

                if(len(R1_edge_start)>= edge_length_threshold):
                    trueRedge_start = R1_edge_start

                    trueRedge_end = edge_end[:,1][np.argwhere(num_edges==2)][edge_end[:,1][np.argwhere(num_edges==2)]>350]
                    trueRedge_end = np.hstack((trueRedge_end,edge_end[:,2][np.argwhere(num_edges==3)][edge_end[:,2][np.argwhere(num_edges==3)]>350]))
                    trueRedge_end = np.hstack((trueRedge_end,edge_end[:,3][np.argwhere(num_edges==4)][edge_end[:,3][np.argwhere(num_edges==4)]>350]))
                else:
                    print('max(num_edges)=4 invalid true right edge condition encountered . . . check code')

        if(len(trueLedge_start)>len(trueLedge_end)):
            trueLedge_start = np.array([trueLedge_start[i] for i in range(len(trueLedge_end))])
        
        if(len(trueLedge_start)<len(trueLedge_end)):
            trueLedge_end = np.array([trueLedge_end[i] for i in range(len(trueLedge_start))])

        if(len(trueRedge_start)>len(trueRedge_end)):
            trueRedge_start = np.array([trueRedge_start[i] for i in range(len(trueRedge_end))])
        
        if(len(trueRedge_start)<len(trueRedge_end)):
            trueRedge_end = np.array([trueRedge_end[i] for i in range(len(trueRedge_start))])

        if(np.logical_and(len(trueLedge_start)>=edge_length_threshold,len(trueRedge_start)>=edge_length_threshold)):
                    
            line1_start = (round(np.mean((trueLedge_start + trueLedge_end)/2)),0) 
            line1_end = (round(np.mean((trueLedge_start + trueLedge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((trueRedge_start + trueRedge_end)/2)),0)
            line2_end = (round(np.mean((trueRedge_start + trueRedge_end)/2)),img.shape[0])
                
            edge_count = 4
            case_cond = 4        
            
        else:
            print('max(num_edges)=4 with no matching condition reached . . . check code')

    elif edge_condition > 4:

        # greater than 4 max edges case is typically - stringing or rother artifact causing psuedo edges
        # Identify true left edge

        L4_edge_start = edge_start[:,3][np.argwhere(num_edges==5)][edge_start[:,3][np.argwhere(num_edges==5)]<250]
        if(len(L4_edge_start)>=edge_length_threshold):
            trueLedge_start = L4_edge_start
            trueLedge_end = edge_end[:,3][np.argwhere(num_edges==5)][edge_end[:,3][np.argwhere(num_edges==5)]<250]
        else:
            L3_edge_start = edge_start[:,2][np.argwhere(num_edges==5)][edge_start[:,2][np.argwhere(num_edges==5)]<250]
            L3_edge_start = np.hstack((L3_edge_start,edge_start[:,2][np.argwhere(num_edges==4)][edge_start[:,2][np.argwhere(num_edges==4)]<250]))
            L3_edge_start = np.hstack((L3_edge_start,edge_start[:,1][np.argwhere(num_edges==3)][np.logical_and(edge_start[:,1][np.argwhere(num_edges==3)]<250,edge_start[:,1][np.argwhere(num_edges==3)]>60)]))
            
            if(len(L3_edge_start)>=edge_length_threshold):
                trueLedge_start = L3_edge_start

                trueLedge_end = edge_end[:,2][np.argwhere(num_edges==5)][edge_end[:,2][np.argwhere(num_edges==5)]<250]
                trueLedge_end = np.hstack((trueLedge_end,edge_end[:,2][np.argwhere(num_edges==4)][edge_end[:,2][np.argwhere(num_edges==4)]<250]))
                trueLedge_end = np.hstack((trueLedge_end,edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]))
            
            elif(len(L3_edge_start)!= 0 and len(L3_edge_start)<edge_length_threshold):
                trueLedge_start = L3_edge_start

                trueLedge_end = edge_end[:,2][np.argwhere(num_edges==5)][edge_end[:,2][np.argwhere(num_edges==5)]<250]
                trueLedge_end = np.hstack((trueLedge_end,edge_end[:,2][np.argwhere(num_edges==4)][edge_end[:,2][np.argwhere(num_edges==4)]<250]))
                trueLedge_end = np.hstack((trueLedge_end,edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]))

                min_L_edge_threshold = True

            else:
            
                # L2_edge_start = edge_start[:,1][np.argwhere(num_edges==3)][edge_start[:,1][np.argwhere(num_edges==3)]<250]
                # L2_edge_start = np.hstack((L2_edge_start,edge_start[:,0][np.argwhere(num_edges==3)][edge_start[:,0][np.argwhere(num_edges==3)]<250]))

                # if(len(L2_edge_start)>=edge_length_threshold):
                #     trueLedge_start = L2_edge_start

                #     trueLedge_end = edge_end[:,1][np.argwhere(num_edges==3)][edge_end[:,1][np.argwhere(num_edges==3)]<250]
                #     trueLedge_end = np.hstack((trueLedge_end,edge_end[:,0][np.argwhere(num_edges==3)][edge_end[:,0][np.argwhere(num_edges==3)]<250]))
                # else:
                print('max(num_edges)>4 invalid true left edge condition encountered . . . check code')


        # Identify true right edge
        sse_Redge_start = edge_start[:,3][np.argwhere(num_edges==5)][edge_start[:,3][np.argwhere(num_edges==5)]>350]
        sse_Redge_start = np.hstack((sse_Redge_start,edge_start[:,2][np.argwhere(num_edges==4)][edge_start[:,2][np.argwhere(num_edges==4)]>350]))

        if(len(sse_Redge_start)>=edge_length_threshold):
            trueRedge_start = sse_Redge_start

            trueRedge_end = edge_end[:,3][np.argwhere(num_edges==5)][edge_end[:,3][np.argwhere(num_edges==5)]>350]
            trueRedge_end = np.hstack((trueRedge_end,edge_end[:,2][np.argwhere(num_edges==4)][edge_end[:,2][np.argwhere(num_edges==4)]>350]))
        
        elif(len(sse_Redge_start)!= 0 and len(sse_Redge_start)<edge_length_threshold):
            trueRedge_start = sse_Redge_start

            trueRedge_end = edge_end[:,3][np.argwhere(num_edges==5)][edge_end[:,3][np.argwhere(num_edges==5)]>350]
            trueRedge_end = np.hstack((trueRedge_end,edge_end[:,2][np.argwhere(num_edges==4)][edge_end[:,2][np.argwhere(num_edges==4)]>350]))

            min_R_edge_threshold = True

        else:
            
            trueRedge_start = edge_start[:,3][np.argwhere(num_edges==4)][edge_start[:,3][np.argwhere(num_edges==4)]>350]
            trueRedge_start = np.hstack((trueRedge_start,edge_start[:,4][np.argwhere(num_edges==5)][edge_start[:,4][np.argwhere(num_edges==5)]>350]))
            
            trueRedge_end = edge_end[:,3][np.argwhere(num_edges==4)][edge_end[:,3][np.argwhere(num_edges==4)]>350]
            trueRedge_end = np.hstack((trueRedge_end,edge_end[:,4][np.argwhere(num_edges==5)][edge_end[:,4][np.argwhere(num_edges==5)]>350]))

        if(len(trueLedge_start)>len(trueLedge_end)):
            trueLedge_start = np.array([trueLedge_start[i] for i in range(len(trueLedge_end))])

        if(len(trueLedge_start)<len(trueLedge_end)):
            trueLedge_end = np.array([trueLedge_end[i] for i in range(len(trueLedge_start))])

        if(len(trueRedge_start)>len(trueRedge_end)):
            trueRedge_start = np.array([trueRedge_start[i] for i in range(len(trueRedge_end))])

        if(len(trueRedge_start)<len(trueRedge_end)):
            trueRedge_end = np.array([trueRedge_end[i] for i in range(len(trueRedge_start))])


        if(len(trueLedge_start)<edge_length_threshold):
            min_L_edge_threshold = True

        if(len(trueRedge_start)<edge_length_threshold):
            min_R_edge_threshold = True

        # Length check
        if(min_L_edge_threshold or min_R_edge_threshold):
            line1_start = (round(np.mean((trueLedge_start + trueLedge_end)/2)),0) 
            line1_end = (round(np.mean((trueLedge_start + trueLedge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((trueRedge_start + trueRedge_end)/2)),0)
            line2_end = (round(np.mean((trueRedge_start + trueRedge_end)/2)),img.shape[0])
            
            edge_count = 5
            case_cond = 5

        elif(np.logical_and(len(trueLedge_start)>=edge_length_threshold,len(trueRedge_start)>=edge_length_threshold)):

            line1_start = (round(np.mean((trueLedge_start + trueLedge_end)/2)),0) 
            line1_end = (round(np.mean((trueLedge_start + trueLedge_end)/2)),img.shape[0])

            line2_start = (round(np.mean((trueRedge_start + trueRedge_end)/2)),0)
            line2_end = (round(np.mean((trueRedge_start + trueRedge_end)/2)),img.shape[0])
            
            edge_count = 5
            case_cond = 6


        elif(len(edge_start[:,0][np.argwhere(num_edges==2)])>= edge_length_threshold):
                                        
            line1_start = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),0) 
            line1_end = (round(np.mean((edge_start[:,0][np.argwhere(num_edges==2)] + edge_end[:,0][np.argwhere(num_edges==2)])/2)),img.shape[0])

            line2_start = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),0)
            line2_end = (round(np.mean((edge_start[:,1][np.argwhere(num_edges==2)] + edge_end[:,1][np.argwhere(num_edges==2)])/2)),img.shape[0])
            
            edge_count = np.nan
            case_cond = 7
        
        else:
            print('max(num_edges)>4 with no matching condition reached . . . check code')

    else:
        print('Invalid edge condition reached . . . check code')

    # convert to BGR image and draw line
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # computed road width in pixels 
    dist_px = line2_start[0] - line1_start[0]
    dist_mm = round(dist_px*mm_per_pixel,4)
    
    cv2.line(img_color,line1_start,line1_end,(0,255,0),2)
    cv2.line(img_color,line2_start,line2_end,(0,255,0),2)
                       
    # Add Road width value to image
    
    # text
    text = str(dist_mm) + ' mm'
    
    if img_debug:
        print('w = ' + text)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (line1_start[0]+50, round(img.shape[0]/2))
  
    # fontScale
    fontScale = 1
   
    # Blue color in BGR
    color = (255, 0, 0)
  
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    img_color = cv2.putText(img_color, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)      
    
    return [case_cond,img_color,edge_start,edge_end,num_edges,edge_count,dist_px,dist_mm]

# Main Code
num_layers = 20
max_speed = 50
layers = list(range(5,num_layers+1))
#speeds = list(range(10,max_speed+10,10))
#speeds = [10,20,50,30,40]
#frame_skip_start = [32,20,11,15,13] # Start processesing from these frames for different vR - Skip frames before this

# For Debugging errors
layers = [20] # Restart in case of error
speeds = [40] # For testing only

#idx = [int((s/10)-1) for s in speeds]
#frame_start = [frame_skip_start[i] for i in idx]

bad_frames = [109] # Check this - enter probelmatic frame here - start processing from this frame
vidCount = 0
img_debug = True # Enable this flag to save pictures

w_result_columns=['Layer','vR','Frame','ActualTimestamp','CaseCondition','Bottom_Edges','w_Vision']
frame_summary_columns = ['Layer','vR','Start_TS','End_TS','Total_Frames','Per_Frames_Skipped','Skipped_Frames']
lst = []
lst_skip_frames = []
lst_frame_summary = []

# to calculate the transformation matrix
pt1 = [192.30,343.00]  # (x,y) - Top left
pt2 = [1079.0,379.80]  # (x,y) - Top right
pt3 = [153.50,571.90]  # (x,y) - bottom left
pt4 = [1107.10,611.70] # (x,y) - bottom Right

# Actual Dimensions of region selected by 4 points 
scale_width = round(11.7348*200) # mm Actual ruler width measured using calipers
scale_height = round(6.35*200)   # mm Height based on selected 4 points for perspective transform

# Extract top - bottom smaller regions
tStart = [655,0]
tEnd = [1300,345]
bStart = [655,925]
bEnd = [1300,1270]

k1_num_passes = 1 # Default should be 2
run = 'Run-2'

for l in range(len(layers)):
    for v in range(len(speeds)):
        
        lst = []
        lst_skip_frames = []
                
        vidName = 'vid_l'+str(layers[l])+'_vR_'+str(speeds[v])+'.avi'
        print('Processing video: ' + vidName)
        
        idx = df_vidTimes.index[(df_vidTimes.Layer==layers[l]) & (df_vidTimes.Speed==speeds[v])].to_list()[0]
        start_TS = df_vidTimes.Start_Timestamp[idx]
        end_TS = df_vidTimes.End_Timestamp[idx]
        
        print('video: {0} starts at {1} and ends at {2}'.format(vidName,start_TS,end_TS))
        
        srcVideo = SRC_FOLDER + vidName
        #print('video location: ' + srcVideo)
        
        cap = cv2.VideoCapture(srcVideo)
        numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('video {0} has {1} frames'.format(vidName,numFrames))

        # check if video opened successfully
        if (cap.isOpened() == False):
            print("Error reading video file. Exiting ...")
            exit(0)
            
        frameCount = 0

        while(cap.isOpened()):
    
            frame_exists, frame = cap.read()
               
            if frame_exists:
                
                frameCount = frameCount + 1
                
                if(frameCount in bad_frames):
                #if(frameCount >= frame_skip_start[v] and frameCount <= numFrames - 5): 
                    
                    try:
                    
                        #print('Begin processing frame {0}'.format(frameCount))                            

                        # call function - correct perspective transform
                        [img_bgr,imgPersp,imgGrayPersp] = perspCorrection(frame,pt1,pt2,pt3,pt4,scale_width,scale_height)
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # Filter grayscale image
                        # Bilateral filter
                        bfblur = bilateralFilter(imgGrayPersp)

                        # ROI Extraction - Mark rectangle
                        # convert to RGB image and draw line
                        img_ROI = cv2.cvtColor(imgGrayPersp, cv2.COLOR_GRAY2RGB)
                        img_ROI = cv2.rectangle(img_ROI, (bStart[0],bStart[1]), (bEnd[0],bEnd[1]), (255,0,0), 8)
                        #img_ROI = cv2.rectangle(img_ROI, (tStart[0],tStart[1]), (tEnd[0],tEnd[1]), (0,0,255), 8)                        

                        [img_top,img_bottom] = extractTopBottom(bfblur,tStart,tEnd,bStart,bEnd)

                        # Thresholding - Adaptive Gaussian 
                        #thresh_top = gAdaptiveThresholding(img_top)
                        thresh_bottom = gAdaptiveThresholding(img_bottom)

                        #dstPathTop = 'Perspective Corrected\\Top\\'
                        #cv2.imwrite(dstPathTop+'top'+str(i+1)+'.jpg',img_top)    

                        # create kernel 
                        kernel1 = np.ones((8,2),np.uint8)
                        kernel2 = np.ones((5,2),np.uint8)

                        # perform morph operations    
                        #binImgTop=morphOps(thresh_top,kernel1,kernel2)
                        binImgBtm=morphOps(thresh_bottom,kernel1,kernel2,k1_num_passes)

                        # save images - for analysis
                        if(img_debug):
                            pCorrImg_savePath = SRC_FOLDER + "Results/" + run + "/" + str(layers[l]) + "/" + str(speeds[v]) + "/Extracted Images/Gray/"
                            #bfImg_savePath = SRC_FOLDER + "Extracted Images/" + str(layers[l]) + "/" + str(speeds[v]) + "/Bilateral Filter/"
                            #bROIImg_savePath = SRC_FOLDER + "Extracted Images/" + str(layers[l]) + "/" + str(speeds[v]) + "/ROI/"
                            threshImg_savePath = SRC_FOLDER + "Results/" + run + "/" + str(layers[l]) + "/" + str(speeds[v]) + "/Extracted Images/Gray/"
                            binImg_savePath = SRC_FOLDER + "Results/" + run + "/" + str(layers[l]) + "/" + str(speeds[v]) + "/Binary/"

                            cv2.imwrite(pCorrImg_savePath + "pCorr" + str(frameCount) + ".jpg", imgGrayPersp)
                            #cv2.imwrite(bfImg_savePath + "bFil" + str(frameCount) + ".jpg", bfblur)
                            #cv2.imwrite(bROIImg_savePath + "btm_ROI" + str(frameCount) + ".jpg", img_bottom)
                            cv2.imwrite(threshImg_savePath + "thresh" + str(frameCount) + ".jpg", thresh_bottom)                        
                            cv2.imwrite(binImg_savePath + "binary" + str(frameCount) + ".jpg", binImgBtm)

                        # Extrusion width measurement 
                        #[top_img_color,top_edge_start,top_edge_end,top_num_edges,top_edge_count,top_edge_dist_pixels,top_edge_dist] = computeW(binImgTop)
                        [bottom_case_cond,bottom_img_color,bottom_edge_start,bottom_edge_end,bottom_num_edges,bottom_edge_count,bottom_edge_dist_pixels,bottom_edge_dist] = computeW_Rev(binImgBtm,img_debug)
                        
                        # horizontally concatenates images of same height 
                        img_h = cv2.hconcat([cv2.cvtColor(binImgBtm, cv2.COLOR_GRAY2BGR), bottom_img_color])                       
                        
                        # save image - for analysis
                        if(img_debug):
                            wImg_savePath = SRC_FOLDER + "Results/" + run + "/" + str(layers[l]) + "/" + str(speeds[v]) + "/Vision Measurements/"
                            cv2.imwrite(wImg_savePath + "wImg" + str(frameCount) + ".jpg", bottom_img_color)

                            hImg_savePath = SRC_FOLDER + "Results/" + run + "/" + str(layers[l]) + "/" + str(speeds[v]) + "/Summary/"
                            cv2.imwrite(hImg_savePath + "wS_Img" + str(frameCount) + ".jpg", img_h)

                        # Calculate actual timestamp based on excel timestamps and frame number
                        act_TS = start_TS+frameCount*(end_TS-start_TS)/numFrames

                        # Store results in dataframe   
                        lst.append([layers[l],speeds[v],frameCount,act_TS,bottom_case_cond,bottom_edge_count,bottom_edge_dist])
                        
                        #print('Finished processing frame {0}'.format(frameCount))
                        
                    except ValueError as e:
                        #if (len(e.args) > 0 and e.args[0] == 'cannot convert float NaN to integer'):
                        print('Unable to sucessfully process frame {0}, skipping . . .'.format(frameCount))
                        print(e)
                        # Calculate actual timestamp based on excel timestamps and frame number
                        act_TS = start_TS+frameCount*(end_TS-start_TS)/numFrames
                        # Store results in dataframe   
                        lst.append([layers[l],speeds[v],frameCount,act_TS,np.nan,np.nan,np.nan])
                        lst_skip_frames.append([frameCount])

                    except UnboundLocalError as u:
                        print('Unable to sucessfully process frame {0}, skipping . . .'.format(frameCount))
                        print(u)
                        # Calculate actual timestamp based on excel timestamps and frame number
                        act_TS = start_TS+frameCount*(end_TS-start_TS)/numFrames
                        # Store results in dataframe   
                        lst.append([layers[l],speeds[v],frameCount,act_TS,np.nan,np.nan,np.nan])
                        lst_skip_frames.append([frameCount])

            else:
                #print("Can't retrieve frame - stream may have ended. Exiting..")
                break       
                   
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        print('Finished processing video: {0}'.format(vidName))
        print('')
        print('')
        vidCount = vidCount + 1

        if not img_debug:
                     
            results = pd.DataFrame(lst,columns=w_result_columns)
            # Save results to excel
            with pd.ExcelWriter(
                SRC_FOLDER + 'Results/' + run + '/' + 'l' + str(layers[l])+'_vR'+str(speeds[v])+ '_results.xlsx',
                date_format="YYYY-MM-DD",
                datetime_format="YYYY-MM-DD HH:MM:SS.000"
            ) as writer:
                results.to_excel(writer,index=False)
            
            #print('Saved results of video {0} at {1}'.format(vidName,SRC_FOLDER + 'l' + str(layers[l])+'_vR'+str(speeds[v])+ '_results.xlsx'))
            
            lst_frame_summary.append([layers[l],speeds[v],start_TS,end_TS,numFrames,len(lst_skip_frames)/numFrames,lst_skip_frames])

if not img_debug:
    frame_summary_results = pd.DataFrame(lst_frame_summary,columns=frame_summary_columns)

    # Some more cleanup and data addition
    frame_summary_results["Video_Duration"] = frame_summary_results["End_TS"] - frame_summary_results["Start_TS"] 
    frame_summary_results["Video_Duration"] = [x.total_seconds() for x in frame_summary_results["Video_Duration"]]
    frame_summary_results["FPS"] = frame_summary_results["Total_Frames"]/frame_summary_results["Video_Duration"]
    frame_summary_results["Total_Frames_Skipped"] = [len(x) for x in frame_summary_results["Skipped_Frames"]]
    # Re-oder columns
    frame_summary_results = frame_summary_results[["Layer","vR","Start_TS","End_TS","Video_Duration","Total_Frames","FPS","Total_Frames_Skipped","Per_Frames_Skipped","Skipped_Frames"]]

    with pd.ExcelWriter(
                SRC_FOLDER + 'Results/' + run + '/' + 'video_processing_summary.xlsx',
                date_format="YYYY-MM-DD",
                datetime_format="YYYY-MM-DD HH:MM:SS.000"
            ) as writer:
                frame_summary_results.to_excel(writer,index=False)
    print('Processing of all videos completed successfully! Summary results saved at {0}'.format(SRC_FOLDER + 'video_processing_summary.xlsx'))
