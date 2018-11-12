#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])


    def detect_l1_xy(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 1, colour:(102,102,102) in xy plane
        #Obtain the center of link 1
        mask = cv2.inRange(image, (101,101,101),(104,104,104))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = np.array([cx,cy])
        #apply the distance transform
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        #step is how the degree increment to step through in the search
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                #Rotate the template to the desired rotation configuration
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                #Isolate the region of interest in the distance image
                ROI = dist[(cy-self.env.rod_template_1.shape[0]/2):(cy+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                #Apply rotation to the template
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)
                #Combine the template and region of interest together to obtain only the values that are inside the template
                img = ROI*rotatedTemplate
                #Sum the distances and append to the list
                sumlist = np.append(sumlist,np.sum(img))
            #Once all configurations have been searched then select the one with the smallest distance and convert to radians.
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi (REPEAT THE SAME AS ABOVE JUST WITH DIFFERENT LIMITS)
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_1.shape[0]/2):(cy+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_1.shape[0]/2):(cy+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                #cv2.imshow('tmp',img)
                #cv2.waitKey(5)
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_1.shape[0]/2):(cy+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)

    def detect_l1_xz(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 1, colour:(102,102,102) in xy plane
        #Obtain the center of link 1
        mask = cv2.inRange(image, (101,101,101),(104,104,104))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cz = int(M['m01']/M['m00'])
        center = np.array([cx,cz])
        #apply the distance transform
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        #step is how the degree increment to step through in the search
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                #Rotate the template to the desired rotation configuration
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                #Isolate the region of interest in the distance image
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                #Apply rotation to the template
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)
                #Combine the template and region of interest together to obtain only the values that are inside the template
                img = ROI*rotatedTemplate
                #Sum the distances and append to the list
                sumlist = np.append(sumlist,np.sum(img))
            #Once all configurations have been searched then select the one with the smallest distance and convert to radians.
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi (REPEAT THE SAME AS ABOVE JUST WITH DIFFERENT LIMITS)
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                #cv2.imshow('tmp',img)
                #cv2.waitKey(5)
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)

    def detect_l2_xy(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 2, colour:(51,51,51)
        #SAME AS ABOVE METHOD JUST WITH DIFFERENT COLOUR LIMITS
        mask = cv2.inRange(image, (50,50,50),(52,52,52))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = np.array([cx,cy])
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_2.shape[0]/2,self.env.rod_template_2.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_2.shape[0]/2):(cy+self.env.rod_template_2.shape[0]/2)+1,(cx-self.env.rod_template_2.shape[1]/2):(cx+self.env.rod_template_2.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_2,M,self.env.rod_template_2.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_2.shape[0]/2,self.env.rod_template_2.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_2.shape[0]/2):(cy+self.env.rod_template_2.shape[0]/2)+1,(cx-self.env.rod_template_2.shape[1]/2):(cx+self.env.rod_template_2.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_2,M,self.env.rod_template_2.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_2.shape[0]/2,self.env.rod_template_2.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_2.shape[0]/2):(cy+self.env.rod_template_2.shape[0]/2)+1,(cx-self.env.rod_template_2.shape[1]/2):(cx+self.env.rod_template_2.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_2,M,self.env.rod_template_2.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_2.shape[0]/2,self.env.rod_template_2.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_2.shape[0]/2):(cy+self.env.rod_template_2.shape[0]/2)+1,(cx-self.env.rod_template_2.shape[1]/2):(cx+self.env.rod_template_2.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_2,M,self.env.rod_template_2.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)

    def detect_l2_xz(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 1, colour:(102,102,102) in xy plane
        #Obtain the center of link 1
        mask = cv2.inRange(image, (50,50,50),(52, 52, 52))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cz = int(M['m01']/M['m00'])
        center = np.array([cx,cz])
        #apply the distance transform
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        #step is how the degree increment to step through in the search
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                #Rotate the template to the desired rotation configuration
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                #Isolate the region of interest in the distance image
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                #Apply rotation to the template
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)
                #Combine the template and region of interest together to obtain only the values that are inside the template
                img = ROI*rotatedTemplate
                #Sum the distances and append to the list
                sumlist = np.append(sumlist,np.sum(img))
            #Once all configurations have been searched then select the one with the smallest distance and convert to radians.
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi (REPEAT THE SAME AS ABOVE JUST WITH DIFFERENT LIMITS)
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                #cv2.imshow('tmp',img)
                #cv2.waitKey(5)
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)


    def detect_l3_xy(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 3, colour:(0,0,0)
        #SAME AS DETECT_L1 METHOD JUST WITH DIFFERENT COLOUR LIMITS
        mask = cv2.inRange(image, (0,0,0),(2,2,2))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = np.array([cx,cy])
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)

    def detect_l3_xz(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 1, colour:(102,102,102) in xy plane
        #Obtain the center of link 1
        mask = cv2.inRange(image, (0,0,0),(2,2,2))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cz = int(M['m01']/M['m00'])
        center = np.array([cx,cz])
        #apply the distance transform
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        #step is how the degree increment to step through in the search
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                #Rotate the template to the desired rotation configuration
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                #Isolate the region of interest in the distance image
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                #Apply rotation to the template
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)
                #Combine the template and region of interest together to obtain only the values that are inside the template
                img = ROI*rotatedTemplate
                #Sum the distances and append to the list
                sumlist = np.append(sumlist,np.sum(img))
            #Once all configurations have been searched then select the one with the smallest distance and convert to radians.
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi (REPEAT THE SAME AS ABOVE JUST WITH DIFFERENT LIMITS)
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                #cv2.imshow('tmp',img)
                #cv2.waitKey(5)
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)


    def detect_l4_xy(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 3, colour:(0,0,0)
        #SAME AS DETECT_L1 METHOD JUST WITH DIFFERENT COLOUR LIMITS
        mask = cv2.inRange(image, (126,126,126),(129,129,129))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = np.array([cx,cy])
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_3.shape[0]/2,self.env.rod_template_3.shape[1]/2),i,1)
                ROI = dist[(cy-self.env.rod_template_3.shape[0]/2):(cy+self.env.rod_template_3.shape[0]/2)+1,(cx-self.env.rod_template_3.shape[1]/2):(cx+self.env.rod_template_3.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_3,M,self.env.rod_template_3.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)

    def detect_l4_xz(self,image,quadrant):
        #In this method you can focus on detecting the rotation of link 1, colour:(102,102,102) in xy plane
        #Obtain the center of link 1
        mask = cv2.inRange(image, (126,126,126), (129,129,129))
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cz = int(M['m01']/M['m00'])
        center = np.array([cx,cz])
        #apply the distance transform
        dist = cv2.distanceTransform(cv2.bitwise_not(mask),cv2.DIST_L2,0)
        sumlist = np.array([])
        #step is how the degree increment to step through in the search
        step = 0.5
        if quadrant == "UR":
            #should be between 0 and math.pi/2
            for i in np.arange(0,90,step):
                #Rotate the template to the desired rotation configuration
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                #Isolate the region of interest in the distance image
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                #Apply rotation to the template
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)
                #Combine the template and region of interest together to obtain only the values that are inside the template
                img = ROI*rotatedTemplate
                #Sum the distances and append to the list
                sumlist = np.append(sumlist,np.sum(img))
            #Once all configurations have been searched then select the one with the smallest distance and convert to radians.
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step))*math.pi)/180.0)
        elif quadrant == "UL":
            #should be between math.pi/2 and math.pi (REPEAT THE SAME AS ABOVE JUST WITH DIFFERENT LIMITS)
            for i in np.arange(90+step,180,step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*step)+90+step)*math.pi)/180.0)
        elif quadrant == "LR":
            #should be between -0 and -math.pi/2
            for i in np.arange(-step,-90,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                #cv2.imshow('tmp',img)
                #cv2.waitKey(5)
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-step)*math.pi)/180.0)
        elif quadrant == "LL":
            #should be between -math.pi/2 and -math.pi
            for i in np.arange(-90-step,-179.5,-step):
                M = cv2.getRotationMatrix2D((self.env.rod_template_1.shape[0]/2,self.env.rod_template_1.shape[1]/2),i,1)
                ROI = dist[(cz-self.env.rod_template_1.shape[0]/2):(cz+self.env.rod_template_1.shape[0]/2)+1,(cx-self.env.rod_template_1.shape[1]/2):(cx+self.env.rod_template_1.shape[1]/2)+1]
                rotatedTemplate = cv2.warpAffine(self.env.rod_template_1,M,self.env.rod_template_1.shape)

                img = ROI*rotatedTemplate
                sumlist = np.append(sumlist,np.sum(img))
            return (self.coordinate_convert(center),(((np.argmin(sumlist)*-step)-90-step)*math.pi)/180.0)






    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="TORQUE"
        #Run 100000 iterations
        prev_JAs = np.zeros(3)
        prev_jvs = collections.deque(np.zeros(3),1)

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        for _ in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')

            jointAngles = np.array([0.5,-0.5,0.5,-0.5])
            # self.env.step((np.zeros(3),np.zeros(3),jointAngles, np.zeros(3)))
            self.env.step((np.zeros(3),np.zeros(3),np.zeros(3), np.zeros(4)))
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
