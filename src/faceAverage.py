import os
import cv2
import numpy as np
import math
from PIL import Image
import sys
from faceFeaturesDetector import Detective
from tqdm import tqdm
import dlib
import time
import matplotlib.pyplot as plt
from skimage import io

class Averager(object): 

    def __init__(self, width=600, height=800):
        self.width = width
        self.height = height
        self.detective = Detective()

    def loadImages(self, detections):
        pbar = tqdm(range(len(detections)))
        for i in pbar:
            pbar.set_description(f"Loading: ...{detections[i]['imgPath'][-22:]}")
            if detections[i]['img'] is None:
                detections[i]['img'] = io.imread(detections[i]['imgPath'])
        
        return [np.float32(im['img'])/255.0 for im in detections]


    def run(self, path, ext=['*.jpg','*.jpeg'], window=False, windowTime=500, showWarps=False, useCaching=True):
        
        self.inputpath = path
        self.images = self.detective.getImages(path, ext=ext).features(useCaching=useCaching).detections
        w, h = self.width, self.height
        
        allPoints = [im['shape'] for im in self.images]
        images = self.loadImages(self.images)

        # Eye corners
        eyecornerDst = [ (np.int(0.38 * w ), np.int(h / 2.5)), (np.int(0.62 * w ), np.int(h / 2.5)) ]

        imagesNorm = []
        pointsNorm = []
        
        # Add boundary points for delaunay triangulation
        boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
        
        # Initialize location of average points to 0s
        pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32())
        
        n = len(allPoints[0])

        numImages = len(images)
        
        # Warp images and trasnform landmarks to output coordinate system,
        # and find average of transformed landmarks.
        pbar = tqdm(images)
        for i, _ in enumerate(pbar):
            pbar.set_description(f"Warping: ...{self.images[i]['imgPath'][-22:]}")
            points1 = allPoints[i]

            # Corners of the eye in input image
            eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] 
            
            # Compute similarity transform
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst)


            # Apply similarity transformation
            img = cv2.warpAffine(images[i], tform, (w,h))

            # Apply similarity transform on points
            points2 = np.reshape(np.array(points1), (68,1,2))        
            
            points = cv2.transform(points2, tform)
            
            points = np.float32(np.reshape(points, (68, 2)))
            
            # Append boundary points. Will be used in Delaunay Triangulation
            points = np.append(points, boundaryPts, axis=0)
            
            # Calculate location of average landmark points.
            pointsAvg = pointsAvg + points / numImages
            
            pointsNorm.append(points)
            imagesNorm.append(img)
        

        
        # Delaunay triangulation
        rect = (0, 0, w, h)
        dt = self.calculateDelaunayTriangles(rect, np.array(pointsAvg))

        # Output image
        output = np.zeros((h,w,3), np.float32())


        if window:
            if showWarps:
                realImgWaitTime = int(windowTime * 0.25)
                warpImgWaitTime = windowTime - realImgWaitTime
            else:
                realImgWaitTime = windowTime


            cv2.namedWindow('Face Average', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Face Average', w*2, h)
            cv2.waitKey(1000)
            # win = dlib.image_window()

        # Warp input images to average image landmarks
        for i in range(0, len(imagesNorm)) :
            img = np.zeros((h,w,3), np.float32())
            # Transform triangles one by one
            for j in range(0, len(dt)) :
                tin = [] 
                tout = []
                
                for k in range(0, 3) :                
                    pIn = pointsNorm[i][dt[j][k]]
                    pIn = self.constrainPoint(pIn, w, h)
                    
                    pOut = pointsAvg[dt[j][k]]
                    pOut = self.constrainPoint(pOut, w, h)
                    
                    tin.append(pIn)
                    tout.append(pOut)
            

                self.warpTriangle(imagesNorm[i], img, tin, tout)
            
            if window:
                oldImg = cv2.cvtColor(imagesNorm[i], cv2.COLOR_BGR2RGB)
                resultImg = cv2.cvtColor(output / (i+1), cv2.COLOR_BGR2RGB)
                theimg = np.hstack((resultImg, oldImg))
                cv2.imshow('Face Average', theimg)
                cv2.waitKey(realImgWaitTime)
                if showWarps:
                    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    theimg = np.hstack((resultImg, newImg))
                    cv2.imshow('Face Average', theimg)
                    cv2.waitKey(warpImgWaitTime)




            # Add image intensities for averaging
            output = output + img


        # Divide by numImages to get average
        output = output / numImages
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # Display result
        if window:
            cv2.resizeWindow('Face Average', w, h)
            cv2.imshow('Face Average', output)
            cv2.waitKey(10000)

        self.result = output * 255


        return self
    
    def save(self, name=None):
        if name is None:
            if 'datasets' in self.inputpath:
                name = '-'.join(self.inputpath.split('datasets')[1].split('/'))[1:]
                name = './results/' + name + '.jpg'
            else:
                name = './results/' + self.inputpath.split('/')[-1] + '.jpg'

        
        cv2.imwrite(name, self.result)
        return self


    def similarityTransform(self, inPoints, outPoints) :
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)  
    
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        
        return tform


    # Check if a point is inside a rectangle
    def rectContains(self, rect, point) :
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True

    # Calculate delanauy triangle
    def calculateDelaunayTriangles(self, rect, points):
        # Create subdiv
        subdiv = cv2.Subdiv2D(rect)
    
        # Insert points into subdiv
        for p in points:
            subdiv.insert((p[0], p[1]))

    
        # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
        triangleList = subdiv.getTriangleList()

        # Find the indices of triangles in the points array

        delaunayTri = []
        
        for t in triangleList:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))
            
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])        
            
            if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):                    
                        if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)                            
                if len(ind) == 3:                                                
                    delaunayTri.append((ind[0], ind[1], ind[2]))
            

        
        return delaunayTri


    def constrainPoint(self, p, w, h) :
        p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
        return p

    # Apply affine transform calculated using srcTri and dstTri to src and
    # output an image of size.
    def applyAffineTransform(self, src, srcTri, dstTri, size) :
        
        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
        
        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

        return dst


    # Warps and alpha blends triangular regions from img1 and img2 to img
    def warpTriangle(self, img1, img2, t1, t2) :

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = [] 
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        
        size = (r2[2], r2[3])

        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        
        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

