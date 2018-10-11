import sys
import os
import dlib
import glob
import skimage
from skimage import io
from tqdm import tqdm


class Detective(object):
    def __init__(self, predictor_path="src/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


    def getImages(self, path, ext=["*jpg"], template=None):
        self.files = []
        for e in ext:
            self.files.extend(glob.glob(os.path.join(path, e)))
        if template != None:
            index = [i for i, s in enumerate(self.files) if template in s]
            assert index != [], "> Template '{t}' name was not found".format(t=template)
            index = index[0]
            self.files = [self.files[index]] + self.files[0:index] + self.files[index+1:]
        return self

    # returns image, d
    def imageFeatures(self, imgPath, useCaching=False):
        cacheFailure = not useCaching
        if useCaching:
            faces = []
            if os.path.exists(imgPath + '.ff'):
                with open(imgPath + '.ff', 'r') as file:
                    file = file.readlines()
                    i = 0
                    numFaces = int(file[i])
                    for _ in range(numFaces):
                        i += 1
                        l, t, r, b = file[i].split()
                        box = (l, t, r, b)
                        shapeList = []
                        for _ in range(68):
                            i += 1
                            x, y = file[i].split()
                            shapeList.append((int(x), int(y)))

                        faces.append({
                            "imgPath": imgPath,
                            "img":None,
                            "box": box,
                            "shape": shapeList,
                        }) 
            else:
                cacheFailure = True

        if cacheFailure is True:
            img = io.imread(imgPath)
            faces = []
            dets = self.detector(img, 1)
            toFile = str(len(dets)) + '\n'
            for i,d in enumerate(dets):
                toFile += f"{d.left()} {d.top()} {d.right()} {d.bottom()}\n"

                shape = self.predictor(img, d)
                for i in range(0, 68):
                    toFile += str(shape.part(i))[1:-1].replace(',', '') + '\n'

                
                faces.append({
                    "imgPath": imgPath,
                    "img":img,
                    "box": ((d.left(), d.top()), (d.right(), d.bottom())),
                    "shape": [(int(s.x), int(s.y)) for s in shape.parts()],
                })

            if useCaching:
                with open(imgPath + '.ff', 'w') as output:
                    output.write(toFile)
            
        return faces
        
    def features(self, useCaching=True):
        self.detections = []
        with tqdm(self.files) as pbar:
            for f in pbar:
                det = self.imageFeatures(f, useCaching=useCaching)
                if len(det) == 0:
                    print(f)
                pbar.set_description(f"Faces detected: {len(det)} in ...{det[0]['imgPath'][-10:]}")
                self.detections.extend(det)
                    
            
        return self


