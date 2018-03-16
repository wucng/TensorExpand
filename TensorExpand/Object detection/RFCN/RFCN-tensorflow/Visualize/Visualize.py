# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import cv2
import numpy as np

class Palette:
    @staticmethod
    def bitShift(v,s):
        if s>0:
            return v<<s
        else:
            return v>>(-s)
 
    def setFixColor(self, category, color):
        if len(self.cmap)<=category:
            return
 
        if self.bgr:
            self.cmap[category][2]=color[0]
            self.cmap[category][1]=color[1]
            self.cmap[category][0]=color[2]
        else:
            self.cmap[category]=color
 
    def modify(self, p):
        for k in p:
            self.setFixColor(k, p[k])
 
    def __init__(self, N, bgr=True, modifier=None):
        self.list=None
        self.bgr=bgr
        self.cmap=np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            id=(i+1)
            r=0
            g=0
            b=0
 
            for j in range(8):
                r=r | Palette.bitShift((id >> 0) & 1, 7-j)
                g=g | Palette.bitShift((id >> 1) & 1, 7-j)
                b=b | Palette.bitShift((id >> 2) & 1, 7-j)
                id = Palette.bitShift(id, -3)
 
            if bgr:
                self.cmap[i]=[b,g,r]
            else:
                self.cmap[i]=[r,g,b]
 
        if modifier is not None:
            self.modify(modifier)
 
    def getMap(self, list=False):
        if list:
            if self.list is None:
                self.list=self.cmap.tolist()
            return self.list
        else:
            return self.cmap

def drawBoxes(img, boxes, categories, names, palette, scores=None, fade=False):
    def clipCoord(xy):
        return np.minimum(np.maximum(np.array(xy,dtype=np.int32),0),[img.shape[1]-1, img.shape[0]-1]).tolist()

    cmap = palette.getMap(list=True)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
    fontSize = 0.8
    fontThickness = 1
    pad=5


    img=np.copy(img)

    for box in range(boxes.shape[0]):
        if fade and scores is not None:
            iOrig = img
            img=np.copy(img)

        topleft = tuple(clipCoord(boxes[box][0:2]))
        if categories is not None:
            color = tuple(cmap[categories[box]])
        else:
            color = (0,0,255)
        cv2.rectangle(img, topleft, tuple(clipCoord(boxes[box][2:5])), color, thickness=4)
        if names:
            title=names[box]
            if scores is not None:
                title+=": %.2f" % scores[box]
            textpos=[topleft[0], topleft[1]-pad]
            size = cv2.getTextSize(title, font, fontSize, fontThickness)[0]

            boxTL = textpos[:]
            boxTL[1] = boxTL[1] - size[1]
            boxBR = list(topleft)
            boxBR[0] = boxBR[0] + size[0]
            
            cv2.rectangle(img, tuple(boxTL), tuple(boxBR), color, thickness=-1)
            cv2.rectangle(img, tuple(boxTL), tuple(boxBR), color, thickness=4)
            cv2.putText(img, title, tuple(textpos), font, fontSize, (255,255,255), thickness=fontThickness)
        
        if fade and scores is not None:
            img = scores[box] * img + (1.0-scores[box]) * iOrig
    return img

def tile(cols, rows, imgs, titles=None):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fontSize = 1
    fontThickness = 2
    pad=10
    titleColor = (255,192,0)

    hImg = imgs[0]
    i = 0
    z = None
    row = []
    for c in range(cols):
        col = []
        for r in range(rows):
            if i<len(imgs):
                img = imgs[i]
                if titles is not None and i<len(titles):
                    img = img.copy()
                    size = cv2.getTextSize(titles[i], font, fontSize, fontThickness)[0]
                    cv2.putText(img, titles[i], (pad, size[1]+pad), font, fontSize, titleColor, thickness=fontThickness)

                col.append(img)
            else:
                if z is None:
                    z = np.zeros_like(imgs[0])
                col.append(z)
            i+=1
        row.append(np.concatenate(col, axis=0))

    return np.concatenate(row, axis=1)
