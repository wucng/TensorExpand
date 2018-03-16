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

import tensorflow as tf 
import BoxEngine.BoxUtils as BoxUtils

def smooth_l1(x):
    with tf.name_scope("smooth_l1"):
        abs_x = tf.abs(x)
        lessMask = tf.cast(abs_x < 1.0, tf.float32)

        return lessMask * (0.5 * tf.square(x)) + (1.0 - lessMask) * (abs_x - 0.5)

def reshapeAll(l, shape=[-1]):
    with tf.name_scope("reshapeAll"):
        res=[]
        for e in l:
            res.append(tf.reshape(e, shape))
        return res

def boxRegressionLoss(boxes, rawSizes, refBoxes, boxSizes):
    with tf.name_scope("rawBoxRegressionLoss"):
        x, y, w, h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(boxes, axis=1))
        wRel, hRel = tf.unstack(rawSizes, axis=1)
        boxH, boxW = tf.unstack(boxSizes, axis=1)
        ref_x, ref_y, ref_w, ref_h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(refBoxes, axis=1))

        x,y,wRel,hRel, boxH,boxW, ref_x,ref_y,ref_w,ref_h = reshapeAll([x,y,wRel,hRel, boxH,boxW, ref_x,ref_y,ref_w,ref_h])
        
        wrelRef = tf.log(ref_w/boxW)
        hrelRef = tf.log(ref_h/boxH)

        # Smooth L1 loss is defined on NN output values, but only the box sizes are available here. However
        # we can transform back the coordinates in a numerically stable way in the NN output space:
        #
        # tx-tx' = (x-x')/wa

        return smooth_l1((x-ref_x)/boxW) + smooth_l1((y-ref_y)/boxH) + smooth_l1(wRel - wrelRef) + smooth_l1(hRel - hrelRef)
        