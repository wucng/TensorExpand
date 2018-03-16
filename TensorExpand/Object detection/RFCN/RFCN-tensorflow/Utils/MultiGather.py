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

def gather(tensors, indices):
    with tf.name_scope("multiGather"):
        res = []
        for a in tensors:
            res.append(tf.gather_nd(a, indices))
        return res

def gatherTopK(t, k, others=[], sorted=False):
    res=[]
    with tf.name_scope("gather_top_k"):
        isMoreThanK = tf.shape(t)[-1]>k
        values, indices = tf.cond(isMoreThanK, lambda: tuple(tf.nn.top_k(t, k=k, sorted=sorted)), lambda: (t, tf.zeros((0,1), tf.int32)))
        indices = tf.reshape(indices, [-1,1])
        res.append(values)

        for o in others:
            res.append(tf.cond(isMoreThanK, lambda: tf.gather_nd(o, indices), lambda: o))

    return res
