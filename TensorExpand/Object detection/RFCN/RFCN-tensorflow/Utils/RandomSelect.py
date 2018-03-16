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

def randomSelectIndex(fromCount, n):
	with tf.name_scope("randomSelectIndex"):
		n = tf.minimum(fromCount, n)
		i = tf.random_shuffle(tf.range(fromCount, dtype=tf.int32))[0:n]
		return tf.expand_dims(i,-1)

def randomSelectBatch(t, n):
	with tf.name_scope("randomSelectBatch"):
		count = tf.shape(t)[0]
		return tf.gather_nd(t, randomSelectIndex(count,n))
