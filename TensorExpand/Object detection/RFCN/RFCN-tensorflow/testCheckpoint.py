#!/usr/bin/python
#
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
import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Checkpoint tester.')
parser.add_argument('-stats', type=int, default=0, help='Enable statistics')
parser.add_argument('-n', type=str, default="", help='Network checkpoint')
opt=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""


reader = tf.contrib.framework.load_checkpoint(opt.n)

sumSize=0.0

sizes = []
names = []

sMap = reader.get_variable_to_shape_map()

for v, s in sMap.items():
	size=0.0
	if len(s)>0:
		size = np.prod(s)
	
	size *= 4.0
	sumSize += size
	size /= 1024.0*1024.0
	sizes.append(size)
	names.append(v)


i = np.argsort(sizes)[::-1]
sizes = np.array(sizes)[i]
names = np.array(names)[i]

cumulativeSize = 0

for i in range(len(sizes)):
	sSize= "%.2f Mb" % sizes[i]
	sName = names[i]

	cumulativeSize += sizes[i]

	sTotalSize= "%.2f Mb" % cumulativeSize

	print("%10s %20s %15s \t %s" % (sSize, str(sMap[sName]), sTotalSize, sName))

print("Total size: %.2f Mb" % (sumSize/(1024.0*1024.0)))
		

if opt.stats==1:
	print("-------------------------------------------------")
	print("Statistics:")
	for i in range(len(sizes)):
		t = reader.get_tensor(names[i])
		
		s = " %f    %f    %f" % (t.min(), t.mean(), t.max())
		print("%60s \t %s" % (s, names[i]))
	