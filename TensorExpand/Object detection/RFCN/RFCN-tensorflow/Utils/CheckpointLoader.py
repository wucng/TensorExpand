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

def getCheckpointVarList(file):
	reader = tf.contrib.framework.load_checkpoint(file)
	varsToRead = []
	loadedVars = []
	for v in reader.get_variable_to_shape_map().keys():
		tfVar = tf.contrib.slim.get_variables_by_name(v)
		tfVarFitlered=[]
		for var in tfVar:
			if var.op.name==v:
				tfVarFitlered.append(var)
 
		if len(tfVarFitlered)==0:
			continue
 
		varsToRead += tfVarFitlered
		loadedVars.append(v)
	
	del reader
 
	return varsToRead, loadedVars

def loadVarsFromCheckpoint(sess, vars, file):
	restorer=tf.train.Saver(var_list = vars)
	restorer.restore(sess, file)
	del restorer

def loadExitingFromCheckpoint(file, sess):
	varsToRead, loadedVars = getCheckpointVarList(file)
	loadVarsFromCheckpoint(sess, varsToRead, file)
	return loadedVars

def loadCheckpoint(sess, saveDir, resume, ignoreVarsInFileNotInSess=False):
	def initGlobalVars():
		if "global_variables_initializer" in tf.__dict__:
			sess.run(tf.global_variables_initializer())
		else:
			sess.run(tf.initialize_all_variables())

	firstError = True
	with tf.name_scope('checkpoint_load') as scope:
		if resume is not None:
			last=resume
		else:
			last=tf.train.latest_checkpoint(saveDir)

		if last is not None:
			print("Resuming "+last)

			varsToRead, loadedVars = getCheckpointVarList(last)
			
			allVars=tf.global_variables()
			allVars = [v.op.name for v in allVars]

			allRestored = True
			for v in allVars:
				if v not in loadedVars:
					if firstError:
						print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
						firstError = False
					print("   WARNING: Not loaded: "+v)
					allRestored = False

			if not ignoreVarsInFileNotInSess:
				for v in loadedVars:
					if v not in allVars:
						if firstError:
							print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
							firstError = False
						print("   WARNING: Variable doesn't exists: "+v)

			if not allRestored:
				print("Missing variable found. Initializing variables first.")
				initGlobalVars()

			loadVarsFromCheckpoint(sess, varsToRead, last)
			return True
		else:
			print("Checkpoint not found. Initializing variables.")
			initGlobalVars()
			return False

def importIntoScope(sess, file, fromScope=None, toScope=None, collection=tf.GraphKeys.GLOBAL_VARIABLES, ignore=[]):
	toRun=[]
	reader = tf.contrib.framework.load_checkpoint(file)

	vlist=tf.get_collection(collection, scope=toScope)
	knownNames = reader.get_variable_to_shape_map().keys()
	loaded = []
	varsToLoad = {}

	for va in vlist:
		if fromScope is not None and toScope is not None:
			name = va.op.name.replace(toScope+"/", fromScope+"/", 1)
		else:
			name = va.op.name

		if name not in knownNames:
			print("WARNING: Variable \""+name+"\" not found in file to load.")
			continue

		ignored = False
		for i in ignore:
			if va.op.name.startswith(i) or name.startswith(i):
				print("WARNING: ignoring loading of variable \""+name+"\"")
				ignored = True
				break

		if not ignored:
			loaded.append(name)
			varsToLoad[name] = va

	for name in knownNames:
		if name not in loaded:
			print("WARNING: Unused variable: "+name)

	loadVarsFromCheckpoint(sess, varsToLoad, file)
