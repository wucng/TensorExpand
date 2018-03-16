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

import os
import json
import argparse

class StorableArgparse:
	def __init__(self, description):
		self.parser = argparse.ArgumentParser(description=description)
		self.loaded={}
		self.args={}
		self.parsed=None

	def add_argument(self, name, type, default=None, help="", save=True):
		self.parser.add_argument(name, type=type, default=None, help=help)
		if name[0]=='-':
			name = name[1:]

		self.args[name]={
			"type": type,
			"default": default,
			"save": save
		}

	def do_parse_args(self, loaded={}):
		self.parsed=self.parser.parse_args()
		for k, v in self.parsed.__dict__.items():
			if v is None:
				if k in loaded and self.args[k]["save"]:
					self.parsed.__dict__[k] = loaded[k]
				else:
					self.parsed.__dict__[k] = self.args[k]["default"]
		return self.parsed

	def parse_or_cache(self):
		if self.parsed is None:
			self.do_parse_args()

	def parse_args(self):
		self.parse_or_cache()
		return self.parsed

	def save(self, fname):
		self.parse_or_cache()
		with open(fname, 'w') as outfile:
			json.dump(self.parsed.__dict__, outfile, indent=4)
			return True

	def load(self, fname):
		if os.path.isfile(fname):
			map={}
			with open(fname,"r") as data_file:
				map=json.load(data_file)

			self.do_parse_args(map)
		return self.parsed