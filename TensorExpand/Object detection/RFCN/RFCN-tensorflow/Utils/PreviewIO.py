import os
import cv2
import glob
import sys

class PreviewInput:
	IMG = 0
	DIR = 1
	VID = 2
	NONE = 3
	
	def __init__(self, path):
		self.path = path
		self.fps = 30
		self.currName="unknown"

		if os.path.isdir(self.path):
			self.type=self.DIR
			self.files = glob.glob(self.path+'/*.*')
			self.currFile = 0
		elif self.path.split('.')[-1].lower() in ['avi', 'mp4', 'mpeg', "mov"]:
			self.cap = cv2.VideoCapture(opt.i)
			self.frameIndex = 0
			self.type=self.VID
			if int((cv2.__version__).split('.')[0]) < 3:
				self.fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
			else:
				self.fps = cap.get(cv2.CAP_PROP_FPS)
 
			if self.fps<1:
				self.fps=1
		elif self.path.split('.')[-1].lower() in ['png','bmp','jpg','jpeg']:
			self.type=self.IMG
			self.fps=0
		else:
			print("Invalid file: "+self.path)
			sys.exit(-1)

	def get(self):
		if self.type==self.DIR:
			while True:
				if self.currFile >= len(self.files):
					return None
 
				f = self.files[self.currFile]
				self.currFile+=1

				if f.split('.')[-1].lower() not in ['png','bmp','jpg','jpeg']:
					print("Unknown file: "+f)
					continue

				self.currName = f.split("/")[-1]
				
				img = cv2.imread(f)
				if img is None:
					print("Failed to load image: "+f)
					continue

				return img
		elif self.type==self.IMG:
			self.type=self.NONE
			self.currName = self.path.split("/")[-1]
			return cv2.imread(self.path)
		elif self.type==self.VID:
			ret, frame = self.cap.read()
			if ret==False:
				return None

			self.currName="frame%.6d.jpg" % self.frameIndex
			self.frameIndex += 1
			return frame
		else:
			return None

	def getFps(self):
		return self.fps

	def getDelay(self):
		if self.fps==0:
			return 0
		else:
			return int(1.0/self.fps)

	def getName(self):
		return self.currName

class PreviewOutput:
	IMG = 0
	DIR = 1
	VID = 2
	NONE = 3

	def __init__(self, path, fps=30):
		self.path=path
		self.fps=fps
		if path is None or path=="":
			self.type=self.NONE
		elif os.path.isdir(self.path):
			self.type=self.DIR
		elif self.path.split('.')[-1].lower() in ['avi', 'mp4', 'mpeg', "mov"]:
			self.type=self.VID
			self.writer=None
		elif self.path.split('.')[-1].lower() in ['png','bmp','jpg','jpeg']:
			self.type=self.IMG
		else:
			print("Invalid file: "+self.path)
			sys.exit(-1)

	def put(self, name, frame):
		if self.type==self.DIR:
			cv2.imwrite(self.path+"/"+name, frame)
		elif self.type==self.IMG:
			cv2.imwrite(self.path, frame)
		elif self.type==self.VID:
			if self.writer is None:
				if hasattr(cv2, 'VideoWriter_fourcc'):
					fcc=cv2.VideoWriter_fourcc(*'MJPG')
				else:
					fcc=cv2.cv.CV_FOURCC(*'MJPG')

				self.writer = cv2.VideoWriter(self.path, fcc, int(self.fps), (frame.shape[1], frame.shape[0]))
			self.writer.write(frame)
		else:
			pass
