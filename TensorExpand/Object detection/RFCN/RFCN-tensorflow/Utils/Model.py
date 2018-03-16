import os

URL="http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"
FILENAME="./inception_resnet_v2_2016_08_30.ckpt"

def download():
	if not os.path.isfile(FILENAME):
		print("Checkpoint file doesn't exists. Downloading it from tensorflow slim model list.")
		import requests, tarfile, io
		request = requests.get(URL)
		decompressedFile = tarfile.open(fileobj=io.BytesIO(request.content), mode='r|gz')
		decompressedFile.extractall()
		print("Done.")
