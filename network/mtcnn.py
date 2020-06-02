import cv2
import numpy as np

from ctypes import *

class Mtcnn:
	def __init__(self, binary_path):
		self.mtcnn_lib = cdll.LoadLibrary(binary_path)
		self.mtcnn_lib.mtcnn_init()
		self.img_num = 0
		return

	def resize_img(self, img):
		fitted_img = cv2.resize(img, (640, 480))
		return fitted_img

	def inference(self, img):
		self.mtcnn_num = (c_int*1)()
		self.pos = (c_int*1000)()
		self.lne_input = np.expand_dims(img, axis = 0)
		self.mtcnn_lib.mtcnn_run(self.lne_input.ctypes.data_as(POINTER(c_ubyte)), self.mtcnn_num, self.pos)
		return self.mtcnn_num[0]

	def post_draw(self, img, result):
		for i in range(result):
			cv2.rectangle(img, (self.pos[i*4+0], self.pos[i*4+1]), \
					           (self.pos[i*4+2], self.pos[i*4+3]), (255, 0, 0), 1)
		return cv2.imencode('.jpg', img)[1].tobytes()
