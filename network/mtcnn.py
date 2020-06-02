'''
 *   Copyright (c) 2020 LG Electronics Inc.
 *
 *   This program or software including the accompanying associated documentation
 *   ("Software") is the proprietary software of LG Electronics Inc. and or its
 *   licensors, and may only be used, duplicated, modified or distributed pursuant
 *   to the terms and conditions of a separate written license agreement between you
 *   and LG Electronics Inc. ("Authorized License"). Except as set forth in an
 *   Authorized License, LG Electronics Inc. grants no license (express or implied),
 *   rights to use, or waiver of any kind with respect to the Software, and LG
 *   Electronics Inc. expressly reserves all rights in and to the Software and all
 *   intellectual property therein. If you have no Authorized License, then you have
 *   no rights to use the Software in any ways, and should immediately notify LG
 *   Electronics Inc. and discontinue all use of the Software.
'''
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
