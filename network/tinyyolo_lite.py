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
import os, sys
import numpy as np
from lne_tflite import interpreter as lt

from network.interpret_yolo import interpret_output, iou, show_results

class TinyYoloNet:
	def __init__(self, model_path):
		self.interpreter = lt.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()
		self.output_detail = self.interpreter.get_output_details()

		input_shape = self.input_detail[0]['shape']
		self.width = input_shape[1]
		self.height = input_shape[2]

	def resize_img(self, img):
		fitted_img = cv2.resize(img, (self.width, self.height))
		fitted_img = cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
		fitted_img = np.expand_dims(fitted_img, axis = 0)
		return fitted_img

	def crop_img(self, img):
		(y,x,channel) = img.shape
		x_prime = y
		img = img[0:y, int((x-x_prime)/2):int((x+x_prime)/2)]
		return img

	def inference(self, img):
		self.interpreter.set_tensor(self.input_detail[0]['index'], img)
		self.interpreter.invoke()
		output_lne = self.interpreter.get_tensor(self.output_detail[0]['index'])
		return output_lne

	def post_process(self, output_lne):
		return interpret_output(output_lne, self.width, self.height)

	def post_draw(self, img, result):
		 tinyyolo_img = show_results(img, result, self.width, self.height)
		 return cv2.imencode('.jpg', tinyyolo_img)[1].tobytes()
