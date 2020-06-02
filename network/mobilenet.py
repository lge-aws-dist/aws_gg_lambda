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

# LNE set
from lne_tflite import interpreter as lt

class MobileNet:
	def __init__(self, model_path, label_path):
		self.interpreter = lt.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()
		self.output_detail = self.interpreter.get_output_details()

		with open(label_path) as f:
			l_lines = f.readlines()
			self.labels = [ line for line in l_lines ]

		input_shape = self.input_detail[0]['shape']
		self.width = input_shape[1]
		self.height = input_shape[2]
	
	def resize_img(self, img):
		fitted_img = cv2.resize(img, (self.width, self.height))
		fitted_img = cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB)
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
		lne_answer = np.argmax(output_lne)
		answer = self.labels[lne_answer].strip()
		return answer

	def post_draw(self, img, answer):
		blue = (255, 0, 0)
		thickness = 3
		center_x = img.shape[1]
		center_y = img.shape[0]
		location = ((center_x // 2) - 250, (center_y // 2) + 200)
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale= 2
		img = np.ascontiguousarray(img, dtype=np.uint8)
		cv2.putText(img, answer, location, font, fontScale, blue, thickness)
		return cv2.imencode('.jpg', img)[1].tobytes()
