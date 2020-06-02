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
import sys
import time
import numpy as np
import logging
import threading
import greengrasssdk

from flask import Flask, render_template, Response

from network.mtcnn import Mtcnn
from network.mobilenet import MobileNet
from network.tinyyolo_lite import TinyYoloNet

app = Flask(__name__)

#ejlim
# Setup logging to stdout
#logger = logging.getLogger(__name__)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Setup logging to stdout
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

client = greengrasssdk.client('iot-data')

select_network = '0'
answer = 'No data'

def get_input_topic(context):
	try:
		topic = context.client_context.custom['subject']
	except Exception as e:
		logging.error('Topic could not be parsed. ' + repr(e))
	return topic

def get_input_message(event):
	try:
		message = event['network']
	except Exception as e:
		logging.error('Message could not be parsed. ' + repr(e))
	return message

#DQ1 camera data format is planar, so convert planar to packed
def camera_convert(frame):
    return np.transpose(frame.reshape(3, 480, 640), (1, 2, 0)) 

def run_lne():
	try:
		global select_network
		global answer
		lne_mtcnn = Mtcnn('./lib/libdq1face.so')
		lne_mobile = MobileNet('/home/ubuntu/models/mobilenet.lne', './labels/labels.txt')
		lne_tinyyolo = TinyYoloNet('/home/ubuntu/models/tiny-yolo.lne')
		
		cap = cv2.VideoCapture(0)
		while(cap.isOpened()):
			ret, frame = cap.read()
			frame = camera_convert(frame)
			if select_network == '1':
				input_img = lne_mtcnn.resize_img(frame)
				answer_mtcnn = lne_mtcnn.inference(input_img)
				answer = str(answer_mtcnn)
				network_img = lne_mtcnn.post_draw(input_img, answer_mtcnn)
	
			elif select_network == '2':
				input_img = lne_mobile.crop_img(frame)
				input_img = lne_mobile.resize_img(input_img)
				answer = lne_mobile.inference(input_img)
				network_img = lne_mobile.post_draw(frame, answer)
			
			elif select_network == '3':
				input_img = lne_tinyyolo.crop_img(frame)
				input_img = lne_tinyyolo.resize_img(input_img)
				lne_result = lne_tinyyolo.inference(input_img)
				result = lne_tinyyolo.post_process(lne_result[0, 0, 0, :])
				answer = ', '.join([result[i][0] for i in range(len(result))])
				if not answer.strip():
					answer = 'No data'
				network_img = lne_tinyyolo.post_draw(frame, result)

			else:
				network_img = cv2.imencode('.jpg', frame)[1].tobytes()
				answer = 'No data'

			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + network_img + b'\r\n')

	except Exception as e:
		logging.error(e)

def greengrass_ML():
	global answer
	while True:
		client.publish(topic='lge/answer_topic', payload=answer)
		time.sleep(1)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(run_lne(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
	app.run(host='0.0.0.0', port='1234', debug=False)

def greengrass_run():
	net_thread = threading.Thread(target=greengrass_ML)
	stream_thread = threading.Thread(target=run_stream)
	net_thread.start()
	stream_thread.start()

greengrass_run()

def function_handler(event, context):
	try:
		global select_network
		input_topic = get_input_topic(context)
		input_message = get_input_message(event)
		if 'network' in input_topic:
			select_network = input_message
		client.publish(topic='lge/answer_topic', payload='{}, {}'.format(input_topic, select_network))
	except Exception as e:
		logging.error(e)

	return
