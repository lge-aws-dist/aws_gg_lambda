import sys
import time
import logging
import greengrasssdk

# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

client = greengrasssdk.client('iot-data')


def function_handler(event, context):
	try:
		topic = context.client_context.custom['subject']
		
		logging.debug("greengrassControl receive topic : " + str(topic));
		logging.debug("greengrassML received message : " + str(event));
		
		# check if it received from AWS IoT or from greengrassML
		if 'greengrassControl' in topic:
			#forward payload to greengrassML received from IOT core
			logging.debug("Receive topic from AWS IoT");
			client.publish(topic='lge/lg8111/greengrassML/inference/request', payload=event)

		if 'greengrassML' in topic:
			#forward payload to IoT Core  received from greengrassML
			logging.debug("Receive topic from greengrassML");
			client.publish(topic='lge/lg8111/greengrassControl/control/event', payload=event)

		
	except Exception as e:
		logging.error(e)

	return
