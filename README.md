## Environment

  - LG8111 AI board(Ubuntu 18.04)
  - Python 2.7
  - Opencv
  - python 2.7: Opencv 3.2.0
  - c++: Opencv 4.1.2


## Directory Description
  - greengrasssdk: AWS Greengrass SDK
  - network: functions required to inferece using mtcnn, tiny-yolo, mobilenet 
  - greengrassML.py: Lambda function deployed by AWS Greengrass Group. After deploy on core device, it is called by runtime in AWS Greengrass core S/W

   Note) Network model itself required to perform local inference are located in LG8111 board.
      Please contace us via e-mail to get more information about LG8111 borad.(lge-aws-dist@lge.com)


## How to set AWS greengrass Group

  - zip Lambda function and upload it to AWS-Lambda Service

```
$ zip -r greengrassML.zip greengrasssdk templates labels lib network greengrassML.py
```

  - Local Resource

```
set Local Resource as /home/ubuntu/models
```


## How To test on AWS IoT

  [__Subscriptions__]

   Source    | Target    | Topic              
   --------- | --------- | ------------------ 
   Lambda    | IoT Cloud | lge/answer_topic   
   IoT Cloud | Lambda    | lge/select_network 

  [__MQTT Client message__]

   Mtcnn                     | MobileNet                 | Tiny-Yolo                 
   ------------------------- | ------------------------- | ------------------------- 
   {<br/>    "network": "1"<br/>} | {<br/>    "network": "2"<br/>} | {<br/>    "network": "3"<br/>} 



## How to see the Infrerence result 

- This application Output inference image using Flask

- Open browser and access with http://[LG811 AI board IP]:1234 

  ex) LG 8111 AI Board IP: 192.168.0.9 -> http://192.168.0.9:1234
