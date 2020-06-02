## 예제 코드 개발환경

- DQ1 B0 Ubuntu
- Python 2.7
- Opencv
  - python 2.7: Opencv 3.2.0
  - c++: Opencv 4.1.2


## 코드 구성

- Lambda 등록 파일 및 폴더
  - greengrasssdk: AWS Greengrass SDK
  - network: mtcnn, tiny-yolo, mobilenet 구동 코드
  - greengrassML.py: AWS 동작을 위한 main code
- S3 업로드 폴더
  - models
  - Lambda size제한으로 S3 에 업로드 및 사용


## 사용 방법

- Lambda upload 파일 압축

```
$ zip -r greengrassML.zip greengrasssdk templates labels lib network greengrassML.py
```

- LNE binary 압축 및 S3에 업로드

```
$ cd models
$ zip -r network.zip *
```

- Local Resource의 Machine Learning local 경로는  /home/ubuntu/models로 설정해야 함

- AWS IoT 설정

  [__Subscriptions__]

   Source    | Target    | Topic              
   --------- | --------- | ------------------ 
   Lambda    | IoT Cloud | lge/answer_topic   
   IoT Cloud | Lambda    | lge/select_network 

  [__MQTT Client message__]

   Mtcnn                     | MobileNet                 | Tiny-Yolo                 
   ------------------------- | ------------------------- | ------------------------- 
   {<br/>    "network": "1"<br/>} | {<br/>    "network": "2"<br/>} | {<br/>    "network": "3"<br/>} 



## Inference 영상 출력 방법

- Flask를 이용하여 inference 영상을 출력

- http://[DQ1 IP]:1234 로 접속하여 영상 출력 확인 가능

  ex) DQ1 IP: 192.168.0.9 -> http://192.168.0.9:1234
