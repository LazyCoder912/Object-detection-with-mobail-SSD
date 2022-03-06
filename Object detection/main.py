import cv2
import numpy as np

thres = 0.45 # Threshold to detect object

cap = cv2.VideoCapture("v.mp4")
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    colors=np.random.uniform(0,255,size=(len("jsgdjb"),3))
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,(255,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper()+" "+str(round(confidence*100,2)),(box[0],box[1]+2),
                        cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),2)
    cv2.imshow("Output", img)
    if cv2.waitKey(1)==27:
        break;
cap.release()
cv2.destroyAllWindows()
