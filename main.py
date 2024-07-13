import cv2

thres = 0.47  # mizan dorosti marzi

cap = cv2.VideoCapture(0)  # video ra az durbin migirad ham chenin agar nam file bashad az file migirad
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# vared kardan nam class ha

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# vared kardan shabkeh
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# tanzimat shabke
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # gereftan har frame
    success, frame = cap.read()

    # tashkhis ashay mojod dar har frame
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    # nomayesh code shay va mokhtasad box neshan dahandeh
    print(classIds, bbox)

    if len(classIds) != 0:

        # keshidan box dor shey dar har frame va neveshtan deghat va darsad dorosti
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

            # araye taqir rang
            if classId % 4 == 0:
                cv2.rectangle(frame, box, color=(255, 0, 0), thickness=2)
                cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            elif classId % 4 == 1:
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            elif classId % 4 == 2:
                cv2.rectangle(frame, box, color=(0, 0, 255), thickness=2)
                cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2)
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2)

            elif classId % 4 == 3:
                cv2.rectangle(frame, box, color=(0, 255, 255), thickness=2)
                cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 225), 2)
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 225), 2)

    # namayesh tasavir posh sar ham ke besurat video zaher misheard
    cv2.imshow("Output", frame)
    cv2.waitKey(1)
