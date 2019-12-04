from scipy.spatial import distance as dis
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pyglet
import argparse
import imutils
import time
import dlib
import cv2

#计算嘴的长宽比,euclidean(u, v, w=None)用于计算两点的欧几里得距离
def mouthRatio(mouth):
    left=dis.euclidean(mouth[2],mouth[10])
    mid=dis.euclidean(mouth[3],mouth[9])
    right=dis.euclidean(mouth[4],mouth[8])
    horizontal=dis.euclidean(mouth[0],mouth[6])
    return 10.0*horizontal/(3.0*left+4.0*mid+3.0*right)

#计算眼睛的长宽比
def eyesRatio(eye):
    left = dis.euclidean(eye[1], eye[5])
    right = dis.euclidean(eye[2], eye[4])
    horizontal = dis.euclidean(eye[0], eye[3])
    return 2.0*horizontal/(left+right)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0)
args = vars(ap.parse_args())

#眼睛长宽比的阈值，如果超过这个值就代表眼睛长/宽大于采集到的平均值，默认已经"闭眼"
eyesRatioLimit=0
#数据采集的计数，采集30次然后取平均值
collectCount=0
#用于数据采集的求和
collectSum=0
#是否开始检测
startCheck=False

#统计"闭眼"的次数
eyesCloseCount=0

#初始化dlib
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("68_face_landmarks.dat")

#获取面部各器官的索引
#左右眼
(left_Start,left_End)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_Start,right_End)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#嘴
(leftMouth,rightMouth)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
#下巴
(leftJaw,rightJaw)=face_utils.FACIAL_LANDMARKS_IDXS['jaw']
#鼻子
(leftNose,rightNose)=face_utils.FACIAL_LANDMARKS_IDXS['nose']
#左右眉毛
(left_leftEyebrow,left_rightEyebrow)=face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
(right_leftEyebrow,right_rightEyebrow)=face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']

#开启视频线程
vsThread=VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

#循环检测
while True:
    #对每一帧进行处理，设置宽度并转化为灰度图
    frame = vsThread.read()
    frame = imutils.resize(frame, width=720)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #检测灰度图中的脸
    faces = detector(img, 0)
    for k in faces:
        #确定面部区域的面部特征点，将特征点坐标转换为numpy数组
        shape = predictor(img, k)
        shape = face_utils.shape_to_np(shape)

        #左右眼
        leftEye = shape[left_Start:left_End]
        rightEye = shape[right_Start:right_End]
        leftEyesVal = eyesRatio(leftEye)
        rightEyesVal = eyesRatio(rightEye)
        #凸壳
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #绘制轮廓
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #取两只眼长宽比的的平均值作为每一帧的计算结果
        eyeRatioVal = (leftEyesVal + rightEyesVal) / 2.0

        #嘴
        mouth=shape[leftMouth:rightMouth]
        mouthHull=cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        #鼻子
        nose=shape[leftNose:rightNose]
        noseHull=cv2.convexHull(nose)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

        #下巴
        jaw=shape[leftJaw:rightJaw]
        jawHull=cv2.convexHull(jaw)
        cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)

        #左眉毛
        leftEyebrow=shape[left_leftEyebrow:left_rightEyebrow]
        leftEyebrowHull=cv2.convexHull(leftEyebrow)
        cv2.drawContours(frame, [leftEyebrowHull], -1, (0, 255, 0), 1)

        #右眉毛
        rightEyebrow=shape[right_leftEyebrow:right_rightEyebrow]
        rightEyebrowHull=cv2.convexHull(rightEyebrow)
        cv2.drawContours(frame, [rightEyebrowHull], -1, (0, 255, 0), 1)

        if collectCount<30:
            collectCount+=1
            collectSum+=eyeRatioVal
            cv2.putText(frame, "DATA COLLECTING", (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            startCheck=False
        else:
            if not startCheck:
                eyesRatioLimit=collectSum/(1.0*30)
                print('眼睛长宽比均值',eyesRatioLimit)
            startCheck=True

        if startCheck:
            #如果眼睛长宽比大于之前检测到的阈值，则计数，闭眼次数超过50次则认为已经"睡着"
            if eyeRatioVal > eyesRatioLimit:
                eyesCloseCount += 1
                if eyesCloseCount >= 50:
                    cv2.putText(frame, "SLEEP!!!", (580, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:  
                eyesCloseCount = 0
            print('眼睛实时长宽比:{:.2f} '.format(eyeRatioVal))
            #眼睛长宽比
            cv2.putText(frame, "EYES_RATIO: {:.2f}".format(eyeRatioVal), (20, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
            #闭眼次数
            cv2.putText(frame,"EYES_COLSE: {}".format(eyesCloseCount),(320,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,160,0),2)

            #通过检测嘴的长宽比检测有没有打哈欠，后来觉得没什么卵用
            #cv2.putText(frame,"MOUTH_RATIO: {:.2f}".format(mouthRatio(mouth)),(30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    #停止
    if key == ord("S"):  break

cv2.destroyAllWindows()
vsThread.stop()