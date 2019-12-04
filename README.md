# Simple-Fatigue-Detection-Based-On-dlib
Use dlib and shape_predictor_68_face_landmarks to detection face.The data of eye aspect ratio is detected 30 times first,then take the average value as the threshold value,continuous detection,it is considered that it has been "asleep" and remind while the aspect ratio exceeded the threshold value 50 times.

通过dlib库与shape_predictor_68_face_landmarks模型库进行人脸检测，先检测30次眼睛长宽比的数据，取平均值作为阈值，再连续检测，如果连续50次超过该阈值，则认为已经“睡着”，进行提醒
