# Simple-Fatigue-Detection-Based-On-dlib
通过dlib库与shape_predictor_68_face_landmarks模型库进行人脸检测，先检测30次眼睛长宽比的数据，取平均值作为阈值，再连续检测，如果连续50次超过该阈值，则认为已经“睡着”，进行提醒
