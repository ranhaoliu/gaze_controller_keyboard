import cv2
import numpy as np
import dlib
# 打开摄像头
cap = cv2.VideoCapture(0)
# 加载人脸检测器 和 调用预测器“shape_predictor_68_face_landmarks.dat”进行68点标定
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()
    # 把bgr 转为灰色图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        # 从人脸中预测出那68 个坐标点
        landmarks = predictor(gray, face)
        # 定位出左眼附近的坐标点
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        left_point2 = (landmarks.part(42).x, landmarks.part(42).y)
        right_point2 = (landmarks.part(45).x, landmarks.part(45).y)
        center_top2 = midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom2 = midpoint(landmarks.part(47), landmarks.part(46))

        hor_line2 = cv2.line(frame, left_point2, right_point2, (0, 255, 0), 2)
        ver_line2 = cv2.line(frame, center_top2, center_bottom2, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()