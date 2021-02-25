#眨眼检测
import cv2
import numpy as np
import dlib
from math import hypot
# 打开摄像头
cap = cv2.VideoCapture(0)
# 加载人脸检测器 和 调用预测器“shape_predictor_68_face_landmarks.dat”进行68点标定
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
# 得到人一只眼睛张开程度的大小
def get_blink_radio (data,landmarks):  # data [36,37,38,39,40,41]
    left_point = (landmarks.part(data[0]).x, landmarks.part(data[0]).y)
    right_point = (landmarks.part(data[3]).x, landmarks.part(data[3]).y)
    center_top = midpoint(landmarks.part(data[1]), landmarks.part(data[2]))
    center_bottom = midpoint(landmarks.part(data[5]), landmarks.part(data[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    # 左眼 水平线的长度
    hor_line_lenght = hypot((left_point[0] - right_point[0]),(left_point[1]-right_point[1]))
    # 左眼 中间竖线的长度
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]),(center_top[1]- center_bottom[1]))

    # print(hor_line_lenght)
    # print(ver_line_lenght)
    # # 右眼， 睁的越大值越小
    # # print(hor_line_lenght/ver_line_lenght)
    ratio = hor_line_lenght / ver_line_lenght
    return ratio
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
        # # 定位出左眼附近的坐标点
        # left_point = (landmarks.part(36).x, landmarks.part(36).y)
        # right_point = (landmarks.part(39).x, landmarks.part(39).y)
        # center_top = midpoint(landmarks.part(37), landmarks.part(38))
        # center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        #
        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        # # 左眼 水平线的长度
        # hor_line_lenght = hypot((left_point[0] - right_point[0]),(left_point[1]-right_point[1]))
        # # 左眼 中间竖线的长度
        # ver_line_lenght = hypot((center_top[0] - center_bottom[0]),(center_top[1]- center_bottom[1]))
        #
        # # print(hor_line_lenght)
        # # print(ver_line_lenght)
        # # # 右眼， 睁的越大值越小
        # # # print(hor_line_lenght/ver_line_lenght)
        # ratio = hor_line_lenght / ver_line_lenght
        # ratio = get_blink_radio([36,37,38,39,40,41],landmarks)
        ratio_left = get_blink_radio([36,37,38,39,40,41],landmarks= landmarks)
        ratio_right = get_blink_radio([42,43,44,45,46,47],landmarks= landmarks)
        if (ratio_left + ratio_right)/2 > 5.5:
            cv2.putText(frame,"Blinking",(50,150),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()