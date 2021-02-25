# 凝视检测 实现当眼睛向左看时，和右看时，窗口变色
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


def get_gaze_radio(eye_points,facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y),
                                ])
    print(left_eye_region)
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    """
        Once we have the coordinates of the left eye, we can create the mask to 
        extract exactly the inside of the left eye and exclude all the sorroundings.
    """
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    # cv2.polylines() 画多变行，第一个参数 表示在那个地方画，第二个参数表示包含多边形顶点的数组，
    # 第三个参数表示是否闭合，第四个参数是多边形的颜色，第五个参数是线条的权重
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    # 填充多边形填充的是蓝色的
    cv2.fillPoly(mask, [left_eye_region], 255)
    # bitwise_and  与操作 ,mask 参与运算
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    """
        We now extract the eye from the face and we put it on his own window.
        Only we need to keep in mind that we can only cut out rectangular
        shapes from the image, so we take all the extremes
        points of the eye (top left and right bottom) to get the rectangle.
        We also get the threshold that will need to detect the gaze.
        从x中找个最小值和最大值，从y中找个最小值和最大值，最小值作为阈值二值化的 阈值，
        最大值作为超过阈值后的赋的值
    """
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = left_eye[min_y: max_y, min_x: max_x]
    """
        And finally we display it on the screen.
        I’m going to increase it’s size so we can see it better.
    """
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0: int(width / 2)]
    # 统计出左边半眼白色区域的
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width / 2): width]
    # 统计出右边半眼白色区域的像素点的个数
    right_side_white = cv2.countNonZero(right_side_threshold)

    # cv2.putText(frame ,str(left_side_white),(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    # cv2.putText(frame ,str(right_side_white),(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    gaze_ratio = 0
    if left_side_white == 0:
        gaze_ratio =1
    elif right_side_white == 0:
        gaze_ratio =5
    else:
        gaze_ratio = left_side_white / right_side_white
    return  gaze_ratio
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
        """
            We can select the second eye simply taking the coordinates from the landmarks points.
            We know that the left eye region corresponds to the landmarks
             with indexes: 36, 37, 38, 39, 40 and 41, so we take them.
        """
        left_eye_region = np.array([ (landmarks.part(36).x, landmarks.part(36).y),
                                     (landmarks.part(37).x, landmarks.part(37).y),
                                     (landmarks.part(38).x, landmarks.part(38).y),
                                     (landmarks.part(39).x, landmarks.part(39).y),
                                     (landmarks.part(40).x, landmarks.part(40).y),
                                     (landmarks.part(41).x, landmarks.part(41).y),
                                    ])
        print(left_eye_region)
        cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)
        """
            Once we have the coordinates of the left eye, we can create the mask to 
            extract exactly the inside of the left eye and exclude all the sorroundings.
        """
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        # cv2.polylines() 画多变行，第一个参数 表示在那个地方画，第二个参数表示包含多边形顶点的数组，
        # 第三个参数表示是否闭合，第四个参数是多边形的颜色，第五个参数是线条的权重
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        # 填充多边形填充的是蓝色的
        cv2.fillPoly(mask, [left_eye_region], 255)
        # bitwise_and  与操作 ,mask 参与运算
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        """
            We now extract the eye from the face and we put it on his own window.
            Only we need to keep in mind that we can only cut out rectangular
            shapes from the image, so we take all the extremes
            points of the eye (top left and right bottom) to get the rectangle.
            We also get the threshold that will need to detect the gaze.
            从x中找个最小值和最大值，从y中找个最小值和最大值，最小值作为阈值二值化的 阈值，
            最大值作为超过阈值后的赋的值
        """
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        """
            And finally we display it on the screen.
            I’m going to increase it’s size so we can see it better.
        """
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)

        height , width = threshold_eye.shape
        left_side_threshold = threshold_eye[0:height, 0 : int(width/2) ]
        # 统计出左边半眼白色区域的
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0:height, int (width/2): width ]
        # 统计出右边半眼白色区域的像素点的个数
        right_side_white = cv2.countNonZero(right_side_threshold)

        # cv2.putText(frame ,str(left_side_white),(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        # cv2.putText(frame ,str(right_side_white),(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
      #  # gaze_ratio =   left_side_white / right_side_white
        # Gaze detection

        gaze_ratio_left_eye = get_gaze_radio([36,37,38,39,40,41],landmarks)
        gaze_ratio_right_eye = get_gaze_radio([42,43,44,45,46,47],landmarks)
        gaze_ratio = int((gaze_ratio_left_eye + gaze_ratio_right_eye)/2  *100)
        new_frame = np.zeros((500,500,3),np.uint8)
        if gaze_ratio <90:
            cv2.putText(frame ,str("LEFT"),(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
            new_frame[:] = (0,0,255)
        elif    gaze_ratio <=106:
           cv2.putText(frame ,str("CENTER"),(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        else:
            new_frame[:] =  (255,0,0)
            cv2.putText(frame ,str("RIGHT"),(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

        # 0.8 - 1.1 中间 ，
        # cv2.putText(frame ,str(gaze_ratio),(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

    cv2.imshow("Threshold", threshold_eye)
    cv2.imshow("left", left_side_threshold)
    cv2.imshow("rihgt", right_side_threshold)
    cv2.imshow("new_frame",new_frame)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(50)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()