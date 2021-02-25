# 画键盘
import cv2
import numpy as np
keyboard = np.zeros((1000, 1500, 3), np.uint8)
# Keys
width = 200
height = 200
th = 3 # thickness
def letter(x,y,text):
    cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)

letter(0, 0, "A")
letter(200, 0, "B")
letter(400, 0, "C")
cv2.imshow("test",keyboard)
cv2.waitKey(0)