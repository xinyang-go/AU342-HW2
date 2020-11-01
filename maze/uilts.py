import cv2
import numpy as np


def create_table():
    img = np.zeros([600, 600], dtype=np.uint8)
    img.fill(255)
    for i in range(7):
        cv2.line(img, (0, i * 100), (600, i * 100), 0, 2)
        cv2.line(img, (i * 100, 0), (i * 100, 600), 0, 2)
    return img


def draw_value(img, x, y, value):
    cv2.putText(img, f"{value[0]:4.1f}", (x * 100 + 25, y * 100 + 25),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
    cv2.putText(img, f"{value[1]:4.1f}", (x * 100 + 25, y * 100 + 75),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
    cv2.putText(img, f"{value[3]:4.1f}", (x * 100 + 0, y * 100 + 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
    cv2.putText(img, f"{value[2]:4.1f}", (x * 100 + 50, y * 100 + 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)


def view(q_table, delay=1):
    img0 = create_table()
    img1 = create_table()
    for x in range(6):
        for y in range(6):
            draw_value(img0, x, y, q_table[x, y, 0])
            draw_value(img1, x, y, q_table[x, y, 1])
    cv2.imshow("q_table0", img0)
    cv2.imshow("q_table1", img1)
    cv2.waitKey(delay)


def state2xyt(state):
    x = int((state[0] - 5) // 40)
    y = int((state[1] - 5) // 40)
    t = int(state[4])
    return x, y, t
