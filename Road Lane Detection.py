import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    img = cv2.addWeighted(img, 0.9, line_img, 1.0, 0.0)

    return img

def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (width*1/10, height),
        (width/2 , height / 2),
        (width*8.5/10, height),
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (float)(y2 - y1) / (float)(x2 - x1)
        if math.fabs(slope) < 0.5:
            pass
        if slope <= 0:
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else:
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])

    min_y = int(image.shape[0]*0.6)
    max_y = int(image.shape[0]*1.2)

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
       deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=20,
    )

    return line_image

img14 = mpimg.imread('road.jpeg')
img15 = pipeline(img14)
cv2.imshow('roads',img15)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('roads1.jpeg',img15)
