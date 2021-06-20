'''
Ayush Mittal
Task - Object Detection / Optical Character Recognition (ORC)
GRIP @ The Sparks Foundation

'''

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# # -------------------- detecting character ---------------------- #

# reading image
imgnum = cv2.imread('images2.png')

# storing height and width of the image
h_img, w_img, _ = imgnum.shape

# storing string containing recognized characters and their box boundaries
boxes = pytesseract.image_to_boxes(imgnum)

for b in boxes.splitlines():
    b = list(b.split())

    # sotring the value of x and y coordinates and height and width of the particular character box
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

    # creating rectangle around the character and putting the same text
    cv2.rectangle(imgnum, (x, h_img-y), (w, h_img-h), (0, 0, 255), 3)
    cv2.putText(imgnum, b[0], (x, h_img - y+25),
                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

# showing the output image
cv2.imshow('Detected Characters', imgnum)
cv2.waitKey(0)


# -------------------- detecting words --------------------------- #
img = cv2.imread('imagehand.png')

# storing string containing box boundaries, confidences,and other information.
boxes = pytesseract.image_to_data(img)

for n, b in enumerate(boxes.splitlines()):
    if n != 0:
        b = list(b.split())
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (w+x, h+y), (0, 0, 255), 3)
            cv2.putText(img, b[11], (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

cv2.imshow('Detected Words', img)
cv2.waitKey(0)


# -------------------- detecting digits only --------------------------- #
imgnum = cv2.imread('images.jpg')
h_img, w_img, _ = imgnum.shape

# adding configuration to the pytesseract so based on that it will filter out the data for us
boxes = pytesseract.image_to_boxes(
    imgnum, config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/')
for b in boxes.splitlines():

    b = list(b.split())
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(imgnum, (x, h_img-y), (w, h_img-h), (0, 0, 255), 3)
    cv2.putText(imgnum, b[0], (x, h_img - y+25),
                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

cv2.imshow('Detected Digits Only', imgnum)
cv2.waitKey(0)
