# download py tesseract software using follouing link
# do not change file path of tesseract while installation
# download all if needed languages and scripts (download will take over 2 hours)

# https://github.com/UB-Mannheim/tesseract/wiki

# run the following in terminal before runnging python program
# pip install opencv-python
# pip install pytesseract

# after running program a camera will appear, click 's' on keyboard to scan for text on camera
# currently the program does not input faraway text or black background text, it needs very large font size or a very good camera
# das it siuuuuuuuuuuu

import cv2
from PIL import Image
from pytesseract import pytesseract

camera=cv2.VideoCapture(0)


while True:
    _,image=camera.read()
    cv2.imshow('Text detection', image)
    if cv2.waitKey(1)& 0xFF==ord('s'):
        cv2.imwrite('test.jpg',image)
        break
camera.release()
cv2.destroyAllWindows()

def tesseract():
    # \t means tab in python like \n so we use \\t to register it a string "\t" and not tab
    #tesseract.exe file location
    link = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
    ImageName = 'test.jpg'
    pytesseract.tesseract_cmd = link
    text = pytesseract.image_to_string(Image.open(ImageName))
    print(text)

tesseract()

