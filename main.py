
# import the necessary packages specific to Computer vision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# function for detecting face and eyes

def detect_and_show_face_in_image(image_for_faceDetection, sz):

    # assigning the har cascades of face and eyes to variables
    face_cascade_classifier = cv2.CascadeClassifier('model//haarcascades//haarcascade_frontalface_default.xml')
    if face_cascade_classifier.empty():
        print('Missing face classifier xml file')

    eye_cascade_classifier = cv2.CascadeClassifier('model//haarcascades//haarcascade_eye_tree_eyeglasses.xml')
    if eye_cascade_classifier.empty():
        print('Missing eye classifier xml file')

    img_gray = cv2.cvtColor(image_for_faceDetection, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_classifier.detectMultiScale(img_gray, scaleFactor=1.3,
                                                     # jump in the scaling factor, as in, if we don't find an image in the
                                                     # current scale, the next size to check will be, in our case, 1.3 times bigger than the current size.
                                                     minNeighbors=5, minSize=(30, 30))
    eye_count = 0
    for (x, y, w, h) in faces:
        if (w > 0 and h > 0):
            image_for_faceDetection = cv2.rectangle(image_for_faceDetection, (x, y), (x + w, y + h), (0, 255, 255), 5)
            face = img_gray[y:y + h, x:x + w]
            eyes = eye_cascade_classifier.detectMultiScale(face)
            roi_color = image_for_faceDetection[y:y + h, x:x + w]

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 5)
                eye_count += 1
    image_for_faceDectection = cv2.putText(image_for_faceDetection,
                                           ("faces =" + str(len(faces)) + "   eyes =" + str(eye_count)),
                                           (sz[0]//2, sz[1]//2),
                                           cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return image_for_faceDetection



#Face detection in Resources
def Face_detection_in_images(path):
    img=cv2.imread(path)

    sz = (img.shape[1], img.shape[0])
    img=detect_and_show_face_in_image(img,sz)
    #print(eyes)
    img = cv2.resize(img, (600, 600))
    cv2.imshow("faces",img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#Face detection in web cam live video
def Face_detection_in_webCam():
    video=cv2.VideoCapture(0)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    fourcc=cv2.VideoWriter_fourcc(*'FMP4')
    demo = cv2.VideoWriter('myvideo.avi',fourcc,20,size)
    while (video.isOpened()):
        ret, img = video.read()
        if (ret):
            img = cv2.resize(img, (600, 600))
            img = detect_and_show_face_in_image(img, size)
            img = cv2.resize(img, (600, 600))
            demo.write(img)
            cv2.imshow("video", img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    video.release()
    demo.release()
    cv2.destroyAllWindows()


def Face_detection_in_video(path):
    video = cv2.VideoCapture(path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    demo = cv2.VideoWriter('myvideo2.avi', fourcc, 20, size)
    while (video.isOpened()):
        ret, img = video.read()
        if (ret):
            img = detect_and_show_face_in_image(img, size)
            img = cv2.resize(img, (600, 600))
            demo.write(img)
            cv2.imshow("video", img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    video.release()
    demo.release()
    cv2.destroyAllWindows()

#face detection in images
Face_detection_in_images("Resources//group_of_people.jpg")
Face_detection_in_video("Resources//samplevideo.mp4")
Face_detection_in_video("Resources//samplevideo2.mp4")
Face_detection_in_webCam()