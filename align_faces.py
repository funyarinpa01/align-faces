# import the necessary packages
from my_imutils.face_utils import FaceAligner
from my_imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
                help="path to facial landmark predictor (default: dlib-models/shape_predictor_68_face_landmarks.dat)",
                default='dlib-models/shape_predictor_68_face_landmarks.dat')
ap.add_argument("-s", "--src", required=True,
                help="path to source directory")
ap.add_argument("-d", "--dest", required=True,
                help="path to destination directory")
args = vars(ap.parse_args())

# Create the directory if it doesn't exist
if not os.path.exists(args['dest']):
    os.makedirs(args['dest'])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)


def align_face(image_name: str) -> None:
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(args['src'] + image_name)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 2)

    # loop over the face detections
    for i, rect in enumerate(rects):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

        # save the output images
        cv2.imwrite(args['dest'] + str(i) + image_name, faceAligned)


for filename in os.listdir(args['src']):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        align_face(filename)
