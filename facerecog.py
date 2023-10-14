import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt

Tk().withdraw()
load_image = askopenfilename()

target_image = fr.load_image_file(load_image)
target_encodings = fr.face_encodings(target_image)

if len(target_encodings) == 0:
    print("No face found in the target image.")
    exit()

target_encoding = target_encodings[0]

def encode_faces(folder):
    list_people_encoding = []
    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))
        known_encodings = fr.face_encodings(known_image)

        if len(known_encodings) > 0:
            known_encoding = known_encodings[0]
            list_people_encoding.append((known_encoding, filename))

    return list_people_encoding

def find_target_face():
    for person in encode_faces('faces/'):
        encoded_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces([encoded_face], target_encoding, tolerance=0.5)
        print(f'{is_target_face} {filename}')

        if any(is_target_face):
            face_locations = fr.face_locations(target_image)
            for location in face_locations:
                top, right, bottom, left = location
                label = filename
                create_frame(target_image, location, label)

def create_frame(image, location, label):
    top, right, bottom, left = location
    cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

def render_image(image):
    if len(image.shape) == 3:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        image_rgb = image

    plt.imshow(image_rgb)
    plt.title('FACE RECOGNITION')
    plt.show()

find_target_face()
render_image(target_image)
