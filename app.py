import os
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
from PIL import Image
import tensorflow as tf 
import matplotlib.pyplot as plt 


st.set_page_config(page_title="Search for Objects in Video.", page_icon=":desktop_computer:")
st.title('Detect Emmotions in Image')


def save_uploaded_file(uploadedfile):
    if os.path.isfile(uploadedfile.name):
        os.remove(uploadedfile.name)
    else:
        print("Error: %s file not found" % uploadedfile.name)
    with open(os.path.join("", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved file :{}".format(uploadedfile.name))


def objects_detection(image_name):
    new_model=tf.keras.models.load_model('Final_Model.h5')
    frame = cv2.imread(image_name)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Faces Not Detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]
    labeled =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imwrite('/savedLabeled.png',labeled)
    st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), caption='Emmotion Dections')
    plt.imshow(cv2.cvtColor(face_roi,cv2.COLOR_BGR2RGB))
    
    final_image = cv2.resize(face_roi,(224,224))
    final_image= np.expand_dims(final_image,axis=0)
    final_image = final_image/255.0

    Predictions = new_model.predict(final_image)

    Emmotions = ['angry','digusted','fear','happy','neutral','sad','surprised']
    
    return Emmotions[np.argmax(Predictions)]





def main():

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    with upload:
        video_file = st.file_uploader('Upload Image File', type=["png", "jpg", "jpeg"], key=None)
    if video_file is not None:
        saved = save_uploaded_file(video_file)
        image = Image.open(video_file.name)
        try:
            labeled_image = objects_detection(video_file.name)
            st.sidebar.success("Emmotion Detected Is: "+labeled_image.upper())
        except Exception as e:
            st.sidebar.warning("No Emmotions Detected")
        
main()
