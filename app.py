from fastai.vision import *
#from fastai.vision.widgets import *
#from ipywidgets import * 
from PIL import Image
import streamlit as st
import urllib.request
import time
#import wget

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Driver Distraction Classification")
st.write('''
  ---
  ## **Classify 10 types of distracted driver behaviour**
   **c0: Safe Driving, 
   c1: Texting - Right, 
   c2: Talking on the Phone - Right, 
   c3: Texting - Left, 
   c4: Talking on the Phone - Left, 
   c5: Adjusting Radio, 
   c6: Drinking, 
   c7: Reaching Behind, 
   c8: Hair or Makeup, 
   c9: Talking to Passenger**
'''
)
# st.image('https://dl.dropboxusercontent.com/s/2873nwxz55lrozc/plants.jpg?dl=0', use_column_width=True)
st.write('''
  ### Please upload a picture for one of these distracted driver behaviour
''')
Uploaded = st.file_uploader('', type=['png','jpg','jpeg'])

with st.spinner('Loading...'):
    time.sleep(3)

vgg16_export_url = "https://drive.google.com/uc?export=download&id=1ep3Z_TtkqREcbisijb7Nhm52YnQ1Pp-Y"
urllib.request.urlretrieve(vgg16_export_url, "vgg16_model.pkl")
vgg16_model = load_learner(Path("."), "vgg16_model.pkl")

if Uploaded is not None:
    img = Image.open(Uploaded)
    img_fastai = PILImage.create(Uploaded)
    st.image(img, caption='Uploaded picture', use_column_width=True)
    st.write("")
    pred,pred_idx,probs = vgg16_model.predict(img_fastai)
    st.write("vgg16 model Prediction: ", pred, "; Probability: ", probs[pred_idx]*100,'%')
