import streamlit as st
from fastai.vision import *
import urllib.request
import PIL
import matplotlib.image as mpimg
import time

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

# Model Export URL
Vgg16_export_url = "https://drive.google.com/uc?export=download&id=1ep3Z_TtkqREcbisijb7Nhm52YnQ1Pp-Y"
Vgg19_export_url = "https://drive.google.com/uc?export=download&id=1pUmdPkUCWIukt6ObdAC3iN86EimvxoUs"
ResNet18_export_url = "https://drive.google.com/uc?export=download&id=1q3slph_m8tksbramxyyX8kpli5Gpetth"
ResNet34_export_url = "https://drive.google.com/uc?export=download&id=1WN6CbB2a5e6e1i2TqVsM1X01kLQQ2iXX"

# Try test image / Upload image
option = st.radio('', ['Try a test image', 'Upload image'])

if option == 'Try a test image':
    test_imgs = os.listdir('test-image/')
    test_img = st.selectbox('Please select a test image:', test_imgs)
    file_path = 'test-image/' + test_img
    img = open_image(file_path)
    display_img = mpimg.imread(file_path)
    # different model prediction
    model_options()

else:
    st.write('''### Please upload a picture for one of these distracted driver behaviour''')
    Uploaded = st.file_uploader('', type=['png','jpg','jpeg'])
    if Uploaded is not None:
      img = open_image(Uploaded)
      display_img = st.image(Uploaded, caption='Uploaded picture', use_column_width=True)
      # different model prediction
      model_options()

def predict_img(model_export_url, img, display_img):
  st.image(display_img, use_column_width=True)
  with st.spinner('Loading...'):
        time.sleep(1)
  # model loading & predicting
  urllib.request.urlretrieve(model_export_url, "model.pkl")
  model = load_learner(Path("."), "model.pkl")
  pred, pred_idx, probs = model.predict(img)
  st.write("Model Prediction: ", pred, "; Probability: ", probs[pred_idx]*100,'%')
  
def model_options():
  model_option = st.radio('', ['Vgg16', 'Vgg19', 'ResNet18', 'ResNet34'])
  if model_option == 'Vgg16':
      predict_img(Vgg16_export_url, img, display_img)
  elif model_option == 'Vgg19':
      predict_img(Vgg19_export_url, img, display_img)
  elif model_option == 'ResNet18':
      predict_img(ResNet18_export_url, img, display_img)
  elif model_option == 'ResNet34':
      predict_img(ResNet34_export_url, img, display_img)
