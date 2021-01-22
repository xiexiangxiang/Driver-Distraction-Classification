import streamlit as st
from fastai.vision import *
from fastai.metrics import *
import urllib.request # get model.pkl
import PIL
import matplotlib.image as mpimg
import time

## Funtions
def predict_img(model_export_url, img, display_img):
  # Display Image
  st.image(display_img, use_column_width=True)
  with st.spinner('Loading...'):
        time.sleep(2)
  # Model Loading & Prediction
  if st.button("Analyse"):
    urllib.request.urlretrieve(model_export_url, "model.pkl")
    model = load_learner(Path("."), "model.pkl")
    pred, pred_idx, probs = model.predict(img)
    st.write("Model Prediction: ", pred, "; Probability: ", probs[pred_idx]*100,'%')
  
def model_options(predict=False, show_performance=False):
  model_option = st.radio('', ['Vgg16', 'Vgg19', 'ResNet18', 'ResNet34'])
  if predict == True:
    if model_option == 'Vgg16':
      predict_img(Vgg16_export_url, img, display_img)
    elif model_option == 'Vgg19':
      predict_img(Vgg19_export_url, img, display_img)
    elif model_option == 'ResNet18':
      predict_img(ResNet18_export_url, img, display_img)
    elif model_option == 'ResNet34':
      predict_img(ResNet34_export_url, img, display_img)
  elif show_performance == True:
    link = '[Google Colab](https://colab.research.google.com/drive/1YWqFjd_2PXyu70D-SHcwaKbZaOs5BNU-?usp=sharing)'
    st.markdown(link, unsafe_allow_html=True)

# Pages
page = st.sidebar.selectbox("Choose a page", ['Baseline Model Prediction', 'Baseline Model Performance'])
#st.set_option('deprecation.showfileUploaderEncoding', False)

# Model Export URL -- urllib.request
Vgg16_export_url = "https://drive.google.com/uc?export=download&id=1qPWKfHgkUQVsccgrkB769iNuvy32Nqrn"
Vgg19_export_url = "https://drive.google.com/uc?export=download&id=1AwSnWOyEr634uynzES0A4UvJfScOjgnk"
ResNet18_export_url = "https://drive.google.com/uc?export=download&id=10WIqVP9sADfTstn1shDOO6PNplwhVvOh"
ResNet34_export_url = "https://drive.google.com/uc?export=download&id=1366mTUbh1WGVLqPwPp4mCeBpeAJiQrik"

## Page - Baseline Model Prediction
if page == 'Baseline Model Prediction':
  st.title("Driver Distraction Classification")
  st.write('''
  ---
  ## ** Baseline Model - Classify 10 types of distracted driver behaviour**
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
  # Try test image / Upload image
  option = st.radio('', ['Try a test image', 'Upload image'])
  if option == 'Try a test image':
    test_imgs = os.listdir('test-image/')
    test_img = st.selectbox('Please select a test image:', test_imgs)
    file_path = 'test-image/' + test_img
    img = open_image(file_path)
    display_img = mpimg.imread(file_path)
    # different model prediction
    model_options(predict=True)
  else:
    st.write('''### Please upload a picture for one of these distracted driver behaviour''')
    Uploaded = st.file_uploader('', type=['png','jpg','jpeg'])
    if Uploaded is not None:
      img = open_image(Uploaded)
      display_img = Uploaded
      # different model prediction
      model_options(predict=True)
  
## Page - Baseline Model Performance
elif page == 'Baseline Model Performance':
  st.title("Driver Distraction Classification")
  st.write('''
  ---
  ## ** Baseline Model Performance [Google Colab] **
   **Test Accuracy,
   Confusion Matrix,
   Most Wronly Predicted Classes**
   ''')
  model_options(show_performance=True)
