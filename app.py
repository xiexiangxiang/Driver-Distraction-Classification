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
  col2.image(display_img, use_column_width=True)
  # Model Loading & Prediction
  if col2.button("Analyse"):
    urllib.request.urlretrieve(model_export_url, "model.pkl")
    model = load_learner(Path("."), "model.pkl")
    pred, pred_idx, probs = model.predict(img)
    st.write("Model Prediction: ", pred, "; Probability: ", probs[pred_idx]*100,'%')
    st.write(probs,'%')
  
def model_options(predict=False):
  model_option = col1.radio('Choose a model:', ['Vgg16', 'Vgg16_b', 'Vgg19','Vgg19_b', 'ResNet18', 'ResNet18_b', 'ResNet34', 'ResNet34_b'])
  if predict == True:
    if model_option == 'Vgg16':
      predict_img(Vgg16_export_url, img, display_img)
    elif model_option == 'Vgg16_b':
      predict_img(Vgg16_b_export_url, img, display_img)
    elif model_option == 'Vgg19':
      predict_img(Vgg19_export_url, img, display_img)
    elif model_option == 'Vgg19_b':
      predict_img(Vgg19_b_export_url, img, display_img)
    elif model_option == 'ResNet18':
      predict_img(ResNet18_export_url, img, display_img)
    elif model_option == 'ResNet18_b':
      predict_img(ResNet18_b_export_url, img, display_img)
    elif model_option == 'ResNet34':
      predict_img(ResNet34_export_url, img, display_img)
    elif model_option == 'ResNet34_b':
      predict_img(ResNet34_b_export_url, img, display_img)

# Pages
page = st.sidebar.selectbox("Choose a page", ['Baseline Model Prediction', 'Ensemble Model Prediction'])
#st.set_option('deprecation.showfileUploaderEncoding', False)

# Model Export URL -- urllib.request
Vgg16_export_url = "https://drive.google.com/uc?export=download&id=1jiRNwi2e_kfh4i8PXPun0Zbo_XLgz1lp"
Vgg19_export_url = "https://drive.google.com/uc?export=download&id=1qFyLokQD_AmhQEiJwAjo4gF0wfBZoh2a"
ResNet18_export_url = "https://drive.google.com/uc?export=download&id=1mQzySjxlfZdzW88UrM74pQ_duLY67k2a"
ResNet34_export_url = "https://drive.google.com/uc?export=download&id=16vDbA-yqcRBGwk_QEcLJcixn8TBYmGLT"

Vgg16_b_export_url = "https://drive.google.com/uc?export=download&id=1hmuOsHrkcKbe3doKGage7UBKgy9Xd9_Z"
Vgg19_b_export_url = "https://drive.google.com/uc?export=download&id=1O83NNtBBZ4mH4p4kw5dPXDNPwuFoxBgn"
ResNet18_b_export_url = "https://drive.google.com/uc?export=download&id=1QrZ8XMtqoKqAAMcAEfP14YsM1E4s5A5f"
ResNet34_b_export_url = "https://drive.google.com/uc?export=download&id=1UoXGwiWn1YV9hnJhLTpSe1ug_yka6u6t"

## Page - Baseline Model Prediction

if page == 'Baseline Model Prediction':
  st.title("Baseline Driver Distraction Classification")
  st.write('''
  ## ** Classify 10 types of distracted driver behaviour **
   c0: Safe Driving, 
   c1: Texting - Right, 
   c2: Talking on the Phone - Right, 
   c3: Texting - Left, 
   c4: Talking on the Phone - Left, 
   c5: Adjusting Radio, 
   c6: Drinking, 
   c7: Reaching Behind, 
   c8: Hair or Makeup, 
   c9: Talking to Passenger
  ''')
  link = '[Google Colab](https://colab.research.google.com/drive/1YWqFjd_2PXyu70D-SHcwaKbZaOs5BNU-?usp=sharing)'
  st.markdown(link, unsafe_allow_html=True)
  st.write('''---''')
  
  # Try test image / Upload image
  option = st.radio('Choose a distrated drving image', ['Try a test image', 'Upload an image'])
  
  if option == 'Try a test image':
    # create 2 columns structure
    col1,col2 = st.beta_columns([1,2]) # 2nd column is 2 times of 1st column
    test_imgs = os.listdir('test-image/')
    test_img = col1.selectbox('Select a test image:', test_imgs)
    file_path = 'test-image/' + test_img
    img = open_image(file_path)
    display_img = mpimg.imread(file_path)
    model_options(predict=True)
  else:
    Uploaded = st.file_uploader('', type=['png','jpg','jpeg'])
    if Uploaded is not None:
      # create 2 columns structure
      col1,col2 = st.beta_columns([1,2])
      img = open_image(Uploaded)
      display_img = Uploaded
      model_options(predict=True)

## Page - Ensemble Model Prediction
elif page == 'Ensemble Model Prediction':
  st.write("...")
