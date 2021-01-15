import streamlit as st
from fastai.vision import *
from fastai.metrics import *
import urllib.request # get model.pkl
import PIL
import matplotlib.image as mpimg
import gdown # get data.zip & model.pth ## Due to *Large File Size*
import zipfile as zf # extract data.zip
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

def plot_interp(data, model_weight_url, model_arch):
  if st.button("Show Graphs"):
    model = load_model_weight(data, model_weight_url, model_arch)
    interp = ClassificationInterpretation.from_learner(model)
    ## 1. Test accuracy
    preds, y = model.get_preds(ds_type=DatasetType.Valid)
    print('Test accuracy = ', accuracy(preds, y).item())
    ## 2. Plot confusion matrix
    interp.plot_confusion_matrix(figsize=(11,11), dpi=60)
    ## 3. Visualise most wrongly predicted images
    interp.plot_top_losses(9, figsize=(15,11))
  
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
    if model_option == 'Vgg16':
      plot_interp(data, Vgg16_weight_url, models.vgg16_bn)
    elif model_option == 'Vgg19':
      plot_interp(data, Vgg19_weight_url, models.vgg19_bn)
    elif model_option == 'ResNet18':
      plot_interp(data, ResNet18_weight_url, models.resnet18)
    elif model_option == 'ResNet34':
      plot_interp(data, ResNet34_weight_url, models.resnet34)

@st.cache(ttl=3600, max_entries=10)
def get_data(Dataset_Zip_url):
  gdown.download(Dataset_Zip_url, 'data.zip', quiet=False) # Load dataset zip
  zf.ZipFile('data.zip').extractall() # After extract => Data Folder 'AUC_Distracted_Driver_Dataset' obtained
  path = 'AUC_Distracted_Driver_Dataset/Camera1/'
  data = ImageDataBunch.from_folder(path, train='train', valid='test', ds_tfms=get_transforms(do_flip=False), size=(223,433), bs=32).normalize(imagenet_stats)
  return data

@st.cache(ttl=3600, max_entries=10, allow_output_mutation=True)
def load_model_weight(data, model_weight_url, model_arch):
  gdown.download(model_weight_url, 'model.pth', quiet=False)
  model = cnn_learner(data, model_arch, metrics=accuracy).load("model") # path => 'AUC_Distracted_Driver_Dataset/Camera1/models/model.pth'
  return model

# Pages
page = st.sidebar.selectbox("Choose a page", ['Baseline Model Prediction', 'Baseline Model Performance'])
st.set_option('deprecation.showfileUploaderEncoding', False)

# Data -- gdown
DataZip_url = 'https://drive.google.com/uc?id=1Hy9tdBjd7qOucIgIiMFYu9mb0_9ng6xx' #add temp empty file named "models"

# Model Export URL -- urllib.request
Vgg16_export_url = "https://drive.google.com/uc?export=download&id=12zOXR8qUdnjsc4JvwMHfj4Pq7_fS1Hsg"
Vgg19_export_url = "https://drive.google.com/uc?export=download&id=1hVShpb9k2o3hLqY2x4ShQ_jR5Uqb1Bjo"
ResNet18_export_url = "https://drive.google.com/uc?export=download&id=1sNmlieI8bJB6yGyOTgj-r5mGhKBbrltd"
ResNet34_export_url = "https://drive.google.com/uc?export=download&id=1r8ohh3cLc1dKP7bLW_OZMfeBDAj0Ugxh"

# Model Weight URL -- gdown
Vgg16_weight_url = "https://drive.google.com/uc?id=1BDFbhKcteZ95rBzhkpRjq1Cxy3f4PMXf"
Vgg19_weight_url = "https://drive.google.com/uc?id=1-NpZYlfUmnKCJihA2BRt7Ws38me_pkfr"
ResNet18_weight_url = "https://drive.google.com/uc?id=1zNjarWxGld7uF4iBze0ZQQx_5SfRejvG"
ResNet34_weight_url = "https://drive.google.com/uc?id=1ZItSPxQ6k_oaR-t-6H6fE3ubJZFj3Z9E"

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
  ## ** Baseline Model Performance**
   **Test Accuracy,
   Confusion Matrix,
   Most Wronly Predicted Classes**
   '''
          )
  # get Data
  data = get_data(DataZip_url)
  # different model performance
  model_options(show_performance=True)
