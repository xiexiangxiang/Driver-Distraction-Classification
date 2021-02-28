import streamlit as st
from fastai.vision import *
from fastai.metrics import *
import urllib.request # get model.pkl
import PIL
import matplotlib.image as mpimg
import time
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

## Funtions
def predict_img(col2, model_export_url, img):
  # Model Loading & Prediction
  if col2.button("Analyse"):
    with st.spinner('Loading...'):
      time.sleep(3)
    urllib.request.urlretrieve(model_export_url, "model.pkl")
    model = load_learner(Path("."), "model.pkl")
    pred, pred_idx, probs = model.predict(img)
    st.write("Model Prediction: ", pred, "; Probability: ", probs[pred_idx].item()*100,'%')
    st.write(probs,'%')
 
def base_model_options(col1, col2, img, predict=False):
  model_option = col1.radio('Choose a model:', ['Vgg16', 'Vgg16_b', 'Vgg19','Vgg19_b', 'ResNet18', 'ResNet18_b', 'ResNet34', 'ResNet34_b'])
  if predict == True:
    if model_option == 'Vgg16':
      predict_img(col2, Vgg16_export_url, img)
    elif model_option == 'Vgg16_b':
      predict_img(col2, Vgg16_b_export_url, img)
    elif model_option == 'Vgg19':
      predict_img(col2, Vgg19_export_url, img)
    elif model_option == 'Vgg19_b':
      predict_img(col2, Vgg19_b_export_url, img)
    elif model_option == 'ResNet18':
      predict_img(col2, ResNet18_export_url, img)
    elif model_option == 'ResNet18_b':
      predict_img(col2, ResNet18_b_export_url, img)
    elif model_option == 'ResNet34':
      predict_img(col2, ResNet34_export_url, img)
    elif model_option == 'ResNet34_b':
      predict_img(col2, ResNet34_b_export_url, img)

@st.cache(allow_output_mutation=True)
def load_models():
  urllib.request.urlretrieve(Vgg16_export_url, "Vgg16.pkl")
  urllib.request.urlretrieve(Vgg16_b_export_url, "Vgg16_b.pkl")
  urllib.request.urlretrieve(Vgg19_export_url, "Vgg19.pkl")
  urllib.request.urlretrieve(Vgg19_b_export_url, "Vgg19_b.pkl")
  urllib.request.urlretrieve(ResNet18_export_url, "ResNet18.pkl")
  urllib.request.urlretrieve(ResNet18_b_export_url, "ResNet18_b.pkl")
  urllib.request.urlretrieve(ResNet34_export_url, "ResNet34.pkl")
  urllib.request.urlretrieve(ResNet34_b_export_url, "ResNet34_b.pkl")
  Vgg16 = load_learner(Path("."), "Vgg16.pkl")
  Vgg16_b = load_learner(Path("."), "Vgg16_b.pkl")
  Vgg19 = load_learner(Path("."), "Vgg19.pkl")
  Vgg19_b = load_learner(Path("."), "Vgg19_b.pkl")
  ResNet18_b = load_learner(Path("."), "ResNet18_b.pkl")
  ResNet34_b = load_learner(Path("."), "ResNet34_b.pkl")
  return Vgg16, Vgg16_b, Vgg19, Vgg19_b, ResNet18_b, ResNet34_b

def ensemble_model_options(col1, col2, img):
  model_option = col1.radio('Choose a model:', ['Vgg16 + Vgg19 + ResNet18_b', 
                                                'Vgg16_b + Vgg19_b + ResNet18_b', 
                                                'Vgg16 + Vgg19 + ResNet18_b + ResNet34_b',
                                                'Vgg16 + Vgg19', 
                                                'Vgg19 + ResNet18_b'])
  Vgg16, Vgg16_b, Vgg19, Vgg19_b, ResNet18_b, ResNet34_b = load_models()
  if model_option == 'Vgg16 + Vgg19 + ResNet18_b':
    if col2.button("Analyse"):
      with st.spinner('Loading...'):
        time.sleep(3)
      _, __, probs1 = Vgg16.predict(img)
      _, __, probs2 = Vgg19.predict(img)
      _, __, probs3 = ResNet18_b.predict(img)
      avg_probs = (probs1 + probs2 + probs3)/3
      ensemble_prob, ensemble_idx = torch.max(avg_probs, 0)
      st.write("Model Prediction: C", ensemble_idx.item(), "; Probability: ", ensemble_prob.item()*100,'%')
      st.write(avg_probs,'%')
  elif model_option == 'Vgg16_b + Vgg19_b + ResNet18_b':
    if col2.button("Analyse"):
      with st.spinner('Loading...'):
        time.sleep(3)
      _, __, probs1 = Vgg16_b.predict(img)
      _, __, probs2 = Vgg19_b.predict(img)
      _, __, probs3 = ResNet18_b.predict(img)
      avg_probs = (probs1 + probs2 + probs3)/3
      ensemble_prob, ensemble_idx = torch.max(avg_probs, 0)
      st.write("Model Prediction: C", ensemble_idx.item(), "; Probability: ", ensemble_prob.item()*100,'%')
      st.write(avg_probs,'%')
  elif model_option == 'Vgg16 + Vgg19 + ResNet18_b + ResNet34_b':
    if col2.button("Analyse"):
      with st.spinner('Loading...'):
        time.sleep(3)
      _, __, probs1 = Vgg16.predict(img)
      _, __, probs2 = Vgg19.predict(img)
      _, __, probs3 = ResNet18_b.predict(img)
      _, __, probs4 = ResNet34_b.predict(img)
      avg_probs = (probs1 + probs2 + probs3 + probs4)/4
      ensemble_prob, ensemble_idx = torch.max(avg_probs, 0)
      st.write("Model Prediction: C", ensemble_idx.item(), "; Probability: ", ensemble_prob.item()*100,'%')
      st.write(avg_probs,'%')
  elif model_option == 'Vgg16 + Vgg19':
    if col2.button("Analyse"):
      with st.spinner('Loading...'):
        time.sleep(3)
      _, __, probs1 = Vgg16.predict(img)
      _, __, probs2 = Vgg19.predict(img)
      avg_probs = (probs1 + probs2)/2
      ensemble_prob, ensemble_idx = torch.max(avg_probs, 0)
      st.write("Model Prediction: C", ensemble_idx.item(), "; Probability: ", ensemble_prob.item()*100,'%')
      st.write(avg_probs,'%')
  elif model_option == 'Vgg19 + ResNet18_b':
    if col2.button("Analyse"):
      with st.spinner('Loading...'):
        time.sleep(3)
      _, __, probs1 = Vgg19.predict(img)
      _, __, probs2 = ResNet18_b.predict(img)
      avg_probs = (probs1 + probs2)/2
      ensemble_prob, ensemble_idx = torch.max(avg_probs, 0)
      st.write("Model Prediction: C", ensemble_idx.item(), "; Probability: ", ensemble_prob.item()*100,'%')
      st.write(avg_probs,'%')

## Large file download functions
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)
                
@st.cache(allow_output_mutation=True)
def load_Mvgg16_trainData(img):
  download_file_from_google_drive("1VFF6WXAvR5d6uB9s3iCEgLaEer10rvbz", "./modified_VGG16.pkl")
  modified_vgg16 = load_learner('./', "modified_VGG16.pkl")
  _, __, probs = modified_vgg16.predict(img)
  urllib.request.urlretrieve(ModifiedVgg16_trainFeature_url, "trainFeature.csv")
  urllib.request.urlretrieve(ModifiedVgg16_trainTarget_url, "trainTarget.csv")
  ## Read Data CSV files
  train_f = np.array(pd.read_csv('trainFeature.csv'))
  train_t = np.array(pd.read_csv('trainTarget.csv'))
  return probs, train_f, train_t
    
def hybrid_model_options(col1, col2, img):
  model_option = col1.radio('Choose a model:', ['Vgg16 + SVM']) ## 'Vgg16 + MLP' => MLP took too long to run
  probs, train_f, train_t = load_Mvgg16_trainData(img)
  if model_option == 'Vgg16 + SVM':
    if col2.button("Analyse"):
      with st.spinner('Loading...'):
        time.sleep(3)
      SVM = SVC(random_state=42, probability=True).fit(train_f, train_t.ravel())
      prob = SVM.predict_proba(np.array(probs).reshape(1, -1))
      Max_prob_idx = np.amax(prob)
      st.write("Model Prediction: C", np.where(prob==Max_prob_idx)[1].item(), "; Probability: ", Max_prob_idx*100,'%')
      st.write(prob)
  #elif model_option == 'Vgg16 + MLP':
    #if col2.button("Analyse"):
      #with st.spinner('Loading...'):
        #time.sleep(3)
      #MLP = MLPClassifier(random_state=42).fit(train_f, train_t.ravel())
      #prob = MLP.predict_proba(np.array(probs).reshape(1, -1))
      #st.write("Probability: ", prob)

def input_image(try_test_image=False, upload_image=False, base_model=False, ensemble_model=False, hybrid_model=False):
  if try_test_image == True:
    # create 2 columns structure
    col1,col2 = st.beta_columns([1,2]) # 2nd column is 2 times of 1st column
    test_imgs = os.listdir('test-image/')
    test_img = col1.selectbox('Select a test image:', test_imgs)
    file_path = 'test-image/' + test_img
    img = open_image(file_path)
    display_img = mpimg.imread(file_path)
    # Display Image
    col2.image(display_img, use_column_width=True)
    if base_model==True:
      base_model_options(col1, col2, img, predict=True)
    elif ensemble_model==True:
      ensemble_model_options(col1, col2, img)
    elif hybrid_model==True:
      hybrid_model_options(col1, col2, img)
  elif upload_image == True:
    Uploaded = st.file_uploader('', type=['png','jpg','jpeg'])
    if Uploaded is not None:
      # create 2 columns structure
      col1,col2 = st.beta_columns([1,2])
      img = open_image(Uploaded)
      display_img = Uploaded
      # Display Image
      col2.image(display_img, use_column_width=True) 
      if base_model==True:
        base_model_options(col1, col2, img, predict=True)
      elif ensemble_model==True:
        ensemble_model_options(col1, col2, img)

# Pages
page = st.sidebar.selectbox("Choose a page", ['Baseline Model Prediction', 'Ensemble Model Prediction', 'Hybrid Model Prediction'])
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
# Hybrid-Train Data
ModifiedVgg16_trainFeature_url = "https://drive.google.com/uc?export=download&id=1Uh7JK1_uLznedvtm9_WLfsw_Aeq-DsUs"
ModifiedVgg16_trainTarget_url = "https://drive.google.com/uc?export=download&id=1itTjk7HfApLGfgrdTnvByKEsEV_KKGRl"

## 1. Page - Baseline Model Prediction
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
  link1 = '[Google Colab](https://colab.research.google.com/drive/1YWqFjd_2PXyu70D-SHcwaKbZaOs5BNU-?usp=sharing)'
  link2 = '[Github](https://github.com/xiexiangxiang/Driver-Distraction-Classification)'
  st.markdown(link1, unsafe_allow_html=True)
  st.markdown(link2, unsafe_allow_html=True)
  st.write('''---''')
  
  # Try test image / Upload image
  option = st.radio('Choose a distrated drving image', ['Try a test image', 'Upload an image'])
  if option == 'Try a test image':
    input_image(try_test_image=True, base_model=True)
  else:
    input_image(upload_image=True, base_model=True)

## 2. Page - Ensemble Model Prediction
elif page == 'Ensemble Model Prediction':
  st.title("Ensemble Driver Distraction Classification")
  st.write('''## ** Top 5 Ensemble CNN Models **''')
  link1 = '[Google Colab](https://colab.research.google.com/drive/1YWqFjd_2PXyu70D-SHcwaKbZaOs5BNU-?usp=sharing)'
  link2 = '[Github](https://github.com/xiexiangxiang/Driver-Distraction-Classification)'
  st.markdown(link1, unsafe_allow_html=True)
  st.markdown(link2, unsafe_allow_html=True)
  st.write('''---''')
  option = st.radio('Choose a distrated drving image', ['Try a test image', 'Upload an image'])
  if option == 'Try a test image':
    input_image(try_test_image=True, ensemble_model=True)
  else:
    input_image(upload_image=True, ensemble_model=True)
  
## 3. Page - Hybrid Model Prediction
elif page == 'Hybrid Model Prediction':
  st.title("Hybrid Driver Distraction Classification")
  st.write('''## ** [VGG16 + SVM] & [VGG16 + MLP] **''')
  link1 = '[Google Colab](https://colab.research.google.com/drive/1YWqFjd_2PXyu70D-SHcwaKbZaOs5BNU-?usp=sharing)'
  link2 = '[Github](https://github.com/xiexiangxiang/Driver-Distraction-Classification)'
  st.markdown(link1, unsafe_allow_html=True)
  st.markdown(link2, unsafe_allow_html=True)
  st.write('''---''')
  option = st.radio('Choose a distrated drving image', ['Try a test image', 'Upload an image'])
  if option == 'Try a test image':
    input_image(try_test_image=True, hybrid_model=True)
  else:
    input_image(upload_image=True, hybrid_model=True)
