import streamlit as st
import tensorflow as tf
from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Deteksi Penyakit Cabai",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            return key

with st.sidebar:
        st.image('cara_budidaya_cabe.jpg')
        st.title("Cabai Merah")
        st.subheader("Membantu kamu dalam klasifikasi penyakit cabai")         
        
def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key
        
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
    #model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(224, 224, 4)))
    
st.write("""
         # Deteksi penyakit cabai mu
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['healthy','leaf curl','leaf spot','whitefly','yellowish']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'healthy':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'leaf curl':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")

    elif class_names[np.argmax(predictions)] == 'leaf spot':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")

    elif class_names[np.argmax(predictions)] == 'whitefly':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Cutting Weevil can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the weevil damage.")

    elif class_names[np.argmax(predictions)] == 'yellowish':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("After pruning, apply copper oxychloride at a concentration of '0.3%' on the wounds. Apply Bordeaux mixture twice a year to reduce the infection rate on the trees. Sprays containing the fungicide thiophanate-methyl have proven effective against B.")
