import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the Keras model
model = load_model('CABAI\CODE\PCD\Chili_Plant_Disease\model\disease.h5')

# Function to preprocess the image and make predictions
def predict_disease(image):
    # Preprocess the image if necessary
    # Make predictions using the model
    prediction = model.predict(image)
    return prediction

def main():
    st.title('Deteksi Penyakit pada Daun Cabai')

    # Upload image
    uploaded_file = st.file_uploader("Upload gambar daun cabai", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption='Gambar daun cabai', use_column_width=True)

        # If there's an image, perform prediction
        if st.button('Deteksi'):
            # Preprocess image if needed
            # Perform prediction
            prediction = predict_disease(image)
            st.write('Prediksi:', prediction)

if __name__ == '__main__':
    main()
