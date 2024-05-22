import streamlit as st
import tensorflow as tf
import numpy as np



#function definition

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 

#Home

st.header("Fruits & Vegetable Recognition System")
img_path="home_img.jpg"
st.image(img_path)

test_image = st.file_uploader("Choose an image:")
if(st.button("Show image")):
    st.image(test_image,width=4,use_column_width=True)

if(st.button("Predict")):
    st.write("Our Prediction")
    result_index = model_prediction(test_image)
    with open("labels.txt") as f:
            content = f.readlines()
    label = []
    for i in content:
            label.append(i[:-1])
    st.success("Model is Predicting it's a {}".format(label[result_index]))

