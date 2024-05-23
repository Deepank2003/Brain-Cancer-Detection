import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model

model_path = "C:/Users/deepa/Downloads/archive/tumor_model_final.keras"
model = load_model(model_path)



def model_prediction(test_image):
    model = load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(150,150))
    img_array = np.array(image)
    img_array = img_array.reshape(1,150,150,3)
    y_pred_prob = model.predict(img_array)
    y_pred = np.argmax(y_pred_prob)
    return y_pred

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("select page",["Home","About","Tumor Prediction","Suggestions"])

if(app_mode=="Home"):
    st.header("BRAIN TUMOR DETECTION SYSTEM")
    image_path = "app background.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    ###### Welcome to our Brain Tumor Detection App, a cutting-edge tool designed to assist healthcare professionals and individuals in the early detection and management of brain tumors.
    #
    Our app harnesses the power of artificial intelligence and advanced imaging analysis algorithms to provide accurate and efficient screening for brain tumors. With just a few simple steps, users can upload brain imaging scans, such as MRI or CT scans, to our secure platform for analysis.
    ###
    Using state-of-the-art machine learning technology, our app swiftly analyzes the uploaded scans to identify potential signs of brain tumors, including gliomas, meningiomas, and pituitary tumors. The app provides detailed reports highlighting any abnormalities detected, along with recommendations for further evaluation and management.

    ### Key features of our Brain Tumor Detection App include:

    1. **Fast and Accurate Analysis:** Our app delivers rapid and reliable analysis of brain imaging scans, allowing healthcare professionals to promptly identify and assess potential brain tumors.
    2. **User-Friendly Interface:** With an intuitive and easy-to-navigate interface, our app ensures a seamless experience for both healthcare professionals and individuals seeking screening for brain tumors.
    3. **Secure Data Handling:** We prioritize the privacy and security of user data, employing robust encryption and adherence to strict data protection standards to safeguard sensitive medical information.
    4. **Comprehensive Reporting:** Our app generates comprehensive reports summarizing the findings of the scan analysis, empowering healthcare professionals to make informed decisions regarding patient care and treatment.
    5. **Continual Improvement:** We are committed to ongoing research and development to enhance the accuracy and effectiveness of our app, ensuring that it remains at the forefront of brain tumor detection technology.

    Whether you are a healthcare professional seeking a reliable screening tool or an individual concerned about potential symptoms, our Brain Tumor Detection App is here to support you on your journey to early detection and improved patient outcomes.
    Join us in the fight against brain tumors and together, let's make a difference in the lives of patients and their families.

    Our goal is to help in Identifying Brain Tumors efficiently. Upload an Image of the MRI scan of the Brain , and our system will Analyze and predict whether the image has Brain Tumor or not, and if it does then it will classify the Type of the Tumor also!
    #
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of the MRI scan of the Brain with suspected disease.
    2. **Analysis:** Our system will process the Image using Advanced Algorithms to identify potential Tumor.
    3. **Results:** View the Results and consult accordingly.


    
      """)

elif(app_mode=="About"):
        st.header("About")
        st.markdown(""" 


### About The Brain Tumors:

1. **Glioma:** Glioma is a type of tumor that originates in the glial cells of the brain or spine. Glial cells are supportive cells that surround and support the neurons in the nervous system. Gliomas are the most common type of primary brain tumor, meaning they originate in the brain rather than spreading from another part of the body. 
#
2. **Pituitary:** Pituitary tumors are classified based on their size and whether they are cancerous (malignant) or non-cancerous (benign). The majority of pituitary tumors are benign, meaning they do not spread to other parts of the body. However, even benign tumors can cause health problems by pressing on nearby structures in the brain or by disrupting hormone production and balance.
#
3. **Meningioma:** Meningiomas are tumors that arise from the meninges, which are the protective layers of tissue that cover the brain and spinal cord. These tumors are typically slow-growing and are usually benign (non-cancerous), although they can occasionally be malignant (cancerous). Meningiomas are the most common type of primary brain tumor, accounting for approximately one-third of all brain tumors.
#
### About Dataset
This dataset contains 7023 images of human brain MRI images which are classified into 4 classes: Glioma - Meningioma - No tumor and Pituitary. The dataset is provided for research purposes only and it is sourced from Kaggle.com/datasets/Brain tumor MRI.
#### Content:

1. Total (7023 images)
2. Training (5702 images)
3. Testing (1311 images)
#
### About our Model
Our Model is Trained using Advanced Machine Learning techniques to provide the Model with Maximum Accuracy and Efficiency. We used multiple Python Modules for our conveniences to build the Model. 

""")

elif(app_mode=="Tumor Prediction"):
        st.header("Tumor Prediction")
        test_image = st.file_uploader("Choose an Image:" ,type=['jpg','png','jpeg'])
        if(st.button("Show Image")):
            st.image(test_image, use_column_width=True)

        if(st.button("Predict")):
            with st.spinner("Please Wait while our Model is Predicting"):  
                st.write("Our Prediction")
                y_pred = model_prediction(test_image)
                labels = ['glioma','meningioma','notumor','pituitary']
                img_array = np.array(test_image)
            st.success("Model is Predicting it's a {} ".format(labels[y_pred]))

elif(app_mode=="Suggestions"):
        st.header("Suggestions")
        st.markdown("""""")