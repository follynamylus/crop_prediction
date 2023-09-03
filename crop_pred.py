import pandas as pd
import streamlit as st 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

tab_1,tab_2 = st.tabs(['VIEW PREDICTION','DATAFRAME AND DOWNLOAD'])

model = pickle.load(open("rf_model", 'rb'))

option = st.sidebar.selectbox("Choose the type of prediction to perform",["Single","Multiple"])
if option.lower() == "single" :
    st.sidebar.title("Data Input")
    n = st.sidebar.number_input("Input ratio of Nitrogen content in soil",0.00,140.00,step=1e-2,format="%.2f")
    p = st.sidebar.number_input("Input ratio of Phosphorus content in soil",5.00,145.00,step=1e-2,format="%.2f")
    k = st.sidebar.number_input("Input ratio of Potassium content in soil",5.00,205.00,step=1e-2,format="%.2f")
    temp = st.sidebar.number_input("Input the degree of Temperature",8.00,42.00,step=1e-2,format="%.2f")
    hum = st.sidebar.number_input("Input the Humidity value",14.00,95.00,step=1e-2,format="%.2f")
    ph = st.sidebar.number_input("Input the PH value",3.00,10.00,step=1e-2,format="%.2f")
    rain = st.sidebar.number_input("Input the amount of rainfall",5.00,300.00,step=1e-2,format="%.2f")
    df = pd.DataFrame()

    df['N'] = [n]
    df['P'] = [p]
    df['K'] = [k]
    df['temperature'] = [temp]
    df['humidity'] = [hum]
    df['ph'] = [ph]
    df['rain'] = [rain]

    data = df.copy()

    pred = model.predict(data)
    
    df["prediction"] = pred
    
    tab_1.title(f"From the features provided, The crop to plant is {pred}")

    tab_2.dataframe(df)
    proba = model.predict_proba(data)
    fig = sns.barplot(x=np.arange(len(proba[0])),y=proba[0])
    classes = ["Soyabeans","Apple","Banana","Beans","Coffee","Cotton","Cowpeas","Grapes","Groundnuts","Maize",
               "Mango","Orange","Peas","Rice","Watermelon"]
    plt.xticks(np.arange(len(proba[0])), labels=[f"Class {i}" for i in classes])
    plt.xlabel('Crop Probability')
    plt.ylabel('Probability')
    plt.xticks(rotation=90)
    plt.title(f'Predicted Probabilities for Single Sample')
    plt.savefig('predicted_probabilities.png')
    tab_2.pyplot()
else :
    file = st.sidebar.file_uploader("Upload file to test")
    if file == None :
        st.write("Input File")
    else :
        df = pd.read_csv(file)
        tab_2.success("Datafram before prediction")
        tab_2.dataframe(df)
        pred = model.predict(df)
        df['predictions'] = pred
        tab_1.success("Predicted Classes value counts")
        tab_1.write(df['predictions'].value_counts())
        tab_2.success("Datafram after prediction")
        tab_2.dataframe(df)

        fig = sns.countplot(data = df, x= "predictions")
        plt.xticks(rotation=90)
        plt.savefig('predicted_probabilities.png')
        tab_1.success("Prediction count plot for each class")
        tab_1.pyplot()
        @st.cache_data
        def convert_df(df):
            '''
            Convert_df function converts the resulting dataframe to a CSV file.
            It takes in a data frame as a aprameter.
            It returns a CSV file
            '''
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        tab_2.success("Print Result as CSV file")
        tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')