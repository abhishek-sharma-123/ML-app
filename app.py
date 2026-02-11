import streamlit as  st
import numpy as np
import pandas as pd
import joblib
import pickle
import os


st.write("🚀 App started")
st.write("Working dir:", os.getcwd())
st.write("Files in root:", os.listdir())

st.write("Models dir exists:", os.path.exists("models"))
st.write("Models content:", os.listdir("models") if os.path.exists("models") else "NO MODELS")

st.write("Car model exists:", os.path.exists("models/car_price/model.pkl"))
st.write("Laptop model exists:", os.path.exists("models/laptop_price/laptop.pkl"))


car_model = joblib.load("Models/CarPrice/CarPricePipe.pkl")


st.write("Laptop model exists:", os.path.exists("models/laptop_price/laptop.pkl"))
laptop_model = joblib.load("Models/LaptopPrice/pipe.pkl")


st.write("House model exists:", os.path.exists("models/HousePrice/HousePricePipe.pkl"))
house_model = joblib.load("Models/HousePrice/HousePricePipe.pkl")
laptop_df = pd.read_csv('Models/LaptopPrice/laptop_cleaned.csv')
house_df = pd.read_csv("Models/HousePrice/house_cleaned.csv")

# @st.cache_resource
# def load_models():
#     st.write("Loading car model...")
#     car = joblib.load(
#         os.path.join(BASE_DIR, "Models", "CarPrice", "CarPricePipe.pkl")
#     )
#     st.write("Loaded car model...")

#     st.write("Loading laptop model...")
#     laptop = joblib.load(
#         os.path.join(BASE_DIR, "Models", "LaptopPrice", "pipe.pkl")
#     )
#     st.write("Loaded laptop model...")

#     st.write("Loading house model...")
#     house = joblib.load(
#         os.path.join(BASE_DIR, "Models", "HousePrice", "HousePricePipe.pkl")
#     )
#     st.write("Loaded house model...")

#     laptop_df = pd.read_csv(
#     os.path.join(BASE_DIR, "Models", "LaptopPrice", "laptop_cleaned.csv")
#     )
#     house_df = pd.read_csv(
#     os.path.join(BASE_DIR, "Models", "HousePrice", "house_cleaned.csv")
#     )
#     # car = joblib.load("Models/CarPrice/CarPricePipe.pkl")
#     # laptop = joblib.load("Models/LaptopPrice/pipe.pkl")
#     # house = joblib.load("Models/HousePrice/HousePricePipe.pkl")
#     # laptop_df = pd.read_csv('Models/LaptopPrice/laptop_cleaned.csv')
#     # house_df = pd.read_csv("Models/HousePrice/house_cleaned.csv")
#     return car, laptop, house, laptop_df, house_df

# car_model, laptop_model, house_model, laptop_df, house_df = load_models()


st.set_page_config(page_title="Multi ML Predictor", layout="centered")

st.title("Multi ML Price Prediction App")

prediction_type = st.selectbox(
    "What do you want to predict?",
    ["Select", "Car Price", "Laptop Price", "House Price"]
)

# Car 
if prediction_type == "Car Price":
    st.header("🚗 Car Price Prediction")
    company = st.text_input("Company Name")
    name = st.text_input("Model Name")
    year = st.number_input("Year of Mfg")
    kms_driven = st.number_input("Kms driven")
    fuel_type = st.selectbox("Fuel Type",["Petrol","Diesel"])


    if st.button("Predict"):
        raw_input = {
            'name':name,
            'company' :company,
            'year':year,
            'kms_driven':kms_driven,
            'fuel_type':fuel_type
        }

        input_df = pd.DataFrame([raw_input])

        prediction = car_model.predict(input_df)[0]
        st.title(f"Estimated Car Price : Rs {round(prediction,2)}")

# Laptop

elif prediction_type == "Laptop Price":
    st.header("💻 Laptop Price Prediction")
    company = st.selectbox("Brand",laptop_df["Company"].unique())
    typeName = st.selectbox("Type Name",["Notebook","Gaming","Ultrabook","2 in 1 Convertible","Workstation","Netbook"])
    ram = st.selectbox("RAM",[2,4,6,8,12,16,24,32,64])
    weight = st.number_input("Weight")
    touchscreen = st.selectbox("Touchscreen",["No","Yes"])
    ips = st.selectbox("IPS",["No","Yes"])
    screen_size = st.number_input("Screensize in inches",10.0,18.0,13.0)
    resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
    cpu = st.selectbox('CPU',laptop_df['Cpu brand'].unique())
    hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
    ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
    gpu = st.selectbox('GPU',laptop_df['Gpu brand'].unique())
    os = st.selectbox('OS',laptop_df['os'].unique())

    if st.button("Predict Laptop Price"):
        ppi = None
        if touchscreen=="Yes":
            touchscreen=1
        else:
            touchscreen=0
        if ips == "Yes":
            ips =1
        else:
            ips =0
        X_res = int(resolution.split("x")[0])
        Y_res = int(resolution.split("x")[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
        query = pd.DataFrame([{
            "Company": company,
            "TypeName": typeName,
            "Ram": ram,
            "Weight": weight,
            "Touchscreen": touchscreen,
            "Ips": ips,
            "ppi": ppi,
            "Cpu brand": cpu,
            "HDD": hdd,
            "SSD": ssd,
            "Gpu brand": gpu,
            "os": os
        }])

        st.title("Estimated Laptop Price : Rs " + str(int(np.exp(laptop_model.predict(query)[0]))))

# House Price

elif prediction_type == "House Price":
    st.header("🏠 House Price Prediction")
    location = st.selectbox("Location",house_df["location"].unique())
    total_sqft = st.number_input("Total Area in sqft")
    bath = st.selectbox("No. of bathrooms",[1,2,3,4,5,6,7,8])
    balcony = st.selectbox("No. of balcony",[0,1,2,3])
    bhk = st.selectbox("BHK",house_df["bhk"].unique())
    if st.button("Predict House Price"):
        query = pd.DataFrame([{
            "location" :location,
            "total_sqft" :total_sqft,
            "bath" : bath,
            "balcony": balcony,
            "bhk" :bhk,
        }])

        prediction = round(house_model.predict(query)[0],2)
        st.title("Estimated House Price : Rs " + str(prediction) + " (in lakhs)")
else:
    st.info("Please select a prediction type from above ☝️")
