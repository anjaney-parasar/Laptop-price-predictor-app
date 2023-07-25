import pandas as pd 
import numpy as np
import streamlit as st	
import pickle
import sklearn
from sklearn.externals import joblib
from joblib import dump, load
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler	

st.write("# Laptop Price Predictor")
st.sidebar.header("Select Laptop Features")



def user_input_features():
	company=st.sidebar.selectbox('Company',('HP', 'Dell', 'Lenovo', 'MSI', 'Acer', 'Chuwi', 'Asus', 'Toshiba', 'Apple', 'Razer', 'Mediacom', 'LG', 'Samsung', 'Microsoft', 'Fujitsu', 'Vero', 'Google', 'Xiaomi', 'Huawei', 'Others'))
	typename=st.sidebar.selectbox('Type',('Workstation', 'Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Netbook'))
	inches=st.sidebar.slider("Screen size(inches)",10.0,18.4,15.0)
	scrres=st.sidebar.selectbox('Screen Resolution',('1920x1080', '1366x768', '3840x2160', '2736x1824', '3200x1800', '2560x1600', '2304x1440', '2560x1440', '2880x1800', '1600x900', '2256x1504', '1920x1200', '2400x1600', '1440x900', '2160x1440'))
	cpu=st.sidebar.selectbox("CPU",('Intel', 'AMD', 'Samsung'))
	ram=st.sidebar.select_slider('RAM',[2,4,6,8,12,16,24,32,64])
	gpu=st.sidebar.selectbox('GPU',('Nvidia Quadro', 'Intel HD', 'Nvidia GeForce', 'Intel UHD', 'AMD Radeon', 'Intel Iris', 'AMD FirePro', 'ARM Mali', 'AMD R4', 'Nvidia GTX', 'AMD R17M-M1-70', 'Intel Graphics'))
	opsys=st.sidebar.selectbox('Operating System',('Windows','Mac', 'Others/No OS/Linux'))
	weight=st.sidebar.slider('Weight(in Kg)',0.65,4.9,2.5)
	memory_in_gb=st.sidebar.select_slider('Memory/ROM(in GB)',[8,16,32,64,128,180,240,256,500,508,512,1000,2000])
	memory_type=st.sidebar.selectbox('Memory Type',('SSD', 'HDD', 'SSD +  1TB HDD', 'Flash Storage', 'Hybrid', 'SSD +  1.0TB Hybrid', 'SSD +  2TB HDD', 'SSD +  256GB SSD', 'Flash Storage +  1TB HDD', 'SSD +  500GB HDD', 'SSD +  512GB SSD', 'HDD +  1TB HDD'))
	data={'Company':company, 'TypeName':typename, 'Inches':inches, 'ScreenResolution':scrres, 'Cpu':cpu,
       'Ram_in_GB':ram, 'Gpu':gpu, 'OpSys':opsys, 'Weight':weight, 'Memory_type':memory_type,
       'Memory_in_GB':memory_in_gb}
	features=pd.DataFrame(data,index=[0])
	return features
input_df=user_input_features()



laptop=pd.read_csv('laptop_price_cleaned.csv')
laptop_features=laptop.drop(columns=['Price_rupees','Product'])
df=pd.concat([input_df,laptop_features])

encode=['Company',
 'TypeName',
 'ScreenResolution',
 'Cpu',
 'Gpu',
 'OpSys',
 'Memory_type']


#Encoding the data: we are using the original data too so that we get all labels
le=LabelEncoder()
for i in encode:
    df[i]=le.fit_transform(df[i])


#Scaling numeric features
sc=StandardScaler()
num_cols=[ 'Inches', 'Ram_in_GB', 'Weight', 'Memory_in_GB']
sc.fit(df[num_cols][1:])
df[num_cols]=sc.transform(df[num_cols])
print(df.head())



df=df[:1]#taking just the input feature
print(df)

#Apply the model and predict the price COOL!
model=load('regression_model.joblib')
y_pred=model.predict(df)
print(y_pred)
st.subheader('Predicted Price in rupees')
st.write(np.round(y_pred,0))

#streamlit run "D:\used car\something.py" 
