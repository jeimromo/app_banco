import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import joblib

#------------PRESENTACION-------------
st.image('imagenes/logo1.jpg')
st.write("<h2 style='text-align: center; font-family: Candara; color:rgb(0, 62, 0);'> Aprobación Crédito Bancario </h2>",unsafe_allow_html=True)

col1, col2= st.columns([1, 1]) 
with col1:
    st.write("<h5 style='text-align: justify; color:rgb(21, 114, 67);'> Queremos hacer tu vida más fácil,\
        por eso hemos utilizado lo mejor de la tecnología para que  \
        mediante el uso de inteligencia artificial  (IA)\
        puedas realizar la solicitud de tu crédito en tiempo real</h5>",unsafe_allow_html=True)
with col2:
    st.image("imagenes/presentacion.jpg")
c1 =st.container()
c1.write("<h3 style='text-align: center; color:rgb(0, 62, 0);'> ¿Cómo funciona? </h3>",unsafe_allow_html=True)
c1.write("<p style='text-align: justify'> Registra tus datos en la barra de la izquierda y da clic en consultar ahora</p>",unsafe_allow_html=True)

#---------------------------------------

#--Cargar los datos de entrenamiento
data = pd.read_csv("credit_drop.csv") #para cargar los datos de entrenamiento
df =data.drop(['ID','Aprobado'],axis=1) #se dejan solo las columnas necesarias
atributos = df.columns.values.tolist() #se crea la lista con los nombres de las columnas


#--Escalador
columnas_x = data[atributos] 
x = columnas_x.values

from sklearn.preprocessing import StandardScaler #realizó el escalador ya que los datos deben quedar entre 0,1
scaler = StandardScaler().fit(x) #Utilizo los datos de entrenamiento para hacer el escalador

#--------------BARRA LATERAL---------------
st.sidebar.markdown("<h1 style='text-align: left; font-size: 30px; color: grey;'> \
                Registra tus datos </h1>", unsafe_allow_html=True)

bar= st.sidebar.container()
with bar:
#--Ingreso de los datos   
    x_usuario = np.zeros(len(atributos))
    for i in range(len(atributos)):
        if i==1 or i==2 or i==7:
            x_usuario[i] = st.number_input(atributos[i], min_value = 0)
        else:
            x_usuario[i] = st.number_input(atributos[i], min_value = 0, step=1)

#-------RESULTADOS DEL MODELO---------------
c2 = st.container()
 
with c2:
    modelo = joblib.load('modelo_arbolTotal.joblib')
    x = x_usuario.reshape(1,-1)
    x2 = scaler.transform(x)

    y_pred = modelo.predict(x2)

    if y_pred < 0.5:
        y_pred = 0
    else:
        y_pred = 1


#--------BUTTON
colb1, colb2, colb3= st.columns([1, 1, 1])
with colb2:
    resultado= st.button(" consultar ahora ▶", key=None, on_click=None, args=None, kwargs=None)

if resultado:

    if y_pred == 1:
        st.error("😔Lo sentimos tu crédito no fue aprobado contacta con uno de nuestros agentes \
            y prepara todo lo necesario para una próxima oportunidad")
        st.image("imagenes/rechazado.png")
    if y_pred == 0:
        st.success("🥳Felicidades tu crédito fue aprobado estás a unos pasos de alcanzar tus sueños 🎉\
        nuestros agentes te contactarán para que recibas todos nuestros beneficios")
        st.image("imagenes/aprobado.png")


#-----------------