import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import joblib

#------------PRESENTACION-------------
st.image('imagenes/logo1.jpg')
c1 =st.container()
c1.title("Aprobación crédito Bancario")
c1.write("<h5 style='text-align: justify; color:rgb(21, 114, 67);'> Queremos hacer tu vida más fácil, por eso \
    hemos utilizado lo mejor de la tecnología para que  \
    mediante el uso de inteligencia artificial  \
    puedas realizar la solicitud de crédito en tiempo real</h5>",unsafe_allow_html=True)
c1.write("<h3 style='text-align: center; font-family: Courier New; color:black;'> ¿Cómo funciona? </h3>",unsafe_allow_html=True)
c1.write("<p style='text-align: justify; color:black;'> Registra tus datos en la barra de la izquierda \
    y da clic en consultar ahora</p>",unsafe_allow_html=True)

#---------------------------------------

#--Cargar los datos de entrenamiento
data = pd.read_csv("credit_drop.csv") #para cargar los datos de entrenamiento
df =data.drop(['ID','Aprobado'],axis=1) #se dejan solo las columnas necesarias
atributos = df.columns.values.tolist()


#--Escalador
columnas_x = data[atributos] 
x = columnas_x.values

from sklearn.preprocessing import StandardScaler #realizó el escalador ya que los datos deben quedar entre 0,1
scaler = StandardScaler().fit(x) #Utilizo los datos de entrenamiento para hacer el escalador

#--------------BARRA LATERAL---------------
st.sidebar.markdown("<h1 style='text-align: left; font-size: 30px; color: grey;'> \
                Registre sus datos </h1>", unsafe_allow_html=True)

bar= st.sidebar.container()
with bar:
#--Ingreso de los datos   
    x_usuario = np.zeros(len(atributos))
    for i in range(len(atributos)):
        x_usuario[i] = st.number_input(atributos[i])

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


#----BUTTON
resultado= st.button("🚀 consultar ahora", key=None, on_click=None, args=None, kwargs=None)

if resultado:

    if y_pred == 0:
        st.write("Lo sentimos tu crédito no fue aprobado 😔, contacta con uno de nuestros agentes \
            y prepara todo lo necesario para una próxima oportunidad</h3>",unsafe_allow_html=True)
        st.image("imagenes/rechazado.png")
    if y_pred == 1:
        st.write("Felicidades su crédito fue aprobado 🥳")
        st.image("imagenes/aprobado.png")

#-----------------