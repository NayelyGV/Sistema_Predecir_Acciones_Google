import numpy as np #biblioteca para operaciones matematicas y manipulaciones de matrices y vectores
import datetime as dt #biblioteca para trabajar con fecha y tiempo
import tensorflow as tf #biblioteca para construir, entrenar y evaluar modelos de redes neuronales 
from yahoo_fin import stock_info as yf #obtener datos de precios de acciones de Yahoo Finance
from sklearn.preprocessing import MinMaxScaler #escalar los datos a un rango específico
from keras.models import Sequential #crea una secuencia de capas en una red neuronal
from keras.layers import Dense, LSTM, Dropout #Dense:crea capas densas o completamente conectadas en la red
#LSTM:construye redes neuronales recurrentes (RNN) con memoria a largo plazo
#Dropout: reduce el sobreajuste en las redes neuronales
import matplotlib.pyplot as plt #Grafica los resultados

# SETTINGS
N_STEPS = 7 #utilizara los ultimos 7 paso para predecir el siguiente
LOOKUP_STEPS = 10 #numeros de dias o pasos en el futuro
TRAIN_SIZE = 0.8#80% de los datos para entrenamiento y 20% para validacion
EPOCHS = 50 #cuantas veces se vera todo el conjuunto de entrenamiento
BATCH_SIZE = 64 #numero de muestras para cada paso de entrenamiento

# Recopilación de datos históricos
def load_data(ticker, n_steps=7):
    data = yf.get_data(ticker)
    data = data.iloc[::-1]  # Invertir el orden de los datos para que los más recientes estén al final
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data[['close']].values)
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# División de datos
def train_test_split(X, y, train_size=0.8):
    split = int(train_size * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train, X_test, y_train, y_test

# Creación del modelo
def build_model(n_steps, n_features, dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(128, input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Entrenamiento del modelo
def train_model(model, X_train, X_test, y_train, y_test, epochs, batch_size):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

# Predicción de precios
def predict_prices(model, X, scaler, lookup_steps):
    prediction = model.predict(X)
    prediction = scaler.inverse_transform(prediction)
    for i in range(len(lookup_steps)):
        future_price = model.predict(X[-N_STEPS:])
        future_price = scaler.inverse_transform(future_price)
        X = np.append(X, future_price.reshape((1,N_STEPS,1)), axis=0)
        prediction = np.append(prediction, future_price)
    return prediction

# Cargar y preparar los datos
X, y, scaler = load_data('GOOGL', N_STEPS)
X_train, X_test, y_train, y_test = train_test_split(X, y, TRAIN_SIZE)

# Crear y entrenar el modelo
model = build_model(N_STEPS, 1)
train_model(model, X_train, X_test, y_train, y_test, EPOCHS, BATCH_SIZE)

# Realizar predicciones
last_sequence = X[-N_STEPS:]
if len(last_sequence) % N_STEPS != 0:
    last_sequence = last_sequence[:-(len(last_sequence) % N_STEPS)]
last_sequence = last_sequence.reshape((-1, N_STEPS, 1))
n_steps = int(len(last_sequence) / LOOKUP_STEPS)
predicted_prices = predict_prices(model, last_sequence, scaler, range(1, n_steps+1))

# Graficar los resultados
actual_prices = scaler.inverse_transform(y[-len(predicted_prices):].reshape(-1,1))
dates = yf.get_data('GOOGL').tail(len(predicted_prices)).index
plt.plot(dates, actual_prices, label='Precio real')
plt.plot(dates, predicted_prices, label='Precio predicho')
plt.title('Predicción de precios de acciones de Google')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.legend()
plt.show()
#plt.savefig('predicciones.png')
