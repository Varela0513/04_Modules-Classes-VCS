import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgboost
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt



from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression

#1. Data Loading
# Load the datasets
train_data = pd.read_csv('aapl_5m_train.csv')
validation_data = pd.read_csv('aapl_5m_validation.csv')

#2. Feature Engineering
# Calculate moving averages
train_data['MA5'] = train_data['Close'].rolling(window=5).mean()
train_data['MA10'] = train_data['Close'].rolling(window=10).mean()

# Define target variable: 1 if the next close is higher than the current close, else 0
train_data['Target'] = (train_data['Close'].shift(-1) > train_data['Close']).astype(int)

# Drop NA values
train_data.dropna(inplace=True)

# Define independent variables and target variable
X = train_data[['MA5', 'MA10']]
Y = train_data['Target']

#3. Modeling
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X, Y)
y_hat = log_reg.predict(X)

# SVC
svc = SVC()
svc.fit(X, Y)

# XGBoost
boosted_model = xgboost.XGBClassifier()
boosted_model.fit(X, Y)
y_hat_boost = boosted_model.predict(X)

print(confusion_matrix(Y, y_hat_boost))
print(confusion_matrix(Y, y_hat))
print(accuracy_score(Y, y_hat_boost))


# Adaptación de datos para TensorFlow
X_tf = np.array(X)
Y_tf = np.array(Y)

# Crear modelo MLP
model_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_tf.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar modelo
model_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar modelo
model_mlp.fit(X_tf, Y_tf, epochs=10, batch_size=10)


# 4. Optimization
def optimize_params(x: np.array) -> float:
    gamma, reg_alpha = x  # Unpack parameters
    n_estimators = 1
    model_ = xgboost.XGBClassifier(n_estimators=n_estimators,
                                   gamma=gamma,
                                   reg_alpha=reg_alpha,
                                   reg_lambda=reg_alpha)
    model_.fit(X, Y)
    y_pred = model_.predict(X)
    acc = accuracy_score(Y, y_pred)
    return -acc


bnds = ((0, 10), (1e-4, 10))
x0 = [0, 1e-3]
res = minimize(optimize_params, bounds=bnds, x0=x0, method="Nelder-Mead", tol=1e-10)
print(res)
print(res.x)

opt_model = xgboost.XGBClassifier(n_estimators=10,
                                  gamma=res.x[0],
                                  reg_alpha=res.x[1],
                                  reg_lambda=res.x[1])
opt_model.fit(X, Y)
print(confusion_matrix(Y, opt_model.predict(X)))


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                                    activation='relu', input_shape=(X_tf.shape[1],)))
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of combinations de hiperparameters to test
    executions_per_trial=3,  # Times to run each combination
    directory='my_dir',  # Directory to save logs
    project_name='intro_to_kt')

tuner.search(X_tf, Y_tf, epochs=10, validation_split=0.2)

#5. Combination of Models
def combine_predictions_dl(preds):
    combined = sum(preds)
    return 1 if combined >= len(preds) / 2 else 0

# Predicciones de modelos de Deep Learning
y_hat_mlp = (model_mlp.predict(X_tf) > 0.5).astype(int).flatten()
# Predicciones de la Regresión Logística
y_hat_lr = y_hat  # Ya tienes esta variable definida

# Predicciones del SVC
y_hat_svc = svc.predict(X)

# Predicciones de XGBoost
y_hat_xgb = y_hat_boost  # Ya tienes esta variable definida

# Predicciones del Modelo MLP
y_hat_mlp = (model_mlp.predict(X_tf) > 0.5).astype(int).flatten()

# Combinar predicciones de todos los modelos
combined_predictions_dl = [combine_predictions_dl([y_hat_lr[i], y_hat_svc[i], y_hat_xgb[i], y_hat_mlp[i]]) for i in range(len(y_hat_lr))]


# Combinar predicciones de todos los modelos
combined_predictions_dl = [combine_predictions_dl([y_hat_lr[i], y_hat_svc[i], y_hat_xgb[i], y_hat_mlp[i]]) for i in range(len(y_hat_lr))]

#6. Backtesting
initial_cash = 1000000  # Starting cash
cash = initial_cash
stock = 0
portfolio_values = []
stop_loss_percentage = 0.05  # 5% stop loss
take_profit_percentage = 0.10  # 10% take profit
commission_rate = 0.001  # 0.1% commission
combined_predictions = combined_predictions_dl

buy_price = 0  # Precio de compra para calcular stop-loss y take-profit

for i in range(len(train_data) - 1):
    current_price = train_data['Close'].iloc[i]
    if stock > 0:
        # Calcula el cambio porcentual desde el precio de compra
        change_percentage = (current_price - buy_price) / buy_price

        # Vender si se alcanza el stop-loss o el take-profit
        if change_percentage <= -stop_loss_percentage or change_percentage >= take_profit_percentage:
            cash += stock * current_price * (1 - commission_rate)
            stock = 0
            continue

    # Señal de compra
    if combined_predictions[i] == 1 and cash >= current_price:
        stock += 1
        cash -= current_price * (1 + commission_rate)
        buy_price = current_price
    # Señal de venta
    elif combined_predictions[i] == 0 and stock > 0:
        cash += stock * current_price * (1 - commission_rate)
        stock = 0

    portfolio_values.append(cash + stock * current_price)

#7. Strategy Selection & Validation
# Feature engineering for validation data
validation_data['MA5'] = validation_data['Close'].rolling(window=5).mean()
validation_data['MA10'] = validation_data['Close'].rolling(window=10).mean()
validation_data.dropna(inplace=True)
X_val = validation_data[['MA5', 'MA10']]

# Predictions on validation data
y_hat_val_lr = log_reg.predict(X_val)
y_hat_val_svc = svc.predict(X_val)
y_hat_val_xgb = opt_model.predict(X_val)

# Define la función para combinar predicciones si aún no está definida
def combine_predictions_dl(preds):
    combined = sum(preds)
    return 1 if combined >= len(preds) / 2 else 0

# Combine predictions
combined_predictions_val = [combine_predictions_dl([y_hat_val_lr[i], y_hat_val_svc[i], y_hat_val_xgb[i]]) for i in range(len(y_hat_val_lr))]

# Resto del código de backtesting en los datos de validación
cash = initial_cash
stock = 0
portfolio_values_val = []

for i in range(len(validation_data) - 1):
    current_price = validation_data['Close'].iloc[i]
    if stock > 0:
        change_percentage = (current_price - buy_price) / buy_price
        if change_percentage <= -stop_loss_percentage or change_percentage >= take_profit_percentage:
            cash += stock * current_price * (1 - commission_rate)
            stock = 0
            continue

    if combined_predictions_val[i] == 1 and cash >= current_price:
        stock += 1
        cash -= current_price * (1 + commission_rate)
        buy_price = current_price
    elif combined_predictions_val[i] == 0 and stock > 0:
        cash += stock * current_price * (1 - commission_rate)
        stock = 0

    portfolio_values_val.append(cash + stock * current_price)

portfolio_values_val.append(cash + stock * validation_data['Close'].iloc[-1])

#8. Results & Conclusions
# Plot portfolio values for training data
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label="Training Data Portfolio Value")
plt.plot(portfolio_values_val, label="Validation Data Portfolio Value", color='orange')
plt.title("Portfolio Value Over Time")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

# Asegúrate de que las longitudes de las listas son correctas
print("Longitud de los datos de entrenamiento:", len(train_data))
print("Longitud de portfolio_values:", len(portfolio_values))
print("Longitud de los datos de validación:", len(validation_data))
print("Longitud de portfolio_values_val:", len(portfolio_values_val))




