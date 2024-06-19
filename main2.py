import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Cargar datos
data = pd.read_csv('muestras.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
n_samples, n_features = X.shape

# Crear el modelo
model = gp.Model()

# Crear variables para los pesos, la constante y las variables z
w = model.addVars(n_features, name="w")
b = model.addVar(name="b")
z = model.addVars(n_features, name="z")

# Parámetro de regularización
lambda_val = 1.0  # Este valor debe ajustarse para obtener solo 5 características no cero

# Definir la función objetivo
obj = gp.quicksum((gp.quicksum(w[i] * X[r, i] for i in range(n_features)) + b - Y[r]) ** 2 for r in range(n_samples)) \
      + lambda_val * gp.quicksum(z[i] for i in range(n_features))
model.setObjective(obj, GRB.MINIMIZE)

# Añadir restricciones
model.addConstrs((w[i] <= z[i] for i in range(n_features)), "c1")
model.addConstrs((w[i] >= -z[i] for i in range(n_features)), "c2")

# Optimizar el modelo
model.optimize()

# Encontrar las cinco características más importantes
w_values = [w[i].X for i in range(n_features)]
significant_features = sorted(range(n_features), key=lambda i: abs(w_values[i]), reverse=True)[:5]

# Imprimir los resultados
features = data.columns[:-1]
print(f'Características más importantes: {[features[i] for i in significant_features]}')
print(f'Valor de lambda: {lambda_val}')
