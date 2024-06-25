import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# Cargar los datos del archivo CSV
data = pd.read_csv('muestras.csv')

# Extraer las variables X (inputs) y Y (output)
X = data.iloc[:, :-1].values  # Las primeras 15 columnas
Y = data.iloc[:, -1].values  # La última columna

# Número de muestras y características
num_samples, num_features = X.shape

# Crear el modelo
model = gp.Model("regresion_lineal_l1")

# Variables
w = model.addVars(num_features, lb=-GRB.INFINITY, name="w")
z = model.addVars(num_features, lb=0, name="z")
b = model.addVar(lb=-GRB.INFINITY, name="b")

# Parámetro lambda
lambda_value = 1  # Ajuste inicial, ajustar según necesidad

# Función objetivo
objective = gp.QuadExpr()
for r in range(num_samples):
    error = b + sum(w[i] * X[r, i] for i in range(num_features)) - Y[r]
    objective.add(error * error)
objective.add(lambda_value * sum(z[i] for i in range(num_features)))
model.setObjective(objective, GRB.MINIMIZE)

# Restricciones
for i in range(num_features):
    model.addConstr(w[i] <= z[i])
    model.addConstr(w[i] >= -z[i])

# Optimizar el modelo y ajustar lambda
while True:
    model.optimize()
    nonzero_weights = [i for i in range(num_features) if abs(w[i].X) > 1e-6]
    if len(nonzero_weights) <= 5:
        break
    lambda_value *= 2
    objective = gp.QuadExpr()
    for r in range(num_samples):
        error = b + sum(w[i] * X[r, i] for i in range(num_features)) - Y[r]
        objective.add(error * error)
    objective.add(lambda_value * sum(z[i] for i in range(num_features)))
    model.setObjective(objective, GRB.MINIMIZE)

# Imprimir los resultados
important_features = [i for i in range(num_features) if abs(w[i].X) > 1e-6]
print(f'Características más importantes: {important_features}')
print(f'Valor de lambda: {lambda_value}')
for i in important_features:
    print(f'Peso w{i+1} ({data.columns[i]}): {w[i].X}')
print(f'Término constante b: {b.X}')
