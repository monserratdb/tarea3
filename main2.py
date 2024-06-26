import gurobipy as gp
import pandas as pd
import numpy as np

# CORRER EL CODIGO EN LA MISMA CARPETA EN QUE SE TIENE muestras.csv

# Cargar datos
data = pd.read_csv('muestras.csv')  
X = data.iloc[:, :-1].values  # todas las columnas menos la última
y = data.iloc[:, -1].values  # la última columna

# valor de lambda obtenido de forma manual
lambda_val = 896 

# crear modelo
model = gp.Model()

# crear variables
w = model.addVars(X.shape[1], lb=-gp.GRB.INFINITY, name="w")
b = model.addVar(lb=-gp.GRB.INFINITY, name="b")
z = model.addVars(X.shape[1], lb=0, name="z")

# crear el objetivo
objective = gp.quicksum((b + gp.quicksum(w[i] * X[r][i] for i in range(X.shape[1])) - y[r])**2 for r in range(X.shape[0])) + lambda_val * gp.quicksum(z[i] for i in range(X.shape[1]))
model.setObjective(objective, gp.GRB.MINIMIZE)

# agregar restricciones
model.addConstrs((w[i] <= z[i] for i in range(X.shape[1])), name="c1")
model.addConstrs((w[i] >= -z[i] for i in range(X.shape[1])), name="c2")

# optimizar
model.optimize()

# encontrar las características más importantes
important_features = [i for i in range(X.shape[1]) if abs(w[i].X) > 1e-6]
important_features.sort(key=lambda i: abs(w[i].X), reverse=True)
important_features = important_features[:5]

# print
print(f"Lambda: {lambda_val}")
print(f"Conjunto E: {important_features}")
print("Características más importantes:")
for i in important_features:
    print(f"{data.columns[i]} (w_{i+1}): {w[i].X}, (z_{i+1}): {z[i].X}")

print(f"Constante (b): {b.X} gramos")

# print valor óptimo
optimal_value = model.ObjVal
print(f"Valor óptimo: {optimal_value:.3f}")
