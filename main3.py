import gurobipy as gp
import pandas as pd
import numpy as np

# Cargar datos
data = pd.read_csv('muestras.csv')  # Reemplaza con el nombre correcto del archivo
X = data.iloc[:, :-1].values  # Todas las columnas menos la última
y = data.iloc[:, -1].values  # La última columna

# Características importantes obtenidas de main2.py
important_features = [0, 1, 2, 10, 12]  

# Filtrar X para las características importantes
X_important = X[:, important_features]

# Definir el modelo
model = gp.Model()

# Crear variables
w = model.addVars(len(important_features), lb=-gp.GRB.INFINITY, name="w")
b = model.addVar(lb=-gp.GRB.INFINITY, name="b")

# Crear el objetivo
objective = gp.quicksum((b + gp.quicksum(w[j] * X_important[r][j] for j in range(len(important_features))) - y[r])**2 for r in range(X.shape[0]))
model.setObjective(objective, gp.GRB.MINIMIZE)

# Optimizar
model.optimize()

# Imprimir resultados
print(f"Variable b: {b.X}")
optimal_value = model.ObjVal
print(f"Valor óptimo: {optimal_value:.3f}")
print("Modelo final de regresión:")
model_str = "Y = "
for j in range(len(important_features)):
    model_str += f"{w[j].X} * {data.columns[important_features[j]]} + "
model_str += f"{b.X}"
print(model_str)
print("Y estaría en gramos")
