import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Leer datos del archivo CSV
data = pd.read_csv('muestras.csv')
X = data.iloc[:, :-1].values  # Características
Y = data.iloc[:, -1].values   # Valores de relleno

# Características seleccionadas en la Parte 2
selected_features = [1, 3, 5, 7, 9]  # Ejemplo de índices de las características seleccionadas

# Crear el modelo
model = gp.Model("linear_regression_final")

# Variables de decisión
w = model.addVars(len(selected_features), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w")
b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b")
error = model.addVars(len(X), lb=0, ub=GRB.INFINITY, name="error")

# Función objetivo: minimizar el error cuadrático
model.setObjective(gp.quicksum(error[r] * error[r] for r in range(len(X))), GRB.MINIMIZE)

# Restricciones: calcular el error para cada muestra
for r in range(len(X)):
    model.addConstr(error[r] == b + gp.quicksum(w[i] * X[r, selected_features[i]] for i in range(len(selected_features))) - Y[r])

# Optimizar el modelo
model.optimize()

# Imprimir los resultados
print("Modelo final de regresión lineal:")
print("Y =", " + ".join(f"{w[i].X} * x{selected_features[i]+1}" for i in range(len(selected_features))), f"+ {b.X}")

for i in range(len(selected_features)):
    print(f"x{selected_features[i]+1}: {data.columns[selected_features[i]]}")
print(f"Y: Relleno óptimo de la almohada (gramos)")

