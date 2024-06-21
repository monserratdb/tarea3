import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Leer datos del archivo CSV
data = pd.read_csv('muestras.csv')
X = data.iloc[:, :-1].values  # Características
Y = data.iloc[:, -1].values   # Valores de relleno

# Número de muestras y características
n_samples, n_features = X.shape

# Crear el modelo
model = gp.Model("linear_regression")

# Variables de decisión
w = model.addVars(n_features, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w")
b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b")

# Error cuadrático
error = model.addVars(n_samples, lb=0, ub=GRB.INFINITY, name="error")

# Función objetivo: minimizar el error cuadrático
model.setObjective(gp.quicksum(error[r] * error[r] for r in range(n_samples)), GRB.MINIMIZE)

# Restricciones: calcular el error para cada muestra
for r in range(n_samples):
    model.addConstr(error[r] == b + gp.quicksum(w[i] * X[r, i] for i in range(n_features)) - Y[r])

# Optimizar el modelo
model.optimize()

# Imprimir los resultados
for i in range(n_features):
    print(f"Peso w{i+1} ({data.columns[i]}): {w[i].X}")
print(f"Término b: {b.X}")
