import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Cargar datos
data = pd.read_csv('muestras.csv')
X = data.iloc[:, :-1].values  # Características
Y = data.iloc[:, -1].values  # Variable objetivo

# Crear el modelo
model = gp.Model()

# Crear variables para los pesos y la constante, con restricciones no negativas si es necesario
w = model.addVars(len(X[0]), lb=-GRB.INFINITY, name="w")
b = model.addVar(lb=-GRB.INFINITY, name="b")

# Crear la función objetivo
obj = sum((b + gp.quicksum(w[i] * X[r][i] for i in range(len(X[0]))) - Y[r]) ** 2 for r in range(len(Y)))
model.setObjective(obj, GRB.MINIMIZE)

# Optimizar el modelo
model.optimize()

# Verificar si la solución es óptima
if model.status == GRB.OPTIMAL:
    # Imprimir los resultados
    for i in range(len(X[0])):
        print(f"w_{i+1} (característica {data.columns[i]}): {w[i].X}")
    print(f"b (constante): {b.X}")
else:
    print("No se encontró una solución óptima")
