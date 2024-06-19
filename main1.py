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

# Crear variables para los pesos y la constante
w = model.addVars(n_features, name="w")
b = model.addVar(name="b")

# Definir la función objetivo
obj = gp.quicksum((gp.quicksum(w[i] * X[r, i] for i in range(n_features)) + b - Y[r]) ** 2 for r in range(n_samples))
model.setObjective(obj, GRB.MINIMIZE)

# Optimizar el modelo
model.optimize()

# Imprimir los resultados
features = data.columns[:-1]
for i in range(n_features):
    print(f'{features[i]}: {w[i].X}')
print(f'b: {b.X}')
