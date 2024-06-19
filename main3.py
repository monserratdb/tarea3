import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Cargar datos
data = pd.read_csv('muestras.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# Filtrar las cinco características más importantes
significant_features = [0, 1, 2, 3, 4]  # Reemplazar con índices obtenidos en la Parte 2
X_selected = X[:, significant_features]

# Crear el modelo
model = gp.Model()

# Crear variables para los pesos y la constante
w = model.addVars(len(significant_features), name="w")
b = model.addVar(name="b")

# Definir la función objetivo
obj = gp.quicksum((gp.quicksum(w[i] * X_selected[r, i] for i in range(len(significant_features))) + b - Y[r]) ** 2 for r in range(len(Y)))
model.setObjective(obj, GRB.MINIMIZE)

# Optimizar el modelo
model.optimize()

# Imprimir los resultados
features = data.columns[significant_features]
model_string = "Y = " + " + ".join([f"{w[i].X} * {features[i]}" for i in range(len(significant_features))]) + f" + {b.X}"
print(model_string)
