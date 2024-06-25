import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Cargar los datos del archivo CSV
data = pd.read_csv('muestras.csv')

# Extraer las variables X (inputs) y Y (output)
X = data.iloc[:, :-1].values  # Las primeras 15 columnas
Y = data.iloc[:, -1].values  # La última columna

# Importar las características importantes de la Parte 2
important_features = [0, 1, 2, 3, 4]  # Ejemplo, ajuste según resultados de main2.py

# Crear el modelo
model = gp.Model("regresion_lineal_reducida")

# Variables
w = model.addVars(len(important_features), lb=-GRB.INFINITY, name="w")
b = model.addVar(lb=-GRB.INFINITY, name="b")

# Función objetivo
objective = gp.QuadExpr()
for r in range(len(X)):
    error = b + sum(w[j] * X[r, i] for j, i in enumerate(important_features)) - Y[r]
    objective.add(error * error)
model.setObjective(objective, GRB.MINIMIZE)

# Optimizar el modelo
model.optimize()

# Imprimir el modelo final
model_string = "Y = " + " + ".join([f"{w[j].X:.2f} * x{important_features[j]+1}" for j in range(len(important_features))]) + f" + {b.X:.2f}"
print(f'Modelo final: {model_string}')
for j, i in enumerate(important_features):
    print(f'x{i+1} ({data.columns[i]}): {w[j].X}')
print(f'Término constante b: {b.X}')
