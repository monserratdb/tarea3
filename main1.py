import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# CORRER EL CODIGO EN LA MISMA CARPETA EN QUE SE TIENE muestras.csv
# Cargar datos
data = pd.read_csv('muestras.csv')
X = data.iloc[:, :-1].values  # características
Y = data.iloc[:, -1].values  # variable obj

# crear el modelo
model = gp.Model()

# crear variables para los pesos y la constante, se pone -GRB.INFINITY porque pueden ser negativos los pesos
w = model.addVars(len(X[0]), lb=-GRB.INFINITY, name="w")
b = model.addVar(lb=-GRB.INFINITY, name="b")

# crear la f.o
obj = sum((b + gp.quicksum(w[i] * X[r][i] for i in range(len(X[0]))) - Y[r]) ** 2 for r in range(len(Y)))
model.setObjective(obj, GRB.MINIMIZE)

# optimizar
model.optimize()

# verificar si la solución es óptima
if model.status == GRB.OPTIMAL:
    # print resultados
    for i in range(len(X[0])):
        print(f"w_{i+1} (característica {data.columns[i]}): {w[i].X}")
    print(f"b (constante): {b.X} gramos")
else:
    print("No se encontró una solución óptima")

optimal_value = model.ObjVal
print(f"Valor óptimo: {optimal_value:.3f}")
