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
model = gp.Model("linear_regression_l1")

# Parámetro lambda (ajustable)
lambda_value = 0.1

# Variables de decisión
w = model.addVars(n_features, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w")
b = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b")
z = model.addVars(n_features, lb=0, ub=GRB.INFINITY, name="z")
error = model.addVars(n_samples, lb=0, ub=GRB.INFINITY, name="error")

# Función objetivo: minimizar el error cuadrático con penalización L1
model.setObjective(gp.quicksum(error[r] * error[r] for r in range(n_samples)) + lambda_value * gp.quicksum(z[i] for i in range(n_features)), GRB.MINIMIZE)

# Restricciones: calcular el error para cada muestra
for r in range(n_samples):
    model.addConstr(error[r] == b + gp.quicksum(w[i] * X[r, i] for i in range(n_features)) - Y[r])

# Restricciones para la penalización L1
for i in range(n_features):
    model.addConstr(w[i] <= z[i])
    model.addConstr(-w[i] <= z[i])

# Establecer parámetros para mejorar la estabilidad numérica
model.setParam('NumericFocus', 3)

# Optimizar el modelo
model.optimize()

# Verificar y ajustar el valor de lambda
while True:
    # Optimizar el modelo
    model.optimize()

    # Verificar el estado del modelo
    if model.status != GRB.OPTIMAL:
        print("El modelo no se resolvió de manera óptima.")
        break

    # Contar cuántos pesos son distintos de cero
    non_zero_weights = [i for i in range(n_features) if abs(w[i].X) > 1e-6]

    if len(non_zero_weights) == 5:
        print(f"λ: {lambda_value}")
        print("Características más importantes:")
        for i in non_zero_weights:
            print(f"w{i+1} ({data.columns[i]}): {w[i].X}")
        break
    else:
        # Incrementar λ y reiniciar el modelo
        lambda_value *= 2
        model.setObjective(gp.quicksum(error[r] * error[r] for r in range(n_samples)) + lambda_value * gp.quicksum(z[i] for i in range(n_features)), GRB.MINIMIZE)

print("Modelo optimizado con 5 características más importantes.")
