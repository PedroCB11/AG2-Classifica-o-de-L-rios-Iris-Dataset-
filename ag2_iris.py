import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("iris.csv")


mapa = {
    "Iris-setosa": 1,
    "Iris-versicolor": 2,
    "Iris-virginica": 3
}
df["species"] = df["species"].replace(mapa).astype("int64")


X = df[["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]]
y = df["species"]

#  treino e teste (60% / 40%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, shuffle=True, random_state=42
)

# modelo escolhido 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("=== AVALIAÇÃO DO MODELO ===")
print(f"Acurácia: {acc:.2f}")
print("\nRelatório de Classificacao:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

#  Permitir entrada de novos dados pelo usuário
def prever():
    print("\n=== Previsão Interativa ===")
    sl = float(input("Sepal length (cm): "))
    sw = float(input("Sepal width (cm): "))
    pl = float(input("Petal length (cm): "))
    pw = float(input("Petal width (cm): "))

    pred = model.predict([[sl, sw, pl, pw]])[0]
    nomes = {1: "Iris-setosa", 2: "Iris-versicolor", 3: "Iris-virginica"}
    print(f"\ A flor prevista é: {nomes[pred]}")

# Matriz de confusao (grafico 1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Matriz de Confusão")
plt.colorbar()

classes = ["Setosa", "Versicolor", "Virginica"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Colocar os valores dentro dos quadrados
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")

plt.ylabel("Real")
plt.xlabel("Previsto")
plt.tight_layout()
plt.show()

#grafico 2 (dsitribuicao de especies)

plt.figure(figsize=(6, 5))
df["species"].value_counts().sort_index().plot(kind="bar", color=["red","green","blue"])

plt.title("Distribuição das Espécies")
plt.xticks([0,1,2], ["Setosa", "Versicolor", "Virginica"], rotation=0)
plt.ylabel("Quantidade")
plt.xlabel("Espécie")
plt.tight_layout()
plt.show()


#grafico 3 (ESPAÇO 3D DAS FEATURES COLORIDO)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

x = df["sepal_length_cm"]
y2 = df["sepal_width_cm"]
z = df["petal_length_cm"]
cores = df["species"]

ax.scatter(x, y2, z, c=cores, cmap="viridis", s=50)

ax.set_title("Visualização 3D das Amostras (Iris Dataset)")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")

plt.show()