
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(page_title="Predicción de Vasopresores", layout="centered")

st.title("🩺 Predicción de Uso de Vasopresores")
st.write("Esta aplicación entrena un modelo de **Random Forest** para predecir la necesidad de vasopresores.")

# ---------------------------
# Cargar datos
# ---------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset.csv")  # Debe estar en el mismo repositorio
    return df

df = cargar_datos()

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# ---------------------------
# Preparar datos
# ---------------------------
X = df.drop("VASOPRESORES", axis=1)
y = df["VASOPRESORES"].map({"NO": 0, "SI": 1})  # Convertir a binario

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Entrenar modelo
# ---------------------------
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# ---------------------------
# Evaluación del modelo
# ---------------------------
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

precision = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
matriz_conf = confusion_matrix(y_test, y_pred)

st.subheader("📊 Resultados del modelo")
st.write(f"**Precisión del modelo:** {precision:.2f}")

# Curva ROC
st.write("**Curva ROC**")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Curva ROC")
ax.legend()
st.pyplot(fig)

# Matriz de confusión
st.write("**Matriz de Confusión**")
fig2, ax2 = plt.subplots()
sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("Predicción")
ax2.set_ylabel("Real")
st.pyplot(fig2)

# ---------------------------
# Formulario de predicción manual
# ---------------------------
st.subheader("📝 Ingresar valores para predicción")

valores = {}
for col in X.columns:
    valores[col] = st.number_input(f"{col}", value=float(X[col].mean()))

if st.button("Predecir"):
    entrada = np.array(list(valores.values())).reshape(1, -1)
    prob = modelo.predict_proba(entrada)[0][1]
    st.write(f"**Probabilidad de necesitar vasopresores:** {prob*100:.2f}%")
