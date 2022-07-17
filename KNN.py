import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

def algoritmo_knn(variaveis, vizinhos):
    data = pd.read_csv("voice.csv")
    data.dropna(inplace=True)

    data.columns

    df_target = data["label"]
    df_data = data.drop(columns=["label"])
    df_data

    print("Dimensão das features: " + str(df_data.shape))
    print("Dimensão do target: " + str(df_target.shape))

    X_train, X_test, y_train, y_test = train_test_split(df_data, 
                                                        df_target,
                                                        random_state=0)

    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_test shape: {}".format(y_test.shape))

    knn = KNeighborsClassifier(n_neighbors=vizinhos)

    knn.fit(X_train, y_train)

    teste1 = np.array([variaveis])
    prediction = knn.predict(teste1)
    print("Prediction: {}".format(prediction))
    st.markdown("### Previsão e métricas do algoritmo: ")
    st.text("Previsão para os valores fornecidos: " + str(prediction))

    y_pred = knn.predict(X_test)
    print("Test set predictions:\n {}".format(y_pred))

    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    y_model = model.predict(X_test)
    accuracy_score(y_test, y_model)

    mat = confusion_matrix(y_test, y_pred)
    print(mat)

    fig = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('Valor verdadeiro')
    plt.ylabel('Valor previsto')

    st.text("Precisão: " + str(round(knn.score(X_test, y_test),2)))
    st.text("Acurácia: " + str(round(accuracy_score(y_test, y_model),2)))
    st.text("Matriz de confusão: ")
    st.pyplot(plt.show())