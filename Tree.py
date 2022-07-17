from sklearn import tree
import graphviz 
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def algoritmo_arvore(variaveis):
    data = pd.read_csv("voice.csv")
    data.dropna(inplace=True)
    df_target = data["label"]
    df_data = data.drop(columns=["label"])

    X = df_data
    Y = df_target
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    previsao = clf.predict([variaveis])
    st.markdown("### Previsão e métricas do algoritmo: ")
    st.text("Previsão para os valores fornecidos: " + str(previsao))

    X_train, X_test, y_train, y_test = train_test_split(df_data, 
                                                        df_target,
                                                        random_state=0)

    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2) 
    decision_tree.fit(X_train, y_train)
    y_model = decision_tree.predict(X_test)
    y_test = np.array(y_test)
    precisao = precision_score(y_test, y_model, average='micro')
    st.text("Precisão: " + str(round(precisao,3)))



    X_train, X_test, y_train, y_test = train_test_split(df_data, 
                                                        df_target,
                                                        random_state=0)

    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2) 
    decision_tree.fit(X_train, y_train)
    y_model = decision_tree.predict(X_test)
    y_test = np.array(y_test)
    acuracia = accuracy_score(y_test, y_model)
    st.text("Acurácia: " + str(round(acuracia,3)))



    X_train, X_test, y_train, y_test = train_test_split(df_data, 
                                                        df_target,
                                                        random_state=0)

    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2) 
    decision_tree.fit(X_train, y_train)
    y_model = decision_tree.predict(X_test)
    y_test = np.array(y_test)

    #mat = confusion_matrix(y_test, y_pred)
    mat = confusion_matrix(y_test, y_model)
    print(mat)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('Valor verdadeiro')
    plt.ylabel('Valor previsto')

    st.text("Matriz de confusão: ")
    st.pyplot(plt.show())
