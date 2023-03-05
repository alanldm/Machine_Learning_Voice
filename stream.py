import streamlit as st
import pandas as pd
import random
import KNN
import Tree

st.title('Unidade 3 - Machine Learning')

st.markdown('* Alan Lima de Medeiros')

option = st.sidebar.selectbox('Escolha a técnica de Aprendizagem de Máquina: ', ('KNN', 'Árvore de Decisão'))

data = pd.read_csv('voice.csv')

if(option=='KNN'):

    st.markdown('### Algoritmo KNN')

    vizinhos = st.radio('Escolha a quantidade de vizinhos: ', (3, 5, 7), horizontal=True)
    tipo = st.selectbox('Escolha a forma dos dados para se realizar uma predição: ', ('Gerar valores aleatórios', 'Preencher manualmente'))
    if tipo == 'Gerar valores aleatórios':
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            meanfreq = data.loc[random.randrange(0, 3167), 'meanfreq']
            st.write("Meanfreq:")
            st.write(meanfreq)
            sd = data.loc[random.randrange(0, 3167), 'sd']
            st.write("Sd:")
            st.write(sd)
            median = data.loc[random.randrange(0, 3167), 'median']
            st.write("Median:")
            st.write(median)
            Q25 = data.loc[random.randrange(0, 3167), 'Q25']
            st.write("Q25:")
            st.write(Q25)
            Q75 = data.loc[random.randrange(0, 3167), 'Q75']
            st.write("Q75:")
            st.write(Q75)
        with col2:
            IQR = data.loc[random.randrange(0, 3167), 'IQR']
            st.write("IQR:")
            st.write(IQR)
            skew = data.loc[random.randrange(0, 3167), 'skew']
            st.write("Skew:")
            st.write(skew)
            kurt = data.loc[random.randrange(0, 3167), 'kurt']
            st.write("Kurt:")
            st.write(kurt)
            spent = data.loc[random.randrange(0, 3167), 'sp.ent']
            st.write("Sp.ent:")
            st.write(spent)
            sfm = data.loc[random.randrange(0, 3167), 'sfm']
            st.write("Sfm:")
            st.write(sfm)
        with col3:
            mode = data.loc[random.randrange(0, 3167), 'mode']
            st.write("Mode:")
            st.write(mode)
            centroid = data.loc[random.randrange(0, 3167), 'centroid']
            st.write("Centroid:")
            st.write(centroid)
            meanfun = data.loc[random.randrange(0, 3167), 'meanfun']
            st.write("Meanfun:")
            st.write(meanfun)
            minfun = data.loc[random.randrange(0, 3167), 'minfun']
            st.write("Minfun:")
            st.write(minfun)
            maxfun = data.loc[random.randrange(0, 3167), 'maxfun']
            st.write("Maxfun:")
            st.write(maxfun)
        with col4:
            meandom = data.loc[random.randrange(0, 3167), 'meandom']
            st.write("Meandom:")
            st.write(meandom)
            mindom = data.loc[random.randrange(0, 3167), 'mindom']
            st.write("Mindom:")
            st.write(mindom)
            maxdom = data.loc[random.randrange(0, 3167), 'maxdom']
            st.write("Maxdom:")
            st.write(maxdom)
            dfrange = data.loc[random.randrange(0, 3167), 'dfrange']
            st.write("Dfrange:")
            st.write(dfrange)
            modindx = data.loc[random.randrange(0, 3167), 'modindx']
            st.write("Modindx:")
            st.write(modindx)
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            meanfreq = st.number_input('meanfreq: ')
            sd = st.number_input('sd: ')
            median = st.number_input('median: ')
            Q25 = st.number_input('Q25: ')
            Q75 = st.number_input('Q75: ')
        with col2:
            IQR = st.number_input('IQR: ')
            skew = st.number_input('skew: ')
            kurt = st.number_input('kurt: ')
            spent = st.number_input('sp.ent: ')
            sfm = st.number_input('sfm: ')
        with col3:
            mode = st.number_input('mode: ')
            centroid = st.number_input('centroid: ')
            meanfun = st.number_input('meanfun: ')
            minfun = st.number_input('minfun: ')
            maxfun = st.number_input('maxfun: ')
        with col4:
            meandom = st.number_input('meandom: ')
            mindom = st.number_input('mindom: ')
            maxdom = st.number_input('maxdom: ')
            dfrange = st.number_input('dfrange: ')
            modindx = st.number_input('modindx: ')

    variaveis = [meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, spent, sfm, mode, centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx]
    prever = st.button('Prever!')
    if prever==True:
        KNN.algoritmo_knn(variaveis, vizinhos)

else:
    st.markdown('### Algoritmo árvore de decisão')

    tipo = st.selectbox('Escolha a forma dos dados para se realizar uma predição: ', ('Gerar valores aleatórios', 'Preencher manualmente'))
    if tipo == 'Gerar valores aleatórios':
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            meanfreq = data.loc[random.randrange(0, 3167), 'meanfreq']
            st.write("Meanfreq:")
            st.write(meanfreq)
            sd = data.loc[random.randrange(0, 3167), 'sd']
            st.write("Sd:")
            st.write(sd)
            median = data.loc[random.randrange(0, 3167), 'median']
            st.write("Median:")
            st.write(median)
            Q25 = data.loc[random.randrange(0, 3167), 'Q25']
            st.write("Q25:")
            st.write(Q25)
            Q75 = data.loc[random.randrange(0, 3167), 'Q75']
            st.write("Q75:")
            st.write(Q75)
        with col2:
            IQR = data.loc[random.randrange(0, 3167), 'IQR']
            st.write("IQR:")
            st.write(IQR)
            skew = data.loc[random.randrange(0, 3167), 'skew']
            st.write("Skew:")
            st.write(skew)
            kurt = data.loc[random.randrange(0, 3167), 'kurt']
            st.write("Kurt:")
            st.write(kurt)
            spent = data.loc[random.randrange(0, 3167), 'sp.ent']
            st.write("Sp.ent:")
            st.write(spent)
            sfm = data.loc[random.randrange(0, 3167), 'sfm']
            st.write("Sfm:")
            st.write(sfm)
        with col3:
            mode = data.loc[random.randrange(0, 3167), 'mode']
            st.write("Mode:")
            st.write(mode)
            centroid = data.loc[random.randrange(0, 3167), 'centroid']
            st.write("Centroid:")
            st.write(centroid)
            meanfun = data.loc[random.randrange(0, 3167), 'meanfun']
            st.write("Meanfun:")
            st.write(meanfun)
            minfun = data.loc[random.randrange(0, 3167), 'minfun']
            st.write("Minfun:")
            st.write(minfun)
            maxfun = data.loc[random.randrange(0, 3167), 'maxfun']
            st.write("Maxfun:")
            st.write(maxfun)
        with col4:
            meandom = data.loc[random.randrange(0, 3167), 'meandom']
            st.write("Meandom:")
            st.write(meandom)
            mindom = data.loc[random.randrange(0, 3167), 'mindom']
            st.write("Mindom:")
            st.write(mindom)
            maxdom = data.loc[random.randrange(0, 3167), 'maxdom']
            st.write("Maxdom:")
            st.write(maxdom)
            dfrange = data.loc[random.randrange(0, 3167), 'dfrange']
            st.write("Dfrange:")
            st.write(dfrange)
            modindx = data.loc[random.randrange(0, 3167), 'modindx']
            st.write("Modindx:")
            st.write(modindx)
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            meanfreq = st.number_input('meanfreq: ')
            sd = st.number_input('sd: ')
            median = st.number_input('median: ')
            Q25 = st.number_input('Q25: ')
            Q75 = st.number_input('Q75: ')
        with col2:
            IQR = st.number_input('IQR: ')
            skew = st.number_input('skew: ')
            kurt = st.number_input('kurt: ')
            spent = st.number_input('sp.ent: ')
            sfm = st.number_input('sfm: ')
        with col3:
            mode = st.number_input('mode: ')
            centroid = st.number_input('centroid: ')
            meanfun = st.number_input('meanfun: ')
            minfun = st.number_input('minfun: ')
            maxfun = st.number_input('maxfun: ')
        with col4:
            meandom = st.number_input('meandom: ')
            mindom = st.number_input('mindom: ')
            maxdom = st.number_input('maxdom: ')
            dfrange = st.number_input('dfrange: ')
            modindx = st.number_input('modindx: ')

    variaveis = [meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, spent, sfm, mode, centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx]
    prever = st.button('Prever!')
    if prever==True:
        Tree.algoritmo_arvore(variaveis)
