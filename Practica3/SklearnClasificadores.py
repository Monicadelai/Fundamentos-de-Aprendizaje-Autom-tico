# Alfonso Bonilla Trueba
# Monica de la Igleia Martinez
# Pareja 13 Grupo 1461

from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.pipeline import Pipeline

################################################################################
def PorcentajeFallo(real, prediccion):
    fallos = 0
    for i in range(0, len(real)):
        if real[i] !=  prediccion[i]:
            fallos += 1
    return fallos/ float(len(real))

################################################################################
def NaiveBayesSimpleSklearn(laPlace, dataset, porcentaje):

    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - porcentaje))

    if laPlace:
        clasificador_skl = MultinomialNB(alpha=1)
    else:
        clasificador_skl = GaussianNB()

    clasificador_skl.fit(X_train, y_train)
    preds = clasificador_skl.predict(X_test)

    return PorcentajeFallo(y_test, preds)

################################################################################
def NaiveBayesCruzadaSklearn(laPlace, dataset, particiones):

    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]
    kf = KFold(n_splits=particiones, random_state=None, shuffle=True)
    kf.get_n_splits(X)

    preds_aux = []
    y_test_aux = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        if laPlace:
            clasificador_skl = MultinomialNB(alpha=1)
        else:
            clasificador_skl = GaussianNB()

        clasificador_skl.fit(X_train, y_train)

        preds = clasificador_skl.predict(X_test)
        preds_aux.append(preds)
        y_test_aux.append(y_test)

    errores = []
    for i in range(0, len(preds_aux)):
        errores.append(PorcentajeFallo(y_test_aux[i], preds_aux[i]))
    return errores

################################################################################
def KnnSimpleSklearn(dataset, porcentaje, k_vecinos):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - porcentaje))

    clasificador_skl = KNeighborsClassifier(n_neighbors=k_vecinos)

    clasificador_skl.fit(X_train, y_train)
    preds = clasificador_skl.predict(X_test)

    return PorcentajeFallo(y_test, preds)

################################################################################
def KnnCruzadaSklearn(dataset, particiones, k_vecinos):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    kf = KFold(n_splits=particiones, random_state=None, shuffle=True)
    kf.get_n_splits(X)

    preds_aux = []
    y_test_aux = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clasificador_skl = KNeighborsClassifier(n_neighbors=k_vecinos)

        clasificador_skl.fit(X_train, y_train)

        preds = clasificador_skl.predict(X_test)
        preds_aux.append(preds)
        y_test_aux.append(y_test)

    errores = []
    for i in range(0, len(preds_aux)):
        errores.append(PorcentajeFallo(y_test_aux[i], preds_aux[i]))
    return errores

################################################################################
def RegresionLogisticaSimpleSklearn(dataset, porcentaje, epocas):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - porcentaje))

    clasificador_skl = LogisticRegression(max_iter=epocas)

    clasificador_skl.fit(X_train, y_train)
    preds = clasificador_skl.predict(X_test)

    return PorcentajeFallo(y_test, preds)

################################################################################
def RegresionLogisticaCruzadaSklearn(dataset, particiones, epocas):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    kf = KFold(n_splits=particiones, random_state=None, shuffle=True)
    kf.get_n_splits(X)

    preds_aux = []
    y_test_aux = []
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clasificador_skl = LogisticRegression(max_iter=epocas)

        clasificador_skl.fit(X_train, y_train)

        preds = clasificador_skl.predict(X_test)
        preds_aux.append(preds)
        y_test_aux.append(y_test)

    errores = []
    for i in range(0, len(preds_aux)):
        errores.append(PorcentajeFallo(y_test_aux[i], preds_aux[i]))
    return errores

################################################################################
def PreprocesamientoAGSklearnKBest(dataset, porcentaje):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - porcentaje))

    selector = SelectKBest(f_regression)
    selector.fit_transform(X_train, y_train)
    return selector.get_support()

################################################################################
def PreprocesamientoAGSklearnFromModel(dataset, porcentaje):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - porcentaje))

    clasificador_skl = LogisticRegression(max_iter=4).fit(X_train, y_train)

    selector = SelectFromModel(clasificador_skl, prefit=True)
    return selector.get_support()
