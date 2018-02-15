# Alfonso Bonilla Trueba
# Monica de la Igleia Martinez
# Pareja 13 Grupo 1461

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from Clasificador import Clasificador
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import KFold
from argparse import ArgumentParser
import EstrategiaParticionado as EstrategiaParticionado
import Clasificador as Clasificador
import numpy as np
import sys

parser = ArgumentParser(
    version='1.0',
    description="Comparador de errores de Naive Bayes con Sklearn.\n Por defecto, numero de particiones = 5, porcentaje de entrenamiento = 0.7",
    epilog='Alfonso Bonilla Trueba y Monica de la Iglesia Martinez'
)

parser.add_argument("ruta", type=str, help="Ruta del fichero .data")
parser.add_argument("--laPlace", help="Usar correccion de La Place", action="store_true")
parser.add_argument("--porcentaje", type=float, default=0.7, help= "Porcentaje Train Validacion Simple")
parser.add_argument("--particiones", type=int, default=5, help= "Numero de particiones Validacion Cruzada")
parser.add_argument("--sklearn", help="Utiliza el clasificador de la libreria Sklearn", action="store_true")
args = parser.parse_args()

if args.porcentaje > 0.99:
    print "El porcentaje de entrenamiento no puede ser superior a 0.99"
    sys.exit(0)

try:
    del estrategia
    del clasificador
    del dataset
except NameError:
    pass

#####################################################################################################
dataset = Datos(args.ruta)
print "  Fichero: ", args.ruta.split('/')[-1]
print ("  - Porcentaje Entrenamiento en Validacion simple %.2f" % (args.porcentaje*100)) + "%"
print "  - Numero de Particiones en Validacion cruzada: ", args.particiones

if args.laPlace:
    print "  - Usando la Correccion de La Place"
else:
    print "  - Sin usar Correccion de La Place"

if args.sklearn:
    print "  - Usando la libreria Sklearn"
    print "----------------------------------------------------"
    #####################################################################################################
    # Comenzamos a usar SKLEARN
    #####################################################################################################
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]


    # ---------------------------- VALIDACION SIMPLE ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - args.porcentaje))
    # PARECE QUE ESTO YA HA RANDOMIZADO LAS FILAS SOLITO

    if args.laPlace:
        clasificador_skl = MultinomialNB(alpha=1)
    else:
        clasificador_skl = GaussianNB()

    clasificador_skl.fit(X_train, y_train)

    preds = clasificador_skl.predict(X_test)

    fallos = 0
    for i in range(0, len(y_test)):
        if y_test[i] !=  preds[i]:
            fallos += 1
    print("  El error con Validacion Simple es del %.2f" % round(fallos/ float(len(y_test))*100,2)) + "%"

    # ---------------------------- VALIDACION CRUZADA ----------------------------
    kf = KFold(n_splits=args.particiones, random_state=None, shuffle=True)
    kf.get_n_splits(X)

    preds_aux = []
    y_test_aux = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        if args.laPlace:
            clasificador_skl = MultinomialNB(alpha=1)
        else:
            clasificador_skl = GaussianNB()

        clasificador_skl.fit(X_train, y_train)

        preds = clasificador_skl.predict(X_test)
        preds_aux.append(preds)
        y_test_aux.append(y_test)


    porcentaje_aux = []
    for i in range(0, len(y_test_aux)):
        fallos=0
        porcentaje =0
        for j in range(0, len(y_test_aux[i])):
            if y_test_aux[i][j] !=  preds_aux[i][j]:
                fallos += 1
        porcentaje=round(fallos/ float(len(y_test_aux[i]))*100,2)
        porcentaje_aux.append(porcentaje)


    print("  El error con Validacion Cruzada es del %.2f" % round(np.mean(porcentaje_aux),2)) + "%" + (" Con una desviacion de %.2f" % round(np.std(porcentaje_aux),2)) + "%"

else:
    print "  - Usando nuestra libreria"
    print "----------------------------------------------------"
    estrategia = EstrategiaParticionado.ValidacionSimple()
    clasificador = Clasificador.ClasificadorNaiveBayes()
    errores = clasificador.validacion(estrategia, dataset, clasificador, laPlace = args.laPlace, porcentajeTrain=args.porcentaje)

    print("  El error con Validacion Simple es del %.2f" % round(errores[0]*100,2)) + "%"

    del estrategia
    del clasificador

    estrategia = EstrategiaParticionado.ValidacionCruzada()
    clasificador = Clasificador.ClasificadorNaiveBayes()
    errores = clasificador.validacion(estrategia, dataset, clasificador, laPlace = args.laPlace, numeroParts=args.particiones)

    print("  El error medio con Validacion Cruzada es del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Con una desviacion de %.2f" % round(np.std(errores)*100,2)) + "%"

    del estrategia
    del clasificador

del dataset
