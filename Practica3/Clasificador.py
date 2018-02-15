# Alfonso y Monica
# Pareja 13 Grupo 1461

from abc import ABCMeta,abstractmethod
from scipy.stats import norm
from random import uniform
import numpy as np
import math
import warnings
from math import sqrt


from EstrategiaParticionado import EstrategiaParticionado
import EstrategiaParticionado as EstrategiaParticionado

class Clasificador(object):

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
    # de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # devuelve un numpy array con las predicciones
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass


    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    def error(self,datos,pred):
        fallos = 0
        #Comprobacion de seguridad
        if(len(datos) != len(pred)):
            return -1
        #Contamos cuantas clases ha fallado, es decir, son distintas de la real
        for i in range(0, len(datos)):
            if datos[i] !=  pred[i]:
                fallos += 1
        return fallos/ float(len(datos))


    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self,particionado,dataset,clasificador,seed=None, porcentajeTrain = 0.7, numeroParts = 5):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test

        errores = []

        if isinstance(particionado, EstrategiaParticionado.ValidacionSimple):
            particionado.creaParticiones(dataset, porcentajeEntrenamiento = porcentajeTrain)
        else:
            particionado.creaParticiones(dataset, numeroParticiones= numeroParts)

        datos_train = np.array(())
        datos_test = np.array(())


        if particionado.nombreEstrategia == 'Simple':
            #Cogemos los datos train y test del objeto particion, elemento 0 porque es simple
            datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)

            datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)
            #Entrenamos con los datos
            clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)

            #Testeamos
            predicciones = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

            errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], predicciones))

        elif particionado.nombreEstrategia == 'Cruzada':
            for i in range(0, particionado.numeroParticiones):
                #Cogemos los datos train y test del objeto particion
                datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
                datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)

                #Entrenamos con los datos
                clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)
                predicciones = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

                #errores de la prediccion
                errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], predicciones))

        return errores


##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    laPlace = False
    numClases = 0
    array_clases = []


    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        numClases = len(diccionario[-1])

        for colum in range(0, len(atributosDiscretos)-1):                           # itera sobre los atributos de la matriz de entrenamiento
            if atributosDiscretos[colum]:                                           # si el atributo es nominal
                matriz_contador = np.zeros((len(diccionario[colum]) , numClases))   # Matriz de 0 que vamos a rellenar
                for clase in sorted(diccionario[-1].values()):                      # iteramos sobre cada clase posible
                    filas_clase = datostrain[:,-1] == clase                         # nos quedamos con la ultima columna, la de la clase
                    aux_matrix = np.array(())                                       # matriz auxiliar

                    for i in range(0, len(filas_clase)):                            # Del array de clases comprobamos si es true
                        if(filas_clase[i]):                                         # SI es true lo metemos en nuestra matriz aux
                            if(len(aux_matrix) == 0):
                                aux_matrix = datostrain[i]                          # Si esta vacia la aux igualamos
                            else:
                                aux_matrix = np.vstack([aux_matrix, datostrain[i]]) # Cuando contiene ya algo hacemos stack

                    if len(aux_matrix)==0:                                          # Si ninguna fila ha cumplido la condicion continuamos, no hay que contar nada
                        continue;

                    if(np.count_nonzero(filas_clase == True) != 1):                 # Caso en el que hay MAS DE UNA fila coincidente con la clase
                        u,v = np.unique(aux_matrix[:,colum], return_counts=True)    # Contamos cuantas veces aparece cada atributo
                        values = sorted(diccionario[colum].values())                # Ordenamos los valores del diccionario del atributo
                        for j in range(0,len(u)):                                   # Por cada atributo que aparece
                            indice = values.index(u[j])                             # Calculamos su indice
                            matriz_contador[indice][clase] = v[j]                   # Insertamos cuantas veces aparece
                    else:                                                           # Caso en el que solo hay UNA fila coincidente con la clase
                        values = sorted(diccionario[colum].values())
                        indice = values.index(aux_matrix[colum])                    # Calculamos su indice
                        matriz_contador[indice][clase] = 1                          # Como solo hay una fila es que solo aparece una vez

                ceros = np.where(matriz_contador == 0)                              # Contamos los ceros en la tabla
                if(self.laPlace == True and len(ceros[0]) != 0):                    # Si tenemos laplace activados
                    self.array_clases.append(np.add(matriz_contador, 1))            # Si hay algun cero sumamos uno a toda la tabla y la anadimos
                else:
                    self.array_clases.append(matriz_contador)                       # En otro caso anadimos la tabla como esta

            else:                                                                   # si el atributo es continuo
                # Si tengo que hace media y desviacion para cada clase
                matriz_contador = np.zeros((2 , len(diccionario[-1])))              # Matriz de 0 que vamos a rellenar
                for clase in range(0, len(diccionario[-1])):
                    sacado = datostrain[:,-1] == clase                              # Nos quedamos con la ultima columna, la de la clase
                    aux_matrix = np.array(())

                    for i in range(0, len(sacado)):                                 # Del array de clases comprobamos si es true
                        if(sacado[i]):                                              # SI es true lo metemos en nuestra matriz aux
                            if(len(aux_matrix) == 0):
                                aux_matrix = datostrain[i]
                            else:
                                aux_matrix = np.vstack([aux_matrix, datostrain[i]])

                    if len(aux_matrix)==0:                                          # Si ninguna fila ha cumplido la condicion continuamos, no hay que contar nada
                        continue;

                    if(np.count_nonzero(sacado == True) != 1):
                        media = np.array(aux_matrix[:,colum]).astype(np.float)
                        matriz_contador[0][clase] = np.mean(media)
                        matriz_contador[1][clase] = np.std(media)
                    else:
                        matriz_contador[0][clase] = aux_matrix[colum]
                        matriz_contador[1][clase] = 0

                self.array_clases.append(matriz_contador)


    def clasifica(self,datostest,atributosDiscretos,diccionario):
        array_prob_clases = []                                                      # Array que guarda la probabilidad de cada clase
        predicciones = []                                                           # Array que guarda las predicciones de cada fila de la matriz train

        aux = self.array_clases[0]
        for i in range(0, len(diccionario[-1])):
            suma = np.array(aux[:,i]).astype(np.float)                              # Cuenta cuantas veces se da cada clase
            array_prob_clases.append(np.sum(suma))                                  # Anade al array de probabilidades

        for i in range(0, len(datostest)):
            condicionadas = []                                                      # Guarda la formula naive para cada clase
            for clase in range(0, len(diccionario[-1])):
                verosimilitudes = []                                                # Array para la verosimilitud de cada atributo
                for atrib in range (0, len(atributosDiscretos)-1):                  # Itera sobre cada columna (atributos)
                    tablita = self.array_clases[atrib]                              # Accedemos a su tabla de ocurrencia del atributo
                    if atributosDiscretos[atrib]:                                   # Si el atributo ES NOMINAL
                        valores = sorted(diccionario[atrib].values())
                        indice = valores.index(datostest[i][atrib])
                        if tablita[indice][clase] == 0 or array_prob_clases[clase] == 0:
                            verosimilitudes.append(0)
                        else:
                            verosimilitudes.append(tablita[indice][clase]/array_prob_clases[clase])
                    else:                                                           # Si el atributo ES CONTINUO
                        if tablita[1][clase] == 0:
                            verosimilitudes.append(0)
                        else:
                            pdf_norm = norm.pdf(datostest[i][atrib], tablita[0][clase], tablita[1][clase])
                            verosimilitudes.append(pdf_norm)

                cacho = 1
                for elem in verosimilitudes:
                    cacho *= elem
                productorio = cacho * (array_prob_clases[clase]/np.sum(array_prob_clases)) # Naive porductorio
                condicionadas.append(productorio)

            maxVer = np.amax(condicionadas, axis=0)
            predicciones.append(condicionadas.index(maxVer))

        return predicciones                                                         # Retornamos las predicciones calculadas

    def __del__(self):
        del self.array_clases[:]
        self.laPlace = False


##############################################################################

class ClasificadorVecinosProximos(Clasificador):

    k_vecinos = 20
    normalizacion = True

    mediasTrain = []
    desviacionesTrain = []
    datosTrainNorm = np.array(())

    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        if self.normalizacion:
            self.calcularMediasDesv(datosTrain)
            self.normalizarDatos(datosTrain)
        else:
            self.datosTrainNorm = np.copy(datosTrain)

    def clasifica(self,datosTest,atributosDiscretos,diccionario):

        mediasTest = []
        desviacionesTest = []
        predicciones = []

        datostest_aux = np.copy(datosTest)
        if self.normalizacion:
            for i in range(0, datosTest.shape[1] - 1):
                mediasTest.append(np.mean(datosTest[:, i]))
                desviacionesTest.append(np.std(datosTest[:, i]))

            for i in range(0, datostest_aux.shape[0]):
                for j in range(0, datostest_aux.shape[1] - 1):
                    datostest_aux[i][j] = (datostest_aux[i][j] - mediasTest[j]) / desviacionesTest[j]


                                                                                                 # Ahora vamos a calcular las distancias
        for i in range(0, datostest_aux.shape[0]):                                               # Para cada fila de la matriz Test
            aux_distancias = np.array(())
            for j in range(0, self.datosTrainNorm.shape[0]):                                     # Iteramos con todas las filas de train
                distancias_PTP = []                                                              # Aqui vamos a almacenar distancias entre puntos
                for k in range(0, datostest_aux.shape[1] - 1):                                       # Iteramos sobre columnas de test
                    distancias_PTP.append(abs(datostest_aux[i][k] - self.datosTrainNorm[j][k]) ** 2) # Calculamos la distancia entre los puntos al cuadrado
                aux_distancias = np.append(aux_distancias, (sqrt(sum(distancias_PTP))) )

            orden_distancias = sorted(aux_distancias)                                            # Ordenamos las distancias de menor a mayor
            indices = []
            for dist in orden_distancias[:self.k_vecinos]:                                       # Almacenamos los indices de fila de los k menores
                indices.append(list(aux_distancias).index(dist))

            vecinos = self.datosTrainNorm[indices]                                               # Porcion de la matriz train,solo los k menores

            clase, num = np.unique(vecinos[:,-1], return_counts=True)                            # Contamos cuantas veces aparece cada clase en los k vecinos
            n_max = max(num)                                                                     # Nos quedamos con la clase predominante
            pred = int(clase[list(num).index(n_max)])
            predicciones.append(pred)                                                            # Guardamos la clase predominante en el array de predicciones

        return np.array(predicciones)



    def calcularMediasDesv(self,datostrain):
        for i in range(0, datostrain.shape[1] - 1):
            self.mediasTrain.append(np.mean(datostrain[:, i]))
            self.desviacionesTrain.append(np.std(datostrain[:, i]))


    def normalizarDatos(self,datos):
        self.datosTrainNorm = datos
        for i in range(0, datos.shape[0]):
            for j in range(0, datos.shape[1] - 1):
                self.datosTrainNorm[i][j] = (self.datosTrainNorm[i][j] - self.mediasTrain[j]) / self.desviacionesTrain[j]


    def __del__(self):
        del self.mediasTrain[:]
        del self.desviacionesTrain[:]
        del self.datosTrainNorm



##############################################################################

class ClasificadorRegresionLogistica(Clasificador):

    w_entrenada = np.array(())
    cte_aprendizaje = 1
    epocas = 4

    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        n_rows, n_cols = datosTrain.shape
        w = np.array(())

        warnings.filterwarnings("ignore")

        for i in range(0, n_cols):
            w = np.append(w, uniform(-1.0, 1.0))                                                    # Generamos W aleatorio entre -1 y 1

        for epoca in range(0, self.epocas):
            for row in range(0, n_rows):
                x = np.array(())
                x = np.append(x, 1)
                x = np.append(x, datosTrain[row][:-1])
                a = np.dot(np.transpose(w), x)                                                      # Multiplicacion de vectores
                # Calculamos sigma de a
                sigma = 1/(1+np.exp(-a))                                                            # Calculamos Sigma

                for k in range(0, n_cols):
                    w[k] = w[k] - self.cte_aprendizaje *(sigma - datosTrain[row, n_cols-1]) * x[k]  # Actualizamos el w

        #Guardamos la w de la ultima epoca
        self.w_entrenada = w


    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        n_rows, n_cols = datosTest.shape
        predicciones = []

        warnings.filterwarnings("ignore")

        for row in range(0, n_rows):
            x = np.array(())
            x = np.append(x, 1)
            x = np.append(x, datosTest[row][:-1])
            a = np.dot(np.transpose(self.w_entrenada), x)          # Multiplicacion de vectores
            sigma = 1/(1+np.exp(-a))                               # Calculamos Sigma

            if sigma < 0.5:
                predicciones.append(0)
            else:
                predicciones.append(1)

        return np.array(predicciones)

    def __del__(self):
        del self.w_entrenada
