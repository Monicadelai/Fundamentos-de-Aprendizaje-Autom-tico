# Alfonso y Monica
# Pareja 13 Grupo 1461

from abc import ABCMeta,abstractmethod
from numpy import random
from Datos import Datos
import numpy as np

class Particion():

  indicesTrain=[]
  indicesTest=[]

  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object):

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
    nombreEstrategia="null"
    numeroParticiones=0
    particiones=[]

    @abstractmethod
    def creaParticiones(self,datos,seed=None):
        pass

    def __del__(self):
        #del self.nombreEstrategia
        #del self.numeroParticiones
        del self.particiones[:]

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None,porcentajeEntrenamiento = 0.7):

        random.seed(seed)

        self.nombreEstrategia = "Simple"
        self.numeroParticiones = 2

        #Cogemos los datos y los "desordenamos" para correcto entrenamiento
        aux_array = np.array(())
        aux_array = random.permutation((len(datos.datos)))

        #Creamos las particiones de de entrenamiento y de testeo
        aux_particion = Particion()
        aux_particion.indicesTrain = aux_array[0:int(len(aux_array)*porcentajeEntrenamiento)]
        aux_particion.indicesTest = aux_array[int(len(aux_array)*porcentajeEntrenamiento):len(aux_array)]

        #Anadimos la particion creada a la estrategia
        self.particiones.append(aux_particion)
        pass

#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones
    # y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed = None, numeroParticiones = 5):

        random.seed(seed)

        self.nombreEstrategia = "Cruzada"
        self.numeroParticiones = numeroParticiones;

        #Cogemos los datos y los "desordenamos" para correcto entrenamiento
        aux_array = np.array(())
        aux_array = random.permutation((len(datos.datos)))

        aux_indice = (1.0 / numeroParticiones)
        test_tam = int(round(len(aux_array) * aux_indice))

        for i in range(0, numeroParticiones):
            #Creamos las particiones de de entrenamiento y de testeo
            aux_particion = Particion()
            if(i == 0):
                aux_particion.indicesTrain = aux_array[test_tam:]
                aux_particion.indicesTest = aux_array[0:test_tam]
            elif (i == numeroParticiones-1):
                aux_particion.indicesTrain = aux_array[0:-test_tam]
                aux_particion.indicesTest = aux_array[-test_tam:]
            else:
                trozo1 = aux_array[0: test_tam * i]
                trozo2 = aux_array[test_tam * (i+1):]
                aux_particion.indicesTrain = np.concatenate((trozo1, trozo2), axis=0)
                aux_particion.indicesTest = aux_array[ test_tam * i: test_tam * (i+1)]

            #Anadimos la particion creada a la estrategia
            self.particiones.append(aux_particion)
        pass
