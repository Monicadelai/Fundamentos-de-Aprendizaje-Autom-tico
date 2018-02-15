# Alfonso Bonilla Trueba
# Monica de la Igleia Martinez
# Pareja 13 Grupo 1461

from Datos import Datos
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import random
import math

class PreprocesamientoAG(object):

    prob_cruce = 0.6
    prob_mut = 0.35
    prob_mut_bit = 0.01
    prob_elitismo = 0.05
    tam_pobl = 50
    generaciones = 50

    def seleccionaAtributos(self, dataset, clasificador):
        nrows, ncols = dataset.datos.shape
        diccionarios = dataset.diccionarios
        atributosDiscretos = dataset.nominalAtributos
        datos = dataset.datos

        fitnes_last_gen = []

        poblacion = np.array(())
        parada = False # Condicion de Parada

        #Poblacion aleatoria
        for individuo in range(0,self.tam_pobl):
            gen = np.array(())
            for cromosoma in range(0, ncols - 1):
                gen = np.append(gen, random.randint(0,1))
            if individuo == 0:
                poblacion = gen

            else:
                poblacion = np.vstack([poblacion, gen])

        for num_gen in range(0,self.generaciones):
            # Vamos con el Fitness que es el clasificador
            for individuo in poblacion:
                atributos_seleccionados = [i for i,x in enumerate(individuo) if x==1]
                extraccion_datos = dataset.extraeDatosRelevantes(atributos_seleccionados)

                matriz_seleccion = np.concatenate((extraccion_datos, np.atleast_2d(datos[:,-1]).T), axis=1)


                nominalAtributos = dataset.atribDiscretosRelevantes(atributos_seleccionados)

                encAtributos = preprocessing.OneHotEncoder(categorical_features=nominalAtributos[:],sparse=False)
                X = encAtributos.fit_transform(matriz_seleccion[:,:-1])
                Y = matriz_seleccion[:,-1]

                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = (1.0 - 0.7))

                clasificador.fit(X_train, y_train)

                acierto = clasificador.score(X_test, y_test)
                fitnes_last_gen.append(acierto)

                if acierto >= 0.95:
                    print "Fitness del individuo: %f" %acierto
                    return individuo
            # HASTA AQUI BUCLE

            copia_fitnes = []
            copia_fitnes = fitnes_last_gen[:]
            # print len(copia_fitnes)
            lista_indices = []
            for i in range(0,len(fitnes_last_gen)):
                lista_indices.append(i)

            # print len(lista_indices)
            #guardamos en poblacion_aux las muestras de elitismo
            fitnex_max = sorted(fitnes_last_gen, reverse=True)
            fitnex_max_index = []
            for i in range(0,len(fitnes_last_gen)):
                fitnex_max_index.append(copia_fitnes.index(fitnex_max[i]))
                copia_fitnes[copia_fitnes.index(fitnex_max[i])] = -1

            # print len(fitnex_max_index)

            poblacion_aux = np.array(())

            for i in range(0,int(math.ceil(self.prob_elitismo * self.tam_pobl))):
                if i == 0:
                    poblacion_aux = poblacion[fitnes_last_gen.index(fitnex_max[i])]
                    lista_indices.remove(fitnex_max_index[i])
                else:
                    poblacion_aux = np.vstack((poblacion_aux, poblacion[fitnes_last_gen.index(fitnex_max[i])]))
                    lista_indices.remove(fitnex_max_index[i])


            #guardamos en la nueva poblacion las muestras con mutacion
            mutacion = []
            #lista con 9 0's y un 1
            lista_prob_mut = []
            for i in range(0,10):
                if i == 0:
                    lista_prob_mut.append(1)
                else:
                    lista_prob_mut.append(0)
            for i in range(0,int(self.prob_mut * self.tam_pobl)):
                mutacion_aux = []
                mutacion = poblacion[random.choice(lista_indices)][:]
                j=0
                while j<len(mutacion):
                    random.shuffle(lista_prob_mut)
                    mut = random.choice(lista_prob_mut)
                    #hacer la mutacion
                    if mut == 0:
                        mutacion_aux.append(mutacion[j])
                    else:
                        if mutacion[j] == 1:
                            mutacion_aux.append(0.0)
                        else:
                            mutacion_aux.append(1.0)
                    j+=1

                poblacion_aux = np.vstack((poblacion_aux, mutacion_aux))


            #guardamos en la nueva poblacion las muestras con cruce
            padre1 = []
            padre2 = []

            for i in range(0,int((self.prob_cruce * self.tam_pobl)/2)):
                padre1 = poblacion[random.choice(lista_indices)][:]
                padre2 = poblacion[random.choice(lista_indices)][:]
                hijo1 = []
                hijo2 = []

                j=0
                while j<len(padre1):
                    c_x = random.randint(0,1)
                    if c_x == 1:
                        hijo1.append(padre1[j])
                        hijo2.append(padre2[j])
                    else:
                        hijo1.append(padre2[j])
                        hijo2.append(padre1[j])
                    j+=1

                poblacion_aux = np.vstack((poblacion_aux, hijo1))
                poblacion_aux = np.vstack((poblacion_aux, hijo2))


            del copia_fitnes[:]
            del lista_indices[:]
            del fitnex_max_index[:]
            del fitnex_max[:]
            del fitnes_last_gen[:]
            poblacion = poblacion_aux[:]
            num_gen+=1
