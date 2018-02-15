# Alfonso Bonilla Trueba
# Monica de la Igleia Martinez
# Pareja 13 Grupo 1461

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from Clasificador import Clasificador
from argparse import ArgumentParser
from SklearnClasificadores import *
from plotModel import plotModel
import matplotlib.pyplot as plt
import EstrategiaParticionado as EstrategiaParticionado
import Clasificador as Clasificador
import PreprocesamientoAG as AG
import numpy as np
import sys



parser = ArgumentParser(
    version='2.4.3',
    description="Por defecto:\n numero de particiones = 5, porcentaje de entrenamiento = 0.7. El Clasificador usado por defeco es Naive Bayes",
    epilog='Alfonso Bonilla Trueba y Monica de la Iglesia Martinez'
)

parser.add_argument("ruta", type=str, help="Ruta del fichero .data")

clasificador_group = parser.add_mutually_exclusive_group()
clasificador_group.add_argument("--KNN", help="Utiliza el Clasificador de Vecinos Proximos", action="store_true")
clasificador_group.add_argument("--NB", help="Utiliza el Clasificador de Naive Bayes", action="store_true")
clasificador_group.add_argument("--regLog", help="Utiliza el Clasificador de Regresion Logistica", action="store_true")
clasificador_group.add_argument("--AG", help="Utiliza el Algoritmo Genetico", action="store_true")

parser.add_argument("--porcentaje", type=float, default=0.7, help= "Porcentaje Train Validacion Simple")
parser.add_argument("--particiones", type=int, default=5, help= "Numero de particiones Validacion Cruzada")
parser.add_argument("--laPlace", help="Usar correccion de La Place", action="store_true")
parser.add_argument('--k', type=int, nargs='+', help='Numero de Vecinos a probar (Admite una secuencia de numeros para probar varios K en una ejecucion)')
parser.add_argument("--norm", help="Normaliza la matriz de datos", action="store_true")
parser.add_argument("--aprendizaje", type=float, default = 1.0, help= "Constante de Aprendizaje para Regresion Logistica")
parser.add_argument("--epocas", type=int, help= "Numero de epocas para el entrenamiento")
parser.add_argument("--sklearn", help="Utiliza el clasificador de la libreria Sklearn", action="store_true")
parser.add_argument("--plot", help="Crea la grafica del Clasificador seleccionado", action="store_true")
parser.add_argument("--poblacion", type=int, help="Tamano de la poblacion")
parser.add_argument("--generaciones", type=int, help="Numero de generaciones que evoluciona el AG")

args = parser.parse_args()

if (args.KNN or args.regLog) and args.laPlace:
    sys.exit("No se puede aplicar La Place con el Clasificador seleccionado")
if (args.NB or args.regLog) and args.k != None:
    sys.exit("El valor de k solo es para el clasificador KNN")
if (args.NB or args.regLog) and args.norm:
    sys.exit("La normalizacion solo es para el clasificador KNN")
if args.KNN and args.k == None:
    sys.exit("Vecinos Proximos (KNN) necesita al menos un valor de K")
if  args.KNN and min(args.k)<1:
    sys.exit("No se puede introducir un valor de K menor a 1")
if args.NB and args.plot:
    sys.exit("Naive Bayes no admite plot de datos")
if args.porcentaje > 0.99:
    sys.exit("El porcentaje de entrenamiento no puede ser superior a 0.99")
if (args.KNN or args.NB) and args.aprendizaje != 1.0:
    sys.exit("El Clasificador seleccionado no tiene constante de aprendizaje")
if (args.KNN or args.NB) and args.epocas != None:
    sys.exit("El Clasificador seleccionado no tiene epocas en su entrenamiento")
if args.regLog and args.epocas == None:
    sys.exit("Hay que especificar el numero de epocas para la Regresion Logistica")
if  args.regLog and args.epocas < 1:
    sys.exit("El valor de epocas tiene que ser igual o mayor a 1")
if args.plot and args.KNN and len(args.k) > 1:
    sys.exit("El plot de KNN solo puede realizarse de una K por cada vez")
if not args.AG and (args.poblacion != None or args.generaciones != None):
    sys.exit("El atributo es solo para el Algoritmo Genetico")

#####################################################################################################
try:
    del estrategia
    del clasificador
    del dataset
except NameError:
    pass

#####################################################################################################
dataset = Datos(args.ruta)
#####################################################################################################
if args.plot:
    if args.KNN:
        if args.norm:
            titulo =(args.ruta.split('/')[-1]) + (": KNN con k= %d y normalizando") % args.k[0]
        else:
            titulo =(args.ruta.split('/')[-1]) + (": KNN con k= %d sin normalizar") % args.k[0]
        clasificador = Clasificador.ClasificadorVecinosProximos()
        clasificador.k_vecinos = args.k[0]
        clasificador.normalizacion = args.norm
    elif args.regLog:
        titulo = (args.ruta.split('/')[-1]) + (": Regresion Logistica con %d") % args.epocas + (" epocas")
        clasificador = Clasificador.ClasificadorRegresionLogistica()
        clasificador.epocas = args.epocas
        clasificador.cte_aprendizaje = args.aprendizaje
    else:
        sys.exit("Debes especifiar el modelo (KNN o regLog) y sus correspondientes argumentos")

    estrategia = EstrategiaParticionado.ValidacionSimple()
    error = clasificador.validacion(estrategia, dataset, clasificador)
    ii = estrategia.particiones[-1].indicesTrain

    plotModel(dataset.datos[ii,0],dataset.datos[ii,1],dataset.datos[ii,-1]!=0,clasificador,titulo,dataset.diccionarios)

    plt.plot(dataset.datos[dataset.datos[:,-1]==0, 0], dataset.datos[dataset.datos[:,-1]==0, 1],'ro')
    plt.plot(dataset.datos[dataset.datos[:,-1]==1, 0], dataset.datos[dataset.datos[:,-1]==1, 1],'bo')
    plt.show()

    del estrategia
    del clasificador
    del dataset

    sys.exit(0)

#####################################################################################################
print "  Fichero: ", args.ruta.split('/')[-1]
print ("  - Porcentaje Entrenamiento en Validacion simple %.2f" % (args.porcentaje*100)) + "%"
if not args.AG:
    print "  - Numero de Particiones en Validacion cruzada: ", args.particiones

if args.NB:
    if args.laPlace:
        print "  - Usando la Correccion de La Place"
    else:
        print "  - Sin usar Correccion de La Place"

#####################################################################################################
if args.sklearn:
    print "  - Usando la libreria Sklearn"
    print "----------------------------------------------------"

    if args.KNN:
        print "Usando Validacion Simple"
        for k in args.k:
            error = KnnSimpleSklearn(dataset, args.porcentaje, k)
            print("  Error del %.2f" % round(error*100,2)) + "% KNN con k =", k
        print "Usando Validacion Cruzada"
        for k in args.k:
            errores = KnnCruzadaSklearn(dataset, args.particiones, k)
            print("  Error del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Desviacion del %.2f" % round(np.std(errores)*100,2)) + "% KNN con k =", k

    elif args.regLog:
        error = RegresionLogisticaSimpleSklearn(dataset, args.porcentaje, args.epocas)
        print("  El error con Validacion Simple es del %.2f" % round(error*100,2)) + "%"
        errores = RegresionLogisticaCruzadaSklearn(dataset, args.particiones, args.epocas)
        print("  El error medio con Validacion Cruzada es del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Con una desviacion de %.2f" % round(np.std(errores)*100,2)) + "%"
    elif args.AG:
        seleccion = PreprocesamientoAGSklearnKBest(dataset, args.porcentaje)
        print "La Seleccion de atributos usando SelectKBest es:"
        atributos = dataset.extraeNombreAtributos([i for i,x in enumerate(seleccion) if x==True])
        for atributo in atributos:
            print "   - " + atributo

        print "\nLa Seleccion de atributos usando SelectFromModel es:"
        seleccion = PreprocesamientoAGSklearnFromModel(dataset, args.porcentaje)
        atributos = dataset.extraeNombreAtributos([i for i,x in enumerate(seleccion) if x==True])
        for atributo in atributos:
            print "   - " + atributo
    else:
        error = NaiveBayesSimpleSklearn(args.laPlace, dataset, args.porcentaje, args.epocas)
        print("  El error con Validacion Simple es del %.2f" % round(error*100,2)) + "%"
        errores = NaiveBayesCruzadaSklearn(args.laPlace, dataset, args.particiones, args.epocas)
        print("  El error medio con Validacion Cruzada es del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Con una desviacion de %.2f" % round(np.std(errores)*100,2)) + "%"

#####################################################################################################
else:
    print "  - Usando nuestra libreria"
    print "----------------------------------------------------"
    estrategia = EstrategiaParticionado.ValidacionSimple()

    if args.KNN:
        print "Usando Validacion Simple"
        for k in args.k:
            clasificador = Clasificador.ClasificadorVecinosProximos()
            clasificador.k_vecinos = k
            clasificador.normalizacion = args.norm
            errores = clasificador.validacion(estrategia, dataset, clasificador, porcentajeTrain=args.porcentaje)
            print("  Error del %.2f" % round(errores[0]*100,2)) + "% KNN con k =", k
            del clasificador
    elif args.regLog:
        clasificador = Clasificador.ClasificadorRegresionLogistica()
        clasificador.epocas = args.epocas
        clasificador.cte_aprendizaje = args.aprendizaje
        errores = clasificador.validacion(estrategia, dataset, clasificador, porcentajeTrain=args.porcentaje)
        print("  El error con Validacion Simple es del %.2f" % round(errores[0]*100,2)) + "%"
        del clasificador
    elif args.AG:
        procesa = AG.PreprocesamientoAG()
        procesa.generaciones = args.generaciones
        procesa.tam_pobl = args.poblacion

        clasificador_skl = LogisticRegression(max_iter=4)

        seleccionados = procesa.seleccionaAtributos(dataset, clasificador_skl)

        print "La Seleccion de atributos usando AG es:"
        atributos = dataset.extraeNombreAtributos([i for i,x in enumerate(seleccionados) if x==1])
        for atributo in atributos:
            print "   - " + atributo

    else:
        clasificador = Clasificador.ClasificadorNaiveBayes()
        clasificador.laPlace = args.laPlace
        errores = clasificador.validacion(estrategia, dataset, clasificador, porcentajeTrain=args.porcentaje)
        print("  El error con Validacion Simple es del %.2f" % round(errores[0]*100,2)) + "%"
        del clasificador

    del estrategia

    estrategia = EstrategiaParticionado.ValidacionCruzada()

    if args.KNN:
        print "Usando Validacion Cruzada"
        for k in args.k:
            clasificador = Clasificador.ClasificadorVecinosProximos()
            clasificador.k_vecinos = k
            clasificador.normalizacion = args.norm
            errores = clasificador.validacion(estrategia, dataset, clasificador, numeroParts=args.particiones)
            print("  Error del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Desviacion del %.2f" % round(np.std(errores)*100,2)) + "% KNN con k =", k
            del clasificador
    elif args.regLog:
        clasificador = Clasificador.ClasificadorRegresionLogistica()
        clasificador.epocas = args.epocas
        clasificador.cte_aprendizaje = args.aprendizaje
        errores = clasificador.validacion(estrategia, dataset, clasificador, numeroParts=args.particiones)
        print("  El error medio con Validacion Cruzada es del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Con una desviacion de %.2f" % round(np.std(errores)*100,2)) + "%"
        del clasificador
    elif args.AG:
        pass
    else:
        clasificador = Clasificador.ClasificadorNaiveBayes()
        clasificador.laPlace = args.laPlace
        errores = clasificador.validacion(estrategia, dataset, clasificador, numeroParts=args.particiones)
        print("  El error medio con Validacion Cruzada es del %.2f" % round(np.mean(errores)*100,2)) + "%" + (" Con una desviacion de %.2f" % round(np.std(errores)*100,2)) + "%"
        del clasificador

    del estrategia

del dataset
