# FUNDAMENTOS DE APRENDIZAJE AUTOMÁTICO 2017 #

Practicas realizadas en la Escuela Politécnica Superior - Universidad Autónoma de Madrid

* Implementado con Python 2.7
* Utilizando Anaconda 2

## Clase Datos
Almacena los datos recibidos de un fichero de entrada y los almacena en una matriz de numpy.
Ademas de almacenar los datos, crea un diccionario para cada atributo que sea Nominal, para cada atributo este diccionario sigue orden lexicográfico para las claves.
Para valores continuos se alamacenan en tipo float, puesto que este es el tipo que tiene toda la matriz de datos.

## Clase Estrategia de Particionado
Esta clase tiene dos posibilidades: Simple y Cruzada. Para la validación simple se puede decidir cual es el porcentaje de entrenamiento. Para la validación cruzada se puede decidir
en número de particiones. Ambos métodos antes de partir hacen un random de la matriz de datos, asegurando un correcto entrenamiento.

## Clasificador Naive Bayes (NB)
Para el calculo de probabilidades se supone independencia de los datos. En entrena se crean tablas por cada atributo, en las que las filas son los valores que puede tomar
este atributo y las columnas las clases. En cada celda se almacena la frecuencia con la que se da cada valor con cada clase. Para atributos continuos se calcula la media y la varianza por cada posible clase.

En clasifica se hace uso de estas tablas para calcular las veromilitudes, y para los atributos contínuos se calcula su norma pdf. Después de calcular la probabilidad condicionada a cada posible clase nos quedamos con la mayor para realizar la predicción.
Después se compara las predicciones obtenidas con las clases reales y se devuelve la tasa de error en el caso de que la validación sea simple y un array de tasas de errores en caso de que
la validación sea cruzada.

## Clasificador Vecinos Próximos (KNN)
Primero normaliza la matriz de datos, para esto primero calcula la media y la desviación de cada atributo y las alamacena para posteriormente normalizar la matriz de datos.

Para clasificador se basa en el calculo de distancias entre lo que se quiere clasificar y todos los elementos de la matriz de datos de entrenamiento normalizados.
Despúes de este calculo de distancias nos quedamos con las k menores, y vemos cual es la clase predominante en esos k vecinos. La clase predominante será la que predecimos para el elemento a clasificar

## Clasificador Regresión Logístitca (RL)
Primer genera un vector W de con tantos elementos como columnas tiene la matriz de entrenamiento. Las componentes de este vector son aleatorias entre -0.5 y 0.5. Este vector se va actualizando el numero de épocas indicadas en función de los datos de la matriz. [formula]

Para clasificar, se calcula sigma de la fila de datos multiplicado por el vector W entrenado traspuesto. Si el valor de sigma es menor que 0.5 clasificamos como clase 0, en caso contrario clasificamos como clase 1.

## Clasificadores Sklearn
Módulo que contiene tanto para validación simple como para cruzada los mismos modelos que hemos implementado nosotros, pero usando la librería de Sklearn

## Tasas de Error
El fichero TasasClasificador.py es el main implementado para probar cualquier clasificador de los implementados.
Este recibe por argumento los parametros necesarios para la ejecución del clasificador seleccionado (tambien por parámetro).
Si se ejecuta con el parámetro -h ó --help se muestran todos los parámetros que acepta.

A continuación vamos a detallar todos estos parámetros para una correcta ejecución:

* **--porcentaje**              Numero real que representa el porcentaje de Entrenamiento para Validación Simple
* **--particiones**             Número entero para la validación cruzada
* **--NB**                      Utiliza el Clasificador de Naive Bayes
* **--KNN**                     Utiliza el Clasificador de Vecinos Próximos
* **--regLog**                  Utiliza el Clasificador de Regresión Logística
* **--laPlace**                 Utiliza Corrección de La Place (Sólo válido para Naive Bayes)
* **--k**                       Lista de números enteros que indica los vecinos con los que comparar (Sólo válido para KNN)
* **--norm**                    Indica que se quiere utilizar normalización en los datos (Sólo válido para KNN)
* **--epocas**                  Entero que indica el número de epocas que entrenar (Sólo válido para Regresión Logística)
* **--aprendizaje**             Real que indica la constante de aprendizaje (Sólo válido para Regresón Logística)

Todos estos parámetros son opcionales, el único obligatorio es la ruta donde se encuentra el fichero de datos.
Si no se introduce ninguno de los parámetros anteriormente descritos el programa tomará valores por defecto. Estos son:

* Porcentaje de Entrenamiento del 70%
* Número de particiones para Validación Cruzada de 5
* Clasificador de Naive Bayes sin correción de La Place

### Autores
* Alfonso Bonilla Trueba
* Mónica de la Iglesia Martinez
