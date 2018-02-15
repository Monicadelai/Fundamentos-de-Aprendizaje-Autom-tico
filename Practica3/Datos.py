# Alfonso y Monica
# Pareja 13 Grupo 1461
import numpy as np

class Datos(object):

    TiposDeAtributos=('Continuo','Nominal')
    tipoAtributos=[]
    nombreAtributos=[]
    nominalAtributos=[]
    datos=np.array(())
    # Lista de diccionarios. Uno por cada atributo.
    diccionarios=[]

    #Procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):

        with open(nombreFichero,'r') as f:
            #num
            num_lines = int(f.readline())
            #nombres
            self.nombreAtributos = f.readline().strip("\n\r").split(',')
            #tipos
            aux_tipo = f.readline().strip("\n\r").split(',')
            for elem in aux_tipo:
                if elem not in self.TiposDeAtributos:
                    ValueError('Tipo no Soportado')

            for elem in aux_tipo:
                if elem == 'Nominal':
                    self.nominalAtributos.append(True)
                else:
                    self.nominalAtributos.append(False)

            #Guardamos la poscion en bytes donde comienzan los datos en el fichero
            pos_datos = f.tell()

            claves = []
            aux_line = f.readline().strip("\n\r").split(',')
            for i in range(0, len(aux_line)):
                if self.nominalAtributos[i] == True:
                    claves.append({aux_line[i]:0})
                    self.diccionarios.append({})
                else:
                    claves.append({})
                    self.diccionarios.append({})

            for j in xrange(0, num_lines-1):
                aux_line = f.readline().strip("\n\r").split(',')
                for k in range(0, len(aux_line)):
                    if claves[k] != {}:
                        if aux_line[k] not in claves[k].keys():
                            claves[k][aux_line[k]] = 0

            for i in range(0, len(claves)):
                ordenadas =  sorted(claves[i].keys())
                for j in range(0, len(ordenadas)):
                    self.diccionarios[i][ordenadas[j]] = j

            #Volvemos a la posci[1,2on en bytes donde empiezan los datos
            f.seek(pos_datos)

            #GUARDAMOS LOS DATOS CODIFICADOS SEGUN EL DICCIONARIO EN LA MATRIZ
            for i in range (0, num_lines):
                aux_datos = np.array(())
                aux_datos = aux_datos.astype(float)
                aux_line = f.readline().strip("\n\r").split(',')
                for j in range (0, len(aux_line)):
                    if self.nominalAtributos[j] == True:
                        aux_datos = np.append(aux_datos, self.diccionarios[j][aux_line[j]])
                    else:
                        aux_datos = np.append(aux_datos, float(aux_line[j]))

                if i == 0:
                    self.datos = aux_datos
                else:
                    self.datos = np.vstack([self.datos, aux_datos])


    def extraeDatos(self, idx):
        return self.datos[idx]

    def extraeDatosRelevantes(self, idx):
        return self.datos[:,idx]

    def diccionarioRelevante(self, idx):
        return np.array(self.diccionarios)[idx].tolist()

    def atribDiscretosRelevantes(self, idx):
        return np.array(self.nominalAtributos)[idx].tolist()

    def extraeNombreAtributos(self, idx):
        return np.array(self.nombreAtributos)[idx].tolist()

    def __del__(self):
        del self.nominalAtributos[:]
        del self.tipoAtributos[:]
        del self.datos
        del self.diccionarios[:]
        del self.nombreAtributos[:]
