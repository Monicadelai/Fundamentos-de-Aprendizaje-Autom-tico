#!/usr/bin/python
# Importamos el modulo del sistema
# Alfonso Bonilla y Monica de la Iglesia
import sys

#main por defecto
if __name__ == '__main__':
	c = "ABCDEF"
	print c[::2] + '-' + c[1::2]

	c = "hOlA"
	print c.capitalize()

	fichero = 'Foto de vacaciones.JPG'
	print fichero.replace(" ", "_")[:-3] + fichero[-3:].lower()