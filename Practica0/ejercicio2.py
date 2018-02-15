#!/usr/bin/python
# Importamos el modulo del sistema
# Alfonso Bonilla y Monica de la Iglesia
import sys
import numpy

#main por defecto
if __name__ == '__main__':
	sep = "-"
	c = "A-B-C-D-E-F"

	print ''.join(c.split(sep))

	c = "recuadrar"
	print c.rjust(40+len(c), '<').ljust(80+len(c), '>')