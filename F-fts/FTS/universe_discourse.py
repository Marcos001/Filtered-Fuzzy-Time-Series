
import numpy as np

class UniverseDiscourse:

	def __init__(self,):
		'''
		Partition the universe of discourse.
		split_for_size: particionar por quantidade de intervalos com tamanhos iguais.
		'''
		pass
		
	def split_UD(self, starting, support, size):
		'''
		'''

		intervals = []

		for i in range(1, size+1):
			ini = starting + (i - 1) * support
			end = starting + i * support
			intervals.append([ini, end])
		return intervals



	def split_for_support(self, ts, support, d1=None, d2=None):

		'''
		create crip sets based length of support
		(implementing)
		'''

		self.U = [0,0]
		self.support = support
		
		if d1 and d2:
			self.U = [np.min(ts) - d1,  np.max(ts) + d2]
		else:
			self.U = [np.min(ts),  np.max(ts)]
		
		self.size = (self.U[1] - self.U[0]) / self.support

		intervalos = []
		ini = self.U[0]
		fim = 0

		while True:
		    fim = ini + self.support
		    conjunto = [ini, fim]
		    intervalos.append(conjunto)
		    ini = fim
		    if fim >= self.U[1]:
		        break

		return intervalos


	def split_for_size(self, ts, size, d1=None, d2=None):

		'''
		create crip sets based length of interval
		'''

		self.U = [0,0]
		self.size = size
		if d1 and d2:
			self.U = [np.min(ts) - d1,  np.max(ts) + d2]
		else:
			self.U = [np.min(ts),  np.max(ts)]
		
		self.support = (self.U[1] - self.U[0]) / self.size

		intervalos = []
		ini = self.U[0]
		intervalos.append(ini)

		while True:
		    ini += self.support
		    intervalos.append(ini)
		    if ini >= self.U[1]:
		        break

		inter = []

		for i in range(len(intervalos)-2):
			ini = intervalos[i]
			fim = intervalos[i+2]
			inter.append([ini, fim])

		return inter