import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
class fuzzy sets and membership funtions
by default, create fuzzy set symmetrical
'''

class Triangular:

	def __init__(self, a=None, m=None, b=None, crip_sets=None, df_params=None):

		self.a = a
		self.m = m
		self.b = b

		if crip_sets:
			self.define_parameters(crip_sets)
		elif df_params.all():
			self.a = df_params['a']
			self.m = df_params['m']
			self.b = df_params['b']

	def define_parameters(self, _set):
		self.a = np.min(_set)
		self.b = np.max(_set)
		self.m = (self.b + self.a) / 2

	def get_parameters(self,):
		return {'a':self.a, 'm':self.m, 'b':self.b}

	def get_pertinence(self, x):
		'''
		get pertincen of value x
		'''
		if x >= self.a and x < self.m: return (x - self.a) / (self.m - self.a)
		elif x >= self.m and x < self.b: return (self.b - x) / (self.b - self.m)
		else: return 0


class Gaussina:

	def __init__(self, m=None, std=None, crip_sets=None, df_params=None):

		self.m = m
		self.std = std

		if crip_sets:
			self.define_parameters(crip_sets)
		elif df_params.all():
			self.m = df_params['m']
			self.std = df_params['std']

	def define_parameters(self, _set):
		self.m = np.mean(_set)
		self.std = np.std(_set)

	def get_parameters(self,):
		return {'m':self.m, 'std':self.std}

	def get_pertinence(self, x):
		'''
		get pertincen of value x
		'''
		return np.exp(-((x-self.m)**2)/(self.std**2))


class Trapezoidal:

	def __init__(self, a=None, m=None, n=None, b=None, crip_sets=None, df_params=None):

		self.a = a
		self.m = m
		self.n = n
		self.b = b

		if crip_sets:
			self.define_parameters(crip_sets)
		elif df_params.all():
			self.a = df_params['a']
			self.m = df_params['m']
			self.n = df_params['n']
			self.b = df_params['b']

	def define_parameters(self, _set):
		self.a = np.min(_set)
		self.b = np.max(_set)
		self.mini_suport = (self.b - self.a) / 4
		self.m = self.a + self.mini_suport
		self.n = self.b - self.mini_suport

	def get_parameters(self,):
		return {'a':self.a, 'm':self.m, 'n':self.n, 'b':self.b}

	def get_pertinence(self, x):
		'''
		get pertincen of value x
		'''
		if x >= self.a and x < self.m:
			return (x - self.a) / (self.m - self.a)
		elif x >= self.m and x < self.n:
			return 1
		elif x >= self.n and x < self.b: 
			return (self.b - x) / (self.b - self.n)
		else: return 0



class FuzzySets:

	def __init__(self, crip_sets, MF='triangular'):
		'''
		MF: type membership function = triangular(tri), gaussiana(g), trapezoidal(tra) 
		crip_sets: array containing array with crypt sets  [[u.1],[u.2],...[u.n]], 
		with at least two elements.
		'''
		self.mf = MF        
		self.crip_sets = crip_sets
		self.terms = []
		self.size = len(crip_sets)

		self.terms = self.create_terms()
		self.df_fuzzy_sets = self.create_fuzzy_sets()
		self.vector_crip_sets = self.vectorize_data()


	def create_terms(self, name='A'):
		'''
		create terms of the varuable linguistic name(A)
		'''
		terms = []
		for i in range(self.size): terms.append(name+'.'+str(i+1))
		return terms 


	def create_fuzzy_triangular(self,):
		df = []
		for i in range(self.size):
			df.append(Triangular(crip_sets=self.crip_sets[i]).get_parameters())
		return pd.DataFrame(df, columns=['a','m','b'], index=self.terms)


	def create_fuzzy_gaussiana(self,):
		df = []
		for i in range(self.size):
			df.append(Gaussina(crip_sets=self.crip_sets[i]).get_parameters())
		return pd.DataFrame(df, columns=['m','std'], index=self.terms)


	def create_fuzzy_trapezoidal(self,):
		df = []
		for i in range(self.size):
			df.append(Trapezoidal(crip_sets=self.crip_sets[i]).get_parameters())
		return pd.DataFrame(df, columns=['a','m','n','b'], index=self.terms)


	def create_fuzzy_sets(self, ):
		'''
		create a daframe with info pertinence funtions
		'''
		df = []

		if self.mf is 'tri' or self.mf is 'triangular':
			return self.create_fuzzy_triangular()
		elif self.mf is 'g' or self.mf is 'gaussiana':
			return self.create_fuzzy_gaussiana()
		elif self.mf is 'tra' or self.mf is 'trapezoidal':
			return self.create_fuzzy_trapezoidal()
		else:
			print('Invalid name functions fuzzy')


	def get_pertinence(self, sets, value):
		if self.mf is 'tri' or self.mf is 'triangular':
			return Triangular(df_params=sets).get_pertinence(value)
		elif self.mf is 'g' or self.mf is 'gaussiana':
			return Gaussina(df_params=sets).get_pertinence(value)
		else:
			return Trapezoidal(df_params=sets).get_pertinence(value)


	def fuzzificar(self, value):

		max_pert = {'term':None, 'pertinence':0}

		for i in range(self.size):

			pertinence = self.get_pertinence(self.df_fuzzy_sets.iloc[i], value)

			if pertinence > max_pert['pertinence']:
				max_pert['pertinence'] = pertinence
				max_pert['term'] = self.terms[i]

		return max_pert


	def fuzzificar_ts(self, ts_values, plot=False):
		fuzzification = []

		for i in range(len(ts_values)):
			print(ts_values[i],'->',self.fuzzificar(ts_values[i]))
			fuzzification.append(self.fuzzificar(ts_values[i])['term'])

		# plot annotations os term in time series
		if plot:
			plt.plot(ts_values, '-o')
			for i in range(len(ts_values)):
				plt.annotate(fuzzification[i], (i, ts_values[i]))
			plt.show()

		self.df_fts = pd.DataFrame({'ts':ts_values, 'term':fuzzification})


	def vectorize_data(self,):
		x = []
		for i in range(self.size):
			for j in range(len(self.crip_sets[i])):
				x.append(self.crip_sets[i][j])
		return np.array(x)
	

	def plot_MF(self,):
		pass


	def plot_data(self,):
		'''
		plot fuzzy sets
		'''
		import matplotlib.pyplot as plt
		plt.style.use('ggplot')

		X = []
		Y = []
		x = np.sort(self.vector_crip_sets)

		for i in range(self.size):
			y = []
			for j in range(x.shape[0]):
				y.append(self.get_pertinence(self.df_fuzzy_sets.iloc[i], x[j]))
				
			X.append(x)
			Y.append(y)

		# plot
		print('-'*30)
		for i in range(len(Y)):
			plt.plot(X[i], Y[i], '-o')
			print(X[i], '->', Y[i])
		plt.show()
			



if __name__ == '__main__':

	conjuntos = [ list(np.arange(10,20,1)),[15, 25], [20, 30],[25, 27, 30, 35]]

	# fz = FuzzySets(crip_sets=conjuntos, MF='tra')
	# # print(fz.df_fuzzy_sets)
	# # print('-'*30)
	# # print(fz.fuzzificar(18))

	fz = FuzzySets(crip_sets=conjuntos, MF='tra')
	print(fz.df_fuzzy_sets)
	print('-'*30)
	print(fz.vector_crip_sets)
	fz.plot_data()