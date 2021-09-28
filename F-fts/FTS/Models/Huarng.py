import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.style.use('seaborn')

class HUARNG_2001_DB:

	def __init__(self,):
		'''
		Effective lengths of intervals to improve forecasting in fuzzy time series

		# Model for defining the universe of discourse and partitioning of
		breaks
		'''
		self.name_model = 'Model Huarng 2001 - distribution-based length' 
		print('='*len(self.name_model))
		print(self.name_model)
		print('='*len(self.name_model))

		self.__support = None
		self.__diff = []


	def difference_abs(self, ts): 
		"""
		absolute difference of ts
		ts: time series (N, 1)
		return: mean absolute difference, absolute difference
		"""
		diff_abs = [] 
		for i in range(len(ts)-1): 
			diff_abs.append(abs(ts[i]-ts[i+1])) 
		return np.mean(diff_abs), diff_abs 


	def choose_base_map_db(self, diff): 
		'''
		Select the base of intervals for intervals distribuition based
		'''
		if diff >= 0.1 and diff <= 1.0: 
			return 0.1 
		elif diff >= 1.1 and diff <= 10: 
			return 1 
		elif diff >= 11 and diff <= 100: 
			return 10 
		elif diff >= 101 and diff <= 1000: 
			return 100 
		else:
			return diff 


	def cumulative_distribution(self, diff, base):
		'''
		Calcule as ocorrences of cumulative distribution
		'''
		cd = {}
		lim =  base*10
		for i in range(base, lim, base):
			ocorrencias = 0
			
			for j in diff:
				if j >= i:
					ocorrencias += 1
				cd[i] = ocorrencias
		return cd


	def plot_cumulative_distribution(self, cd):
		'''
		cd: cumulative distribuition - dict
		'''
		x = list(cd.keys())
		y = list(cd.values())
		plt.figure(figsize=(12,8));
		plt.yticks(y);
		plt.xticks(x);
		plt.bar(x,y, width=2, color='k');
		plt.show()



class HUARNG_2001_AVG:

	def __init__(self,):
		'''
		determine the length of intervals based on the average.

		Average-based: determines the base through the half of the first difference
		putting on base of base mapping table.
	    '''
		self.name_model = 'Model Huarng 2001 - average-based length' 
		print('='*len(self.name_model))
		print(self.name_model)
		print('='*len(self.name_model))

		# self.__sets: quantidade de conjuntos fuzzy
		# self.__support: comprimento dos intervalos
		# self.__diff: array de diferenças absolutas
		# self.__base: base map table


	def difference_abs(self, ts): 
		"""
		absolute difference of ts
		ts: time series (N, 1)
		return: mean absolute difference, absolute difference
		"""
		diff_abs = [] 
		for i in range(len(ts)-1): 
			diff_abs.append(abs(ts[i]-ts[i+1])) 
		return np.mean(diff_abs), diff_abs 


	def choose_base_map(self, diff_half):
		'''
		Select the base of intervals for intervals averange based
		'''
		if diff_half >= 0.1 and diff_half <= 1.0:
			return 0
		elif diff_half >= 1.1 and diff_half <= 10:
			return 1
		elif diff_half >= 11 and diff_half <= 100: 
			return -1
		elif diff_half >= 101 and diff_half <= 1000:
			return -2
		else:
			return -3


	def average_based_length(self, diff_halt, base):
		'''
		Calculate length of intervals
		'''
		abl = np.around(diff_halt, decimals=base)
		return abl


	def get_support(self,):
		'''
		get support of intervals
		'''
		return 


	'''---------------- DEFINE FUZZY SETS --------------'''

	def fit(self, ts, col=None, d1=None, d2=None, ):

		self.D1 = d1
		self.D2 = d2
		self.ts = ts
		

		if col: self.col = col
		else: self.col = ts.columns.to_list()[0]

		self.__diff_mean, self.__diff = self.difference_abs(ts[self.col].values)

		self.__diff_half = self.__diff_mean / 2

		self.__base = self.choose_base_map(self.__diff_half)

		self.__support = int(self.average_based_length(self.__diff_half, self.__base))

		print('diff.......: {}'.format(self.__diff))
		print('diff mean..: {}'.format(self.__diff_mean))
		print('diff half..: {}'.format(self.__diff_half))
		print('base ......: {}'.format(self.__base))
		print('ABL .......: {}'.format(self.__support))



	def forecasting(self,):

		from Chen_model import Chen1996

		# define model
		model = Chen1996()


		print('support:', self.__support)
		# fitting
		model.fit(ts=self.ts, d1=55, d2=663, support=self.__support)

		# Forecasting
		model.forecasting(validation=True)


	'''--------- FUZZY LOGICAL RELATIONSHIPS -----------'''


pass
''' =========================================================== '''


def DB_Alabama(path_ts):
	print('Length Distribution-based')
		# read ts
	ts = pd.read_csv(path_ts, sep=';', index_col=[0])

	# instancie model
	model = HUARNG_2001_DB()

	# first average-based
	diff_mean, diff = model.difference_abs(ts.Enrollments.values)

	base = model.choose_base_map_db(diff_mean)

	print('diff.......: {}'.format(diff_mean))
	print('DB base ...: {}'.format(base))
	print('diff.......:',diff)

	cd = model.cumulative_distribution(diff, base)

	model.plot_cumulative_distribution(cd)


def AVG_Alabama(path_ts):

	# read ts
	ts = pd.read_csv(path_ts, sep=';', index_col=[0])

	# define model
	model = HUARNG_2001_AVG()

	# fitting model
	model.fit(ts)

	# forecasting
	model.forecasting()


def AVG_Taiex(path_ts, col=None, year=None):

	# read ts
	ts = pd.read_csv(path_ts, index_col=[0], parse_dates=[0])

	if year:
		ts = pd.DataFrame(ts[col][year])
		print(ts.head(2))
		print('='*30)
		print(ts.tail(2))

	print(min(ts[col]), max(ts[col]))
	# instancie model
	model = HUARNG_2001()

	# average-based
	diff_mean, _ = model.difference_abs(ts.avg.values)

	diff_mean_half = diff_mean / 2

	base = model.choose_base_map_avg(diff_mean_half)

	abl = model.average_based_length(diff_mean_half, base)

	print('diff.......: {}'.format(diff_mean))
	print('diff half..: {}'.format(diff_mean_half))
	print('base ......: {}'.format(base))
	print('ABL .......: {}'.format(abl))


if __name__ == '__main__':
	#AVG_Taiex('../data/csv/TAIEX.csv', col='avg', year='1997')
	#AVG_Alabama('../data/csv/Enrollments.csv')
	DB_Alabama('../data/csv/Enrollments.csv')