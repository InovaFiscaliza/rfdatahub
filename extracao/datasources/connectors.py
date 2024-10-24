# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01a_connectors.ipynb.

# %% auto 0
__all__ = ['DBConnector', 'SQLServer', 'MongoDB']

# %% ../../nbs/01a_connectors.ipynb 3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pyodbc
from pymongo import MongoClient

# from motor.motor_asyncio import AsyncIOMotorClient


# %% ../../nbs/01a_connectors.ipynb 5
@dataclass
class DBConnector:
	"""Base class for database connectors"""

	def connect(self):
		raise NotImplementedError

	def get_parallel_connections(self, n):
		# This method returns a list of n pyodbc connection objects in parallel
		with ThreadPoolExecutor(max_workers=n) as executor:
			futures = [executor.submit(self.connect) for _ in range(n)]
			return [future.result() for future in futures]


# %% ../../nbs/01a_connectors.ipynb 7
@dataclass
class SQLServer(DBConnector):
	"""Class for connecting to SQL Server"""

	sql_params: dict

	def connect(self):
		"""This method returns a pyodbc connection object according to the os platform"""
		connection_string = f"""Driver={self.sql_params.get('driver')};
								Server={self.sql_params.get('server')};
								Database={self.sql_params.get('database')};
								Encrypt=no;"""
		if self.sql_params.get('trusted_conn'):
			connection_string += f"""MultipleActiveResultSets={self.sql_params.get('mult_results')};
								  Trusted_Connection=yes;"""
		else:
			connection_string += f"""UID={self.sql_params.get('username')};
								  PWD={self.sql_params.get('password')};
								  Trusted_Connection=no;"""
		try:
			return pyodbc.connect(connection_string, timeout=self.sql_params.get('timeout', 10000))
		except pyodbc.OperationalError as e:
			raise ConnectionError(
				'Não foi possível abrir uma conexão com o SQL Server. Esta conexão somente funciona da rede cabeada!'
			) from e


# %% ../../nbs/01a_connectors.ipynb 9
@dataclass
class MongoDB(DBConnector):
	"""Class for connecting to MongoDB"""

	mongo_uri: str

	def connect(self):
		"""This method returns a connected MongoClient object"""
		return MongoClient(self.mongo_uri)
