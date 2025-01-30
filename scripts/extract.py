import os
import warnings
import sys
import subprocess

import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv
from fastcore.xtras import Path

from rfdatahub.stations import Estacoes

load_dotenv(find_dotenv(), override=True)
warnings.simplefilter('ignore')

SQLSERVER_PARAMS = dict(
	driver=os.environ.get('SQL_DRIVER'),
	server=os.environ.get('SQL_SERVER'),
	database=os.environ.get('SQL_DATABASE'),
	trusted_conn=True,
	mult_results=True,
	encrypt=False,
	timeout=int(os.environ.get('SQL_TIMEOUT')),
)

if sys.platform in ('linux', 'darwin', 'cygwin'):
	SQLSERVER_PARAMS.update(
		{
			'trusted_conn': False,
			'mult_results': False,
			'username': os.environ.get('USERNAME'),
			'password': os.environ.get('PASSWORD'),
		}
	)

MONGO_URI: str = os.environ.get('MONGO_URI')


def get_db(
	path: str = os.environ.get('DESTINATION'),  # Pasta onde salvar os arquivos",
	limit: int = 0,  # Número máximo de registros a serem extraídos da cada base MongoDB, 0: sem limite
	parallel: bool = False,  # Caso verdadeiro efetua as requisições de forma paralela em cada fonte de dados
	read_cache: bool = False,  # Caso verdadeiro lê os dados já existentes, do contrário efetua a atualização dos dados
	reprocess_sources: bool = False,
) -> 'pd.DataFrame':  # Retorna o DataFrame com as bases da Anatel e da Aeronáutica
	"""Função para encapsular a instância e atualização dos dados"""
	import time

	start = time.perf_counter()
	data = Estacoes(SQLSERVER_PARAMS, MONGO_URI, limit, parallel, read_cache, reprocess_sources)
	data.update()
	if path is not None:
		if path := Path(path):
			path.mkdir(parents=True, exist_ok=True)
			print(f'Salvando dados em {path}')
			subprocess.run(
				['powershell', '-Command', f'"robocopy {data.folder} {path} /E /IS /IT"'],
				check=False,
			)
	print(f'Elapsed time: {time.perf_counter() - start} seconds')


if __name__ == '__main__':
	typer.run(get_db)
