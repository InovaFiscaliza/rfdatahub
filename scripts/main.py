import json
import os
import shutil
import warnings
from datetime import datetime
import sys
import subprocess

import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv
from fastcore.xtras import Path

from extracao.stations import Estacoes

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
	parallel: bool = True,  # Caso verdadeiro efetua as requisições de forma paralela em cada fonte de dados
	read_cache: bool = True,  # Caso verdadeiro lê os dados já existentes, do contrário efetua a atualização dos dados
	reprocess_sources: bool = False,
) -> 'pd.DataFrame':  # Retorna o DataFrame com as bases da Anatel e da Aeronáutica
	"""Função para encapsular a instância e atualização dos dados"""
	import time

	start = time.perf_counter()
	data = Estacoes(SQLSERVER_PARAMS, MONGO_URI, limit, parallel, read_cache, reprocess_sources)
	data.update()
	data.save()
	mod_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
	mod_times = {'ANATEL': mod_time, 'AERONAUTICA': mod_time, 'ReleaseDate': mod_time}
	versiondb = json.loads((data.folder / 'Release.json').read_text())
	version = versiondb['rfdatahub']['Version']
	version_parts = version.split('.')
	version_parts[-1] = str(int(version_parts[-1]) + 1)
	new_version = '.'.join(version_parts)
	versiondb['rfdatahub']['Version'] = new_version
	versiondb['rfdatahub'].update(mod_times)
	json.dump(versiondb, (data.folder / 'Release.json').open('w'))
	if path is not None:
		if (path := Path(path)).exists():
			# path.mkdir(parents=True, exist_ok=True)
			print(f'Salvando dados em {path}')
			subprocess.run(
				['powershell', '-Command', f'"robocopy {data.folder} {path} /E /IS /IT"'],
				check=False,
			)

	# Create a release with the version tag and generate release notes
	subprocess.run(
		[
			'gh',
			'release',
			'create',
			new_version,
			'--generate-notes',
			'.\\extracao\\datasources\\arquivos\\saida\\estacoes.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\log.parquet',
		],
		check=False,
	)

	# Delete the 'rfdatahub' release and cleanup the tag
	subprocess.run(
		[
			'gh',
			'release',
			'delete',
			'rfdatahub',
			'--cleanup-tag',
			'-R',
			'InovaFiscaliza/.github',
			'-y',
		],
		check=False,
	)

	# Create a new 'rfdatahub' release with specific title, notes, and files
	subprocess.run(
		[
			'gh',
			'release',
			'create',
			'rfdatahub',
			'-t',
			'RFDataHub',
			'--notes',
			new_version,
			'.\\extracao\\datasources\\arquivos\\saida\\estacoes.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\log.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\smp.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\smp_raw.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\srd.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\srd_raw.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\telecom.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\telecom_raw.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\aero.parquet',
			'.\\extracao\\datasources\\arquivos\\saida\\Release.json',
			'-R',
			'InovaFiscaliza/.github',
		],
		check=False,
	)

	print(f'Elapsed time: {time.perf_counter() - start} seconds')


if __name__ == '__main__':
	typer.run(get_db)
