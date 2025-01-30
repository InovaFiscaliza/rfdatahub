from rfdatahub.stations import Estacoes
import pandas as pd
from fastcore.xtras import Path
import time

start = time.perf_counter()

anatel = Estacoes()

df = pd.read_parquet(
	Path(__file__).parent.parent
	/ 'extracao'
	/ 'datasources'
	/ 'arquivos'
	/ 'saida'
	/ 'telecom.parquet.gzip'
)

df = anatel.validate_coordinates(df)

print(df)

# print(150 * "=")

# print("DISCARDED!")

# print(data.discarded[["Frequência", "Entidade", "Log"]])


print(f'Elapsed time: {time.perf_counter() - start} seconds')
