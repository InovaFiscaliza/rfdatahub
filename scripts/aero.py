import time
import typer
from extracao.datasources.aeronautica import Aero


def main():
	start = time.perf_counter()

	data = Aero()

	data.update()
	data.save()

	print(150 * '=')

	print(data.df.Fonte.value_counts())

	print(150 * '=')

	print(f'Elapsed time: {time.perf_counter() - start} seconds')


if __name__ == '__main__':
	typer.run(main)
