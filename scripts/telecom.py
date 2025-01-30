import time
from rfdatahub.datasources.telecom import Telecom
import typer


def main(limit: int = 0, read_cache: bool = True):
	start = time.perf_counter()
	data = Telecom(limit=limit, read_cache=read_cache)
	data.update()
	data.save()
	print(f'Elapsed time: {time.perf_counter() - start} seconds')


if __name__ == '__main__':
	typer.run(main)
