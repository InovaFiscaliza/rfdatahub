import time
from rfdatahub.datasources.srd import SRD
import typer


def main(limit: int = 0, read_cache: bool = False):
	start = time.perf_counter()
	data = SRD(read_cache=read_cache)
	data.update()
	data.save()
	print(f'Elapsed time: {time.perf_counter() - start} seconds')


if __name__ == '__main__':
	typer.run(main)
