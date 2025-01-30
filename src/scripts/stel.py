import os
import warnings
import sys
import time

import typer
from dotenv import find_dotenv, load_dotenv
from fastcore.xtras import Path
from rfdatahub.datasources.sitarweb import Stel


load_dotenv(find_dotenv(), override=True)
warnings.simplefilter('ignore')


def main(read_cache: bool = False):
	start = time.perf_counter()
	data = Stel(read_cache=read_cache)
	data.update()
	data.save()
	print(f'Elapsed time: {time.perf_counter() - start} seconds')


if __name__ == '__main__':
	typer.run(main)
