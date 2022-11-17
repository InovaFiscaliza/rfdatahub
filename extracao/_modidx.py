# Autogenerated by nbdev

d = { 'settings': { 'allowed_cell_metadata_keys': '',
                'allowed_metadata_keys': '',
                'audience': 'Developers',
                'author': 'Ronaldo S.A. Batista',
                'author_email': 'rsilva@anatel.gov.br',
                'black_formatting': 'True',
                'branch': 'master',
                'clean_ids': 'True',
                'copyright': 'Ronaldo S.A. Batista',
                'custom_sidebar': 'True',
                'description': 'This repository hold automation scripts to fetch, clean and merge data from licensed telecomunications entities available at '
                               'ANATEL Database Systems.',
                'dev_requirements': 'nbdev ipykernel jupyter ipywidgets pandas-profiling black mypy',
                'doc_baseurl': '/',
                'doc_host': 'https://ronaldokun.github.io/anateldb',
                'doc_path': '_docs',
                'git_url': 'https://github.com/ronaldokun/anateldb',
                'host': 'github',
                'jupyter_hooks': 'True',
                'keywords': 'database sql mongodb pyodbc pandas nbdev rich fastcore geopy whylogs pandas-profiling tqdm',
                'language': 'English',
                'lib_name': 'extracao',
                'lib_path': 'extracao',
                'license': 'apache2',
                'min_python': '3.8',
                'nbs_path': 'nbs',
                'readme_nb': 'index.ipynb',
                'recursive': 'False',
                'requirements': 'fastcore pyodbc pymongo pandas openpyxl pyarrow rich geopy tqdm',
                'status': '3',
                'title': 'Estações Licenciadas de Telecomunicações e Radiodifusão',
                'tst_flags': '',
                'user': 'ronaldokun',
                'version': '0.4.0'},
  'syms': { 'extracao\\constants': { 'extracao\\constants.APP_ANALISE': 'https://ronaldokun.github.io/extracao\\constants.html#app_analise',
                                     'extracao\\constants.BW': 'https://ronaldokun.github.io/extracao\\constants.html#bw',
                                     'extracao\\constants.BW_MAP': 'https://ronaldokun.github.io/extracao\\constants.html#bw_map',
                                     'extracao\\constants.BW_pattern': 'https://ronaldokun.github.io/extracao\\constants.html#bw_pattern',
                                     'extracao\\constants.COLS_SRD': 'https://ronaldokun.github.io/extracao\\constants.html#cols_srd',
                                     'extracao\\constants.COLS_TELECOM': 'https://ronaldokun.github.io/extracao\\constants.html#cols_telecom',
                                     'extracao\\constants.COLUNAS': 'https://ronaldokun.github.io/extracao\\constants.html#colunas',
                                     'extracao\\constants.ESTACAO': 'https://ronaldokun.github.io/extracao\\constants.html#estacao',
                                     'extracao\\constants.ESTADOS': 'https://ronaldokun.github.io/extracao\\constants.html#estados',
                                     'extracao\\constants.MONGO_SRD': 'https://ronaldokun.github.io/extracao\\constants.html#mongo_srd',
                                     'extracao\\constants.MONGO_TELECOM': 'https://ronaldokun.github.io/extracao\\constants.html#mongo_telecom',
                                     'extracao\\constants.REGEX_ESTADOS': 'https://ronaldokun.github.io/extracao\\constants.html#regex_estados',
                                     'extracao\\constants.RELATORIO': 'https://ronaldokun.github.io/extracao\\constants.html#relatorio',
                                     'extracao\\constants.SIGLAS': 'https://ronaldokun.github.io/extracao\\constants.html#siglas',
                                     'extracao\\constants.SQL_RADCOM': 'https://ronaldokun.github.io/extracao\\constants.html#sql_radcom',
                                     'extracao\\constants.SQL_STEL': 'https://ronaldokun.github.io/extracao\\constants.html#sql_stel',
                                     'extracao\\constants.SQL_VALIDA_COORD': 'https://ronaldokun.github.io/extracao\\constants.html#sql_valida_coord',
                                     'extracao\\constants.TIMEOUT': 'https://ronaldokun.github.io/extracao\\constants.html#timeout'},
            'extracao\\format': { 'extracao\\format.df_optimize': 'https://ronaldokun.github.io/extracao\\format.html#df_optimize',
                                  'extracao\\format.optimize_floats': 'https://ronaldokun.github.io/extracao\\format.html#optimize_floats',
                                  'extracao\\format.optimize_ints': 'https://ronaldokun.github.io/extracao\\format.html#optimize_ints',
                                  'extracao\\format.optimize_objects': 'https://ronaldokun.github.io/extracao\\format.html#optimize_objects',
                                  'extracao\\format.parse_bw': 'https://ronaldokun.github.io/extracao\\format.html#parse_bw'},
            'extracao\\main': { 'extracao\\main.add_aero': 'https://ronaldokun.github.io/extracao\\main.html#add_aero',
                                'extracao\\main.bump_version': 'https://ronaldokun.github.io/extracao\\main.html#bump_version',
                                'extracao\\main.check_modify_row': 'https://ronaldokun.github.io/extracao\\main.html#check_modify_row',
                                'extracao\\main.get_db': 'https://ronaldokun.github.io/extracao\\main.html#get_db',
                                'extracao\\main.get_modtimes': 'https://ronaldokun.github.io/extracao\\main.html#get_modtimes'},
            'extracao\\merging': { 'extracao\\merging.COLS': 'https://ronaldokun.github.io/extracao\\merging.html#cols',
                                   'extracao\\merging.MAX_DIST': 'https://ronaldokun.github.io/extracao\\merging.html#max_dist',
                                   'extracao\\merging.check_add_row': 'https://ronaldokun.github.io/extracao\\merging.html#check_add_row',
                                   'extracao\\merging.check_merging': 'https://ronaldokun.github.io/extracao\\merging.html#check_merging',
                                   'extracao\\merging.get_frequencies_set': 'https://ronaldokun.github.io/extracao\\merging.html#get_frequencies_set',
                                   'extracao\\merging.get_subsets': 'https://ronaldokun.github.io/extracao\\merging.html#get_subsets',
                                   'extracao\\merging.merge_aero': 'https://ronaldokun.github.io/extracao\\merging.html#merge_aero',
                                   'extracao\\merging.merge_closer': 'https://ronaldokun.github.io/extracao\\merging.html#merge_closer',
                                   'extracao\\merging.merge_triple': 'https://ronaldokun.github.io/extracao\\merging.html#merge_triple'},
            'extracao\\reading': { 'extracao\\reading.read_aero': 'https://ronaldokun.github.io/extracao\\reading.html#read_aero',
                                   'extracao\\reading.read_aisg': 'https://ronaldokun.github.io/extracao\\reading.html#read_aisg',
                                   'extracao\\reading.read_aisw': 'https://ronaldokun.github.io/extracao\\reading.html#read_aisw',
                                   'extracao\\reading.read_base': 'https://ronaldokun.github.io/extracao\\reading.html#read_base',
                                   'extracao\\reading.read_icao': 'https://ronaldokun.github.io/extracao\\reading.html#read_icao',
                                   'extracao\\reading.read_mosaico': 'https://ronaldokun.github.io/extracao\\reading.html#read_mosaico',
                                   'extracao\\reading.read_radcom': 'https://ronaldokun.github.io/extracao\\reading.html#read_radcom',
                                   'extracao\\reading.read_stel': 'https://ronaldokun.github.io/extracao\\reading.html#read_stel',
                                   'extracao\\reading.read_telecom': 'https://ronaldokun.github.io/extracao\\reading.html#read_telecom'},
            'extracao\\updates': { 'extracao\\updates.clean_mosaico': 'https://ronaldokun.github.io/extracao\\updates.html#clean_mosaico',
                                   'extracao\\updates.connect_db': 'https://ronaldokun.github.io/extracao\\updates.html#connect_db',
                                   'extracao\\updates.update_base': 'https://ronaldokun.github.io/extracao\\updates.html#update_base',
                                   'extracao\\updates.update_mosaico': 'https://ronaldokun.github.io/extracao\\updates.html#update_mosaico',
                                   'extracao\\updates.update_radcom': 'https://ronaldokun.github.io/extracao\\updates.html#update_radcom',
                                   'extracao\\updates.update_stel': 'https://ronaldokun.github.io/extracao\\updates.html#update_stel',
                                   'extracao\\updates.update_telecom': 'https://ronaldokun.github.io/extracao\\updates.html#update_telecom',
                                   'extracao\\updates.valida_coord': 'https://ronaldokun.github.io/extracao\\updates.html#valida_coord'}}}