SHELL=/bin/bash

jp_launch:
	source env_vars.env && jt -t gruvboxd && jupyter notebook

toymodel_raw_to_bronze:
	echo "Realizndo el proceso de transformacio datos del toy model"
	source env_vars.env && python "$(PROJECT_PATH)/scripts/preprocessing/pre_toymodel.py"
	echo "Transformacio y almacenamiento exitoso"