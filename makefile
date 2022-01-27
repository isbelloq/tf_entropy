SHELL=/bin/bash

jp_launch:
	source env_vars.env && jt -t gruvboxd && jupyter notebook

toymodel_raw_to_bronze:
	@echo "Realizndo el proceso de transformacio datos del toy model"
	source env_vars.env && python scripts/preprocessing/pre_toymodel.py
	@echo "Transformacio y almacenamiento exitoso"

toymodel_eda:
	@echo "Realizndo el proceso EDA para toy model"
	source env_vars.env && python scripts/eda/eda_toydata.py
	@echo "EDA exitoso"

secop_raw_to_bronze:
	@echo "Realizndo el proceso de transformacio datos del para SECOP I"
	source env_vars.env && python scripts/preprocessing/pre_secop.py
	@echo "Transformacio y almacenamiento exitoso"

secop_eda:
	@echo "Realizndo el proceso EDA para SECOP I"
	source env_vars.env && python scripts/eda/eda_secop.py
	@echo "EDA exitoso"
