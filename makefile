SHELL=/bin/bash

jp_launch:
	source env_vars.env && jt -t gruvboxd && jupyter notebook
