#!/usr/bin/env bash

PROJECT_ROOT="$HOME/data-vis"
cd $PROJECT_ROOT

if [ ! -d $PROJECT_ROOT/venv/2.7 ]; then
	echo "no virtualenv..."
	exit 1
fi

source venv/2.7/bin/activate && pip install --upgrade -r requirements.txt

deactivate

exit 0	
