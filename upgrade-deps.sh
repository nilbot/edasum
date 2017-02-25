#!/usr/bin/env bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $PROJECT_ROOT

if [ ! -d $PROJECT_ROOT/venv/3.6 ]; then
	echo "no virtualenv..."
	exit 1
fi

source venv/3.6/bin/activate && pip install --upgrade -r requirements.txt

deactivate

exit 0
