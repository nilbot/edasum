#!/usr/bin/env bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
cd $PROJECT_ROOT

if [ ! -d $PROJECT_ROOT/venv/3.6 ]; then
	echo "no virtualenv..."
	exit 1
fi

for p in $(ps ux | grep '[n]otebook' | awk '{print $2}'); do kill -9 $p; done

source venv/3.6/bin/activate && jupyter notebook --debug > notebook.log 2>&1 &disown

exit 0
