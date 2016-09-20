#!/usr/bin/env bash
for p in $(ps ux | grep '[n]otebook' | awk '{print $2}'); do kill -9 $p; done

PROJECT_ROOT="$HOME/data-vis"

cd $PROJECT_ROOT

source venv/2.7/bin/activate && jupyter notebook 2>&1 &
disown

deactivate

