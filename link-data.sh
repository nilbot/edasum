#!/usr/bin/env bash

if [ ! $# -eq 1 ]; then
	echo "Usage: $0 date_of_collection";
	exit 1;
fi

DATUM=$1

cd $HOME

if [ -d "crawler" -a -d "data-vis/data" ]; then
	rm -f data-vis/data/*
	ln -s "$HOME/crawler/result/$DATUM/review_items.jl" "$HOME/data-vis/data/review_items.jl"
	ln -s "$HOME/crawler/result/$DATUM/hotel_items.jl" "$HOME/data-vis/data/hotel_items.jl"
	exit 0
fi

exit 1
