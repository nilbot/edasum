import json
import pandas as pd

def as_dataframe(jl):
	return pd.read_json('[%s]' % ','.join(jl.splitlines()))
		