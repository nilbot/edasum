import json
import pandas as pd

def as_dataframe(jl):
    df = pd.read_json('[%s]' % ','.join(jl.splitlines()), convert_dates=True)
    return df

