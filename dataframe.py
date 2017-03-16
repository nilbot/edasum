import json
import pandas as pd

def as_dataframe(jl):
    df = pd.read_json(jl, lines=True, convert_dates=True)
    return df

