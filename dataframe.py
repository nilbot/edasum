import json
import pandas as pd

def as_dataframe(jl):
    df = pd.read_json('[%s]' % ','.join(jl.splitlines()), convert_dates=True)
    ## these 2 lines doesn't seem to work? #TODO
    df['rating_date'] = pd.to_datetime(df['rating_date'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('rating_date',inplace=True)
    return df