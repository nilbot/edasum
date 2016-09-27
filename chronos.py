import pandas as pd
import calendar

def get_paddedyearbyfreq(year, freq):
    start = pd.to_datetime(year)
    if freq is 'W':
        return pd.date_range(start, periods=52,freq='W')
    elif freq is 'M':
        return pd.date_range(start, periods=12,freq='M')
    else:
        raise ValueError('input freq is illegal')
        
def get_yearsfromrange(startstr, endstr):
    start = pd.to_datetime(startstr).year
    end = pd.to_datetime(endstr).year
    return [str(x) for x in range(start,end+1)]

def get_freqfactor(freq):
    if freq is 'M':
        return calendar.month_abbr[1:13]
    elif freq is 'W':
        return ["Week "+str(x) for x in range(1,53)]
    else:
        raise ValueError('input freq is illegal')