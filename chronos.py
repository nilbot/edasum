import pandas as pd
import calendar
import heapq
import matplotlib as plt

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

def get_topHotels(data=None,n=10):
	if data is None:
		raise ValueError("data has to be reviews grouped by hotels")
	dct = {}
	for url, hotel_grp in data:
		subset = hotel_grp.ix[:,'rating'].set_index('timestamp_rating')
		summary = subset.describe()
		dct[url] = summary.get_value('count')
	heap = [(value,key) for key,value in dct.items()]
	res = heapq.nlargest(n,heap)
	res = [(key,value) for value,key in res]
	return res

def pltHotelRatingByYears(hotel, start, end, freq='W', data=None):
    if data is None or freq not in ['D','W','M']:
        raise ValueError("input invalid; freq has to be [D,W,M]; hotel has to be item_id; data has to be reviews")
    hotelDF = data[data.item_id == hotel].set_index('timestamp_rating')
    mu = hotelDF.ix[:,'rating'][start:end].resample(freq).mean()
    cnt = hotelDF.ix[:,'rating'][start:end].resample(freq).count()
    mudfs = [group[1] for group in mu.groupby(mu.index.year)]
    cntdfs = [group[1] for group in cnt.groupby(cnt.index.year)]
    years = get_yearsfromrange(start,end)
    
    interval = ""
    if freq is 'D':
        interval = 'day'
    elif freq is 'W':
        interval = 'week'
    else:
        interval = 'month'

    ylabels = [s+interval for s in ['Avg rating for ', '# of rating for ']]
    
    for i in range(len(mudfs)):
        padded_year_freq = get_paddedyearbyfreq(years[i],freq)
        mudfs[i] = mudfs[i].reindex(padded_year_freq).fillna(0)
        cntdfs[i] = cntdfs[i].reindex(padded_year_freq).fillna(0)
        
        x1 = cntdfs[i].index
        y1 = cntdfs[i].values
        x2 = mudfs[i].index
        y2 = mudfs[i].values

        yield x1,y1,x2,y2,ylabels
        