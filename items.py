from datetime import datetime

class ReviewItem:
    def __init__(self, user_id, item_id, review_id, review_title, rating, rating_perc, rating_date, review_text, site_name, url, batch_id):
        self.user_id = user_id
        self.item_id = item_id
        self.review_id = review_id
        self.review_title = review_title
        self.rating = rating
        self.rating_perc = float(rating_perc)
        self.rating_date = datetime.strptime(rating_date, "%Y-%m-%dT%H:%M:%S")
        self.review_text = review_text
        self.site_name = site_name
        self.url = url
        self.batch_id = batch_id

def as_review(dct):
    if 'user_id' in dct and \
    'item_id' in dct and \
    'review_id' in dct and \
    'review_title' in dct and \
    'rating' in dct and \
    'rating_percentage' in dct and \
    'rating_date' in dct and \
    'review_text' in dct and \
    'site_name' in dct and \
    'url' in dct and \
    'batch_id' in dct:
        return ReviewItem(dct['user_id'],
                         dct['item_id'],
                         dct['review_id'],
                         dct['review_title'],
                         dct['rating'],
                         dct['rating_percentage'],
                         dct['rating_date'],
                         dct['review_text'],
                         dct['site_name'],
                         dct['url'],
                         dct['batch_id'])
    else:
        return dct

class HotelItem:
    def __init__(self, user_id, item_id, description, site_name, url, batch_id):
        self.user_id = user_id
        self.item_id = item_id
        self.description = description
        self.site_name = site_name
        self.url = url
        self.batch_id = batch_id

def as_hotel(dct):
    if 'user_id' in dct and \
    'item_id' in dct and \
    'description' in dct and \
    'site_name' in dct and \
    'url' in dct and \
    'batch_id' in dct:
        return ReviewItem(dct['user_id'],
                         dct['item_id'],
                         dct['description'],
                         dct['site_name'],
                         dct['url'],
                         dct['batch_id'])
    else:
        return dct
