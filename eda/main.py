from scripts.clean_text import clean_text
from scripts.fetch_stocks import fetch_stocks
from scripts.sentiment import add_sentiment
from scripts.merge_sequence import merge_sequence

clean_text()
fetch_stocks()
add_sentiment()
merge_sequence()