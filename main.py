from serpapi import GoogleSearch
import csv

# TO-DO List
# Experiment and figure out ideal parameters for other posture related products
# This is important since a combination of filters can severly limit the products listed

# NP meaning neck pillow
params_NP = {
  "engine": "amazon",
  "k": "Travel Neck Pillow",
  "amazon_domain": "amazon.com",
  "language": "en_US",
  "page": "1",
  "device": "desktop",
  "api_key": "bd030ccbc639f6dcffd341c14a6ee61e72cfdb238de3e95393ca6e5d884325e4",
  # p_72:1248915011 means 4 stars and up
  # p_n_g-101015541307111:122345955011 specifies pillows for standing position
  "rh": "p_72:1248915011, p_n_g-101015541307111:122345955011",
}

# function responsible for first writing the header, then # amount of pages
def csvwriter(file, params, pages):
  diction = (GoogleSearch(params)).get_dict()
  helper(file, "w", diction)
  while True:
    pages -= 1
    if (pages > 0):
      val = int(params['page'])
      params['page'] = str(val + 1)
      diction = (GoogleSearch(params)).get_dict()
      helper(file, "a", diction)
    else:
      break

# helper function for writing to csv file
def helper(file, mode, dict):
   with open (file, mode, newline = "", encoding="utf-8") as csvfile:
      fieldnames = ["Title", "Link", "Thumbnail", "Rating", "Reviews", "Print_Price", "Num_Price"]
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      if (mode == "w"):
        writer.writeheader()
      for results in dict.get('organic_results', []):
          writer.writerow({
            "Title" : results.get('title'),
            "Link" : results.get('link_clean'),
            "Thumbnail" : results.get('thumbnail'),
            "Rating" : results.get('rating'),
            "Reviews" : results.get('reviews'),
            "Print_Price" : results.get('price'),
            "Num_Price" : results.get('extracted_price'),
          })

if __name__ == "__main__":
  # name of file, parameters used for serpAPI, # of pages to be scraped
  csvwriter("neck_pillow_results.csv", params_NP, 2)

"""
responsible for printing information on console, writing to CSV will be used instead
for results in diction.get('organic_results', []):
    title = results.get('title')
    link = results.get('link_clean')
    thumbnail = results.get('thumbnail')
    rating = results.get('rating')
    reviews = results.get('reviews')
    print_price = results.get('price')
    num_price = results.get('extracted_price')
    print(f"Title: {title}")
    print(f"Link: {link}")
    print(f"Thumbnail: {thumbnail}")
    print(f"Rating: {rating}")
    print(f"Reviews: {reviews}")
    print(f"Price: {print_price}")
    print("-" * 100)
"""