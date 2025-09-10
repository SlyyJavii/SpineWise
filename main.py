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
  # p_n_g-101015541307111:122345955011 specifies pillows for sitting position
  "rh": "p_72:1248915011, p_n_g-101015541307111:122345955011",
}

# PC meaning posture corrector, which encompasses a variety of braces tailored for back/shoulder support
params_PC = {
  "engine": "amazon",
  "k": "Posture Corrector",
  "amazon_domain": "amazon.com",
  "language": "en_US",
  "page": "1",
  "device": "desktop",
  "api_key": "bd030ccbc639f6dcffd341c14a6ee61e72cfdb238de3e95393ca6e5d884325e4",
  # p_72:1248903011 means 4 stars and up
  # p_n_g-101015541307111:122345955011 specifies braces for posture correction
  # p_n_g-101015646819111:122566773011 specifies products designed for adults
  # p_n_g-101015233022111:121833112011 specifies unisex products
  "rh": "p_72:1248903011, p_n_feature_three_browse-bin:23711946011, p_n_g-101015646819111:122566773011, p_n_g-101015233022111:121833112011",
}

# RB meaning resistance bands, short exercises with this can help improve posture
params_RB = {
  "engine": "amazon",
  "k": "Resistance Bands",
  "amazon_domain": "amazon.com",
  "language": "en_US",
  "page": "1",
  "device": "desktop",
  "api_key": "bd030ccbc639f6dcffd341c14a6ee61e72cfdb238de3e95393ca6e5d884325e4",
  # p_72:1248957011 means 4 stars and up
  # p_n_condition-type:6503254011 specifies only new products
  # p_n_g-101014802546111:116624134011 specifies bands >= 3 ft
  "rh": "p_72:1248957011, p_n_condition-type:6503254011, p_n_g-101014802546111:116624134011",
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
            delivery = results.get('delivery')
            # this if statement ensures that fields with null data in prices are not written to csv
            if (not (delivery is None) and (not ("out of stock" in delivery[-1])) and (not results.get('price') is None)):
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
  # csvwriter("neck_pillow_results.csv", params_NP, 2)
  csvwriter("posture_corrector_results.csv", params_PC, 2)
  #csvwriter("resistance_bands_results.csv", params_RB, 2)

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