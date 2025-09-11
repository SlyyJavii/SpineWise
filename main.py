from serpapi import GoogleSearch
from base64 import b64encode
from base64 import b64decode
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
import csv

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
  "delivery_zip": 0
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
  "delivery_zip": 0
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
  "delivery_zip": 0
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

# generates a cipher using AES algorithm in Ciphertext Block Chaining mode
def AES_CBC(input, key, mode):
   # encryption
  if (mode == 0 and type(input) == str):
    # zip code needs to be a byte array before passed to encrypt()
    data = input.encode('utf-8')
    # generates symmetric block cipher in CBC mode
    cipher = AES.new(key, AES.MODE_CBC)
    # AES requires data to have a length of 16 bytes, padding is added to ensure that requirement is met
    ct = cipher.encrypt(pad(data, AES.block_size))
    # byte arrays are encoded in base 64
    # iv is an initialization vector, which is a random number added to each cipher object to make sure two identical plain texts have distinct cipher texts
    iv = b64encode(cipher.iv)
    ct = b64encode(ct)
    result = {'iv': iv, 'ciphertext': ct}
    return result
  # decryption
  elif (mode == 1 and type(input) == dict):
    try:
      iv = b64decode(input['iv'])
      ct = b64decode(input['ciphertext'])
      cipher = AES.new(key, AES.MODE_CBC, iv)
      original = unpad(cipher.decrypt(ct), AES.block_size).decode('utf-8')
      return original
    except (ValueError, KeyError):
      print("Incorrect decryption")

def get_key():
  # byte string of length 16 is generated and returned as key
  return get_random_bytes(16)

def get_zip_code():
  while True:
    zip_code = input("Enter your zip code:")
    if (len(zip_code) != 5 or (not zip_code.isdigit())):
      print("\nZip Code is invalid. Try again")
    else:
      break

  params_NP['delivery_zip'] = params_PC['delivery_zip'] = params_RB['delivery_zip'] = zip_code
  return zip_code

if __name__ == "__main__":
  # when integrating, discard main method
  # main method contains example uses of methods in this file
  
  zip_code = get_zip_code()
  key = get_key()
  # AES_CBC(zipcode, key, mode)
  # mode = 0, encryption
  # mode = 1, decryption
  encrypted_data = AES_CBC(zip_code, key, 0)
  print(encrypted_data)
  decrypted_data = AES_CBC(encrypted_data, key, 1)
  print(decrypted_data)

  # csvwriter(name of file, parameters used for serpAPI, # of pages to be scraped)
  # csvwriter("neck_pillow_results.csv", params_NP, 2)
  # csvwriter("posture_corrector_results.csv", params_PC, 2)
  # csvwriter("resistance_bands_results.csv", params_RB, 2)