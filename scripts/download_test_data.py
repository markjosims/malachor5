import urllib.request

"""
Download test data from google drive.
Right now just one wav file.
"""

id="1eZNLsETPnUp0gUBJBblpmaPysu3C57zW"
output="test/data/sample_biling.wav"
url=f'https://drive.google.com/uc?id={id}'

if __name__ == '__main__':
    urllib.request.urlretrieve(url, filename=output)