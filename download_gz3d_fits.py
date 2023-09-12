import os
import requests
from bs4 import BeautifulSoup

def is_link_to_fits_file(link_text):
    if link_text is None:
        return False
    else:
        return ".fits.gz" in link_text.lower()

def get_urls_to_download(file_list_loc):

    url = "https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo3d/v4_0_0/"
    f = requests.get(url)
    soup = BeautifulSoup(f.content, "html.parser")

    # https://realpython.com/beautiful-soup-web-scraper-python/#pass-a-function-to-a-beautiful-soup-method
    link_elements = soup.find_all('a', string=is_link_to_fits_file)
    links = [url+link['href'] for link in link_elements]

    with open(file_list_loc, 'w') as f:
        # f.write('\n'.join(links[:20]))  # for testing
       f.write('\n'.join(links))  # for full list
        
    return
    

def download_url_list(file_list_loc):

    # https://stackoverflow.com/questions/40986340/how-to-wget-a-list-of-urls-in-a-text-file
    """
    Terminal version
    cat data/gz3d/file_list.txt | xargs -n10 -P4 wget --content-disposition -P data/gz3d/fits_gz
    -P = directory prefix
    -c = only download if file-to-download is equal or greater size i.e. skip if present and correct\
    (possibly might not work due to .1 renaming)
    """
    os.system("cat "+file_list_loc+" | xargs -n10 -P4 wget --content-disposition -c -P data/gz3d/fits_gz")

    return

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == '__main__':

    file_list_loc = 'data/gz3d/file_list.txt'
    get_urls_to_download(file_list_loc)
    # exit()
    download_url_list(file_list_loc)