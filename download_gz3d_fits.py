import requests
from bs4 import BeautifulSoup


def is_link_to_fits_file(link_text):
    if link_text is None:
        return False
    else:
        return ".fits.gz" in link_text.lower()

def get_urls_to_download():

    "https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo3d/v4_0_0/"
    with open('/Users/user/repos/zoobot-3d/static_sas_page.html') as f:
        soup = BeautifulSoup(f, "html.parser")

    # https://realpython.com/beautiful-soup-web-scraper-python/#pass-a-function-to-a-beautiful-soup-method
    link_elements = soup.find_all('a', string=is_link_to_fits_file)
    links = [link['href'] for link in link_elements]

    with open('static_sas_page_fits_links.txt', 'w') as f:
        f.write('\n'.join(links[:20]))


if __name__ == '__main__':

    get_urls_to_download()

    # https://stackoverflow.com/questions/40986340/how-to-wget-a-list-of-urls-in-a-text-file
    # on mac: brew install wget
    """
    cat static_sas_page_fits_links.txt | xargs -n10 -P4 wget --content-disposition --trust-server-names
    """


