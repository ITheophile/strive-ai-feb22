import requests
from bs4 import BeautifulSoup


url = "https://www.imdb.com/search/title/?genres=action&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e0da8c98-35e8-4ebd-8e86-e7d39c92730c&pf_rd_r=Y6A61TQKQTHN1PC1ZXKZ&pf_rd_s=center-2&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr2_i_2"
page = requests.get(url)

soup = BeautifulSoup(page.text, 'html.parser')

movies = soup.find_all('span', class_ = 'lister-item-year text-muted unbold')

for movie in movies:
    print(movie.text)
