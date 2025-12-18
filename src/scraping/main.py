from box_score_scraping import scrape_multiple_matches
from main_page_scraping import extract_match_links

# urls = extract_match_links(mainpage='https://lnb.com.br/ldb/tabela-de-jogos/?season%5B%5D=78')
scrape_multiple_matches(urls='failed_urls.txt', output_csv="todas_as_partidas_falhadas_2.csv")