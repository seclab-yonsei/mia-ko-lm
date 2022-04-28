import bs4
import re
import requests


def make_clean(text: str) -> str:
    ## Remove html tags.
    text = re.sub(r"<[^<>]*>", "", text)

    ## Remove additional blanks including '\n'.
    text = re.sub(r"[\s]+", " ", text)

    return text


def get_request_and_content(url: str) -> str:
    ## Try request.
    try:
        req = requests.get(url)
        if req.status_code != requests.codes["ok"]:
            print(f"Request code is not ok: {req.status_code}")
            return None
            
    except Exception as e:
        print(e)
        return None
    
    ## Parsing.
    bs = bs4.BeautifulSoup(req.text, "html.parser")

    ## Get text.
    text = make_clean(bs.text)
    return text