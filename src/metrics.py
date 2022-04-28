import zlib


def calculate_zlib_entropy_ratio(sentence: str) -> int:
    return len(zlib.compress(bytes(sentence, "utf-8")))


def calculate_is_similar(str1: str, str2: str, n_gram: int = 3) -> bool:
    ## Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).
    ## It is same as "Is string 1 is similar with string 2?"
    n_gram_set = lambda x: set([" ".join([str(j) for j in x[i:i+n_gram]]) for i in range(len(x)-n_gram)])

    ## Return true if str1 is similar (or duplicated) to str2 else false.
    ## It is not recommended to mark two strings as similar, trivially.
    return len(n_gram_set(str1) & n_gram_set(str2)) >= len(n_gram_set(str1)) / 2