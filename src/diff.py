import copy
import difflib

import numpy as np


DIFFLIB_ADD_SIGN = "+ "
DIFFLIB_SUB_SIGN = "- "
DIFFLIB_ORI_SIGN = "  "


# class ColorsForPrint:
    
#     RED = "\033[41m"    ## Added
#     GREEN = "\33[42m"   ## Deleted
#     ENDC = "\033[0m"    ## End operator


# class ColorBoxForLatex:

#     @staticmethod
#     def red(text: str, red_color: str = "red") -> str:
#         ## e.g., \colorbox{BrickRed}{This is red}
#         return "\\colorbox{" + red_color + "}{" + text + "}"

#     @staticmethod
#     def green(text: str, green_color: str = "green") -> str:
#         ## e.g., \colorbox{ForestGreen}{This is blue}
#         return "\\colorbox{" + green_color + "}{" + text + "}"


def get_difference(reference: str, hypothesis: str) -> tuple:
    """ Get difference of two string.
        Assert reference includes hypothesis.
    """
    sp, ep = np.inf, np.inf
    out = list(difflib.Differ().compare(hypothesis, reference))

    ## Get start point (sp) by original sentences.
    for i, c in enumerate(out):
        if c.startswith(DIFFLIB_ORI_SIGN):
            sp = min(sp, i)

    ## Get end point (ep) by reversed sentences.
    for i, c in enumerate(out[::-1]):
        if c.startswith(DIFFLIB_ORI_SIGN):
            ep = min(ep, i)

    ## Slice approximately.
    strict_reference = copy.deepcopy(reference[sp:-ep])

    ## Calculate difference ratio in range [0, 1].
    ratio = difflib.SequenceMatcher(None, strict_reference, hypothesis).ratio()

    ## Return.
    return ratio, strict_reference
