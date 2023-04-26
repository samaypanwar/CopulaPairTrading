from statsmodels.tsa.stattools import coint
from itertools import combinations
from itertools import compress
from tqdm import tqdm
import numpy as np
from statsmodels.regression.linear_model import OLS

# import statsmodels.api as sm


def find_pairs(data, method="coint"):

    if method == "coint":

        pvalues = []
        pairs = list(combinations(data.columns, 2))

        for x, y in tqdm(pairs, desc="Finding Pairs..."):
            result = coint(data[y], data[x], autolag="bic")
            pvalues.append(result[1])

        pvalues = np.where(np.array(pvalues) < 0.05, True, False)

        return list(compress(pairs, pvalues))


def find_ols_spread(y, x):

    # x = sm.add_constant(x)
    res = OLS(y, x).fit()

    return res.resid
