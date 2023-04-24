Gaussian Copula modelling for pairs trading of many securities:

https://github.com/AnsonCHOWcm/PairTrading_Copula_Backtrader/blob/main/BackTrader_Copula_PairTrading_Hourly.py

get dataset
find cointegrated series ? then create the instrument in a pairs trading sense
either find pairs and employ strat
OR
do copula modelling and do individual stock longing rather than pairs

START yourself then change analysis to back trader refactor

### Things to do

Errors that are occuring:
1. CLME is not working -- fix
2. Fix CMLE later, work with kendall tau approximation for now, sample and see fit with alpha
3. Marginal sympy not working

1. Plot copula for two variables
2. Check with synthetic fit whether it looks same or not
3. Check for cointegration between pairs and then fit
1. Fit to two pairs of data a copula in jupyter notebook. Put into python file after
2. Expand by increasing the number of series and create
   1. either cointegrated pairs by testing of kendall tau/johansen cointegration
   2. all together
3. Based on the above two approaches, create a basic trading strategy.


### Sampling Algorithm
![img.png](img.png)
