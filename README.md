# credit-card-fraud-detection
Credit card fraud detection on simulated data

# References
- "The average U.S. cardholder makes 251 credit card transactions per year or one transaction every 1¾ days." (https://capitaloneshopping.com/research/number-of-credit-card-transactions/)
- "The average credit card transaction in the U.S. is for $96.49" (https://capitaloneshopping.com/research/cash-vs-credit-card-spending-statistics/)
- "Find log normal parameters for given mean and variance" (https://www.johndcook.com/blog/2022/02/24/find-log-normal-parameters/)

# Simulation Logics and Assumptions
## Number of Transactions per Day
- To model the number of credit card transactions per day for a person, a Poisson distribution would be appropriate. 
- The Poisson distribution is commonly used to model the number of events occurring in a fixed interval of time when these events happen with a known average rate and independently of the time since the last event.
- In reality, a person's credit card transactions may not be completely independent from each other, as there may be patterns or correlations in spending behavior. However, for the purpose of this simulation, we assume that the transactions are independent and identically distributed (i.i.d.) over time.
- For simplicity, we assume that a middle income group person makes an average of 1 transaction per day, which is in the vicinity of the statistics above. For other groups, the value is arbitrarily adjusted higher or lower based on the income group.

## Transaction Value
- For the value of each transaction, the log-normal distribution is used. As transaction values cannot be negative and therefore cannot follow the normal distribution, we assume that the natural log of the transaction values follows a normal distribution. This distribution is appropriate for our simulation because:
    - The amount is bounded at zero
    - It's right-skewed, which aligns with the typical pattern of credit card transactions where there are many small purchases and fewer large ones
    - It can handle a wide range of values, from small everyday purchases to larger, less frequent expenses.
- We will use the following conventions:
    - For the transaction values, they follow the log-normal distribution, and will have the statistical paramters called mean and CV (coefficient of variation). 
    - For the log(txn_value), they will follow the normal distribution, and will have the statistical parameters called mu and sigma.
- While the mean and CV values can be understood easily as they are about monetary values, the mu and sigma values are not as intuitive. We will use the function below to calculate the mu and sigma values given the mean and CV values:
    ```
    variance = (mean * CV)**2
    sigma = sqrt(ln(1 + variance / mean**2))
    mu = ln(mean) - sigma**2 / 2
    ```
- Once the mu and sigma values are calculated, we can use them back in the lognormal distribution function in Numpy to simulate the distribution of transaction values in monetary terms. For example, to generate 100 transaction values:
    ```
    txn_values = np.random.lognormal(mean=mu, sigma=sigma, size=100)
    ```
- Note that though the argument `mean` is used by Numpy, it is actually the mean of the lognormal distribution, not the mean of the transaction values. Thus, we use `mu` here, and should not be confused with the mean of the transaction values as we defined above.
- To better appreciate the use of CV value, it has several important implications for credit card transactions:
    - Relative dispersion: It measures the spread of transaction values relative to the mean, allowing for comparison of variability across different transaction value ranges.
    - Dimensionless measure: The CV is independent of the unit of measurement, making it useful for comparing variability between different types of transactions or customer segments1.
    - Skewness indicator: A higher CV suggests a more right-skewed distribution, indicating a greater proportion of larger transactions relative to the mean3.
    - Spending pattern insight: For credit card transactions, a higher CV might indicate more diverse spending habits, with a mix of small everyday purchases and occasional large expenses.
    - Customer segmentation: Different customer groups may have distinct CVs, reflecting varying spending behaviors.
- For the purpose of the simulation, a range of CV values is chosen to represent different customer segments, from low-spending to high-spending groups. The CV values are used to generate log-normal distributions of transaction amounts for each segment, with the mean transaction value adjusted accordingly.

## Consolidated Spending Profile
- The shape of the distribution will be different for different spending groups and corresponding lifestyles.
- It is assumed that a higher-spending person will have higher mean transaction values, more frequent transactions, and higher spread (CV) to include both daily expenses and more frequent large expenditures.
- These assumptions can be illustrated by following parameters:
    ```
    group_profiles = {
        'name': ['low', 'low-middle', 'middle', 'high-middle', 'high'],
        'txn_mean_low': [5, 20, 40, 60, 80],
        'txn_mean_high': [20, 40, 60, 80, 100],
        'txn_cv_low': [0.3, 0.4, 0.5, 0.6, 0.7],
        'txn_cv_high': [0.4, 0.5, 0.6, 0.7, 0.8],
        'txn_lambda': [0.25, 0.5, 1, 2, 3]
    }
    ```
- For a customer in each group, the actual value of 'txn_mean' and 'txn_cv' is picked from the uniform distribution bounded by the low and high values for that group.

- Using the 'middle' income group as the reference point, a simulation is done to generate 30-day worth of transactions in 50 trials, and compute the average total transaction. The values of 'txn_mean_low' and 'txn_mean_high' are adjusted to arrive at the desired average total transaction value for the 'middle' income group, which is about $1500 per month.

## Fraud Scenarios
There are a few common fraud patterns below that we can simulate. In reality, there are many more sophisticated fraud patterns, but these are some of the basic ones that should be detected by our model.
1. Unusual large transactions scattered through a number of days. This simulates the card-not-present (CNP) fraud occurs when a credit card transaction is made without the physical card being present because the card details have been leaked. The card owner is not aware of the compromised card and continue to use it for legitimate transactions, while the fraudsters make high-value transactions, until the next few days when the card owner notices the fraud. This scenario usually occurs in payments where there is no need to present the physical card for verification, such as online shopping. To detect this scenario, we will need features that can track the legitimate spending habits of the customers, as well as the features about the payment methods/merchant.
2. Large transactions in quick successions with increasing amounts. This simulate the scenario where the fraudster has stolen the card and is trying to make as many transactions as possible before the card owner notices the fraud. The fraudster starts with smaller amounts and increase the amounts in a short period of time.
3. A small transaction followed by a few large transaction in quick successions. This is similar to the previous scenario, but the fraudster starts with a small transaction to test the card, and then make a few large transactions.