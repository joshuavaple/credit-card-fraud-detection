# credit-card-fraud-detection
Credit card fraud detection on simulated data

# References
"The average U.S. cardholder makes 251 credit card transactions per year or one transaction every 1¾ days." (https://capitaloneshopping.com/research/number-of-credit-card-transactions/)

"The average credit card transaction in the U.S. is for $96.49" (https://capitaloneshopping.com/research/cash-vs-credit-card-spending-statistics/)

# Simulation Logics and Assumptions
## Number of Transactions per Day
- To model the number of credit card transactions per day for a person, a Poisson distribution would be appropriate. - The Poisson distribution is commonly used to model the number of events occurring in a fixed interval of time when these events happen with a known average rate and independently of the time since the last event.
- In reality, a person's credit card transactions may not be completely independent from each other, as there may be patterns or correlations in spending behavior. However, for the purpose of this simulation, we assume that the transactions are independent and identically distributed (i.i.d.) over time.
- For simplicity, we assume that a middle income group person makes an average of 1 transaction per day, which is in the vicinity of the statistics above. For other groups, the value is arbitrarily adjusted higher or lower based on the income group.

## Transaction Value
- For the value of each transaction, the log-normal distribution is used because:
    - The amount is bounded at zero
    - It's right-skewed, which aligns with the typical pattern of credit card transactions where there are many small purchases and fewer large ones
    - It can handle a wide range of values, from small everyday purchases to larger, less frequent expenses.
- To estimate the sigma value for a log-normal distribution of credit card transactions, we don't have specific statistics provided in the search results. However, we can use the method of moments to estimate sigma based on the given mean transaction value, e.g., of $96.49. Using the method of moments for a log-normal distribution, we can estimate sigma as follows:
- First, we need to estimate the variance of the transaction amounts. Without specific data, we can make an educated guess that the coefficient of variation (CV) for credit card transactions might be around 0.5 to 1.0.
- Let's assume a CV of 0.75 for this example. The variance can be calculated as:
    ```
    variance = (mean * CV)^2 = (96.49 * 0.75)^2 ≈ 5236.45
    ```
- Now we can use the formula for estimating sigma:
    ```
    σ^2 = ln(1 + variance / mean^2)
    σ^2 = ln(1 + 5236.45 / 96.49^2) ≈ 0.4724
    σ ≈ √0.4724 ≈ 0.6873
    ``` 
- This CV has several important implications for credit card transactions:
    - Relative dispersion: It measures the spread of transaction values relative to the mean, allowing for comparison of variability across different transaction value ranges.
    - Dimensionless measure: The CV is independent of the unit of measurement, making it useful for comparing variability between different types of transactions or customer segments1.
    - Skewness indicator: A higher CV suggests a more right-skewed distribution, indicating a greater proportion of larger transactions relative to the mean3.
    - Spending pattern insight: For credit card transactions, a higher CV might indicate more diverse spending habits, with a mix of small everyday purchases and occasional large expenses.
    - Customer segmentation: Different customer groups may have distinct CVs, reflecting varying spending behaviors and income levels5.
- For the purpose of the simulation, a range of CV values is chosen to represent different customer segments, from low to high income groups. The CV values are used to generate log-normal distributions of transaction amounts for each segment, with the mean transaction value adjusted accordingly.

## Consolidated Spending Profile
- The shape of the distribution will be different for different income groups and corresponding lifestyles.
- It is assumed that a more affluent person will have higher mean transaction values, more frequent transactions, and higher spread (CV) to include both daily expenses and more frequent large expenditures.
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