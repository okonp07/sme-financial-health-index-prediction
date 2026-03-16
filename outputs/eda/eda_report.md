# SME Financial Health EDA

- Rows: 9618
- Columns: 39
- Duplicate IDs: 0
- Duplicate rows: 0

## Target distribution
- Low: 65.29%
- Medium: 29.82%
- High: 4.89%

## Highest missingness features
- uses_informal_lender: 46.67%
- uses_friends_family_savings: 46.66%
- motivation_make_more_money: 44.61%
- funeral_insurance: 43.54%
- medical_insurance: 43.54%
- business_age_months: 42.74%
- future_risk_theft_stock: 42.63%
- has_debit_card: 41.62%
- has_internet_banking: 41.62%
- has_loan_account: 41.58%
- current_problem_cash_flow: 39.28%
- marketing_word_of_mouth: 38.42%
- problem_sourcing_money: 37.29%
- has_mobile_money: 28.60%
- attitude_more_successful_next_year: 27.16%

## Data notes
- Target is heavily imbalanced, with `High` as the rarest class.
- Country patterns are material, especially Malawi and Eswatini.
- Monetary variables are highly skewed and need robust transformations.
- Many access-to-finance fields contain meaningful missing/unknown states.