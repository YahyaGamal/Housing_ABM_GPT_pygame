from config import *

def calculate_max_mortgage(repayment):
    i = INTEREST_RATE
    d = MORTGAGE_DURATION_YEARS
    s = STEPS_PER_YEAR
    try:
        loan = repayment / i * (1 - (1 + i) ** (-d * s))
    except ZeroDivisionError:
        loan = 0
    return loan

def calculate_repayment(loan_value):
    i = INTEREST_RATE
    d = MORTGAGE_DURATION_YEARS
    s = STEPS_PER_YEAR
    return (loan_value * i) / (1 - (1 + i) ** (-d * s))