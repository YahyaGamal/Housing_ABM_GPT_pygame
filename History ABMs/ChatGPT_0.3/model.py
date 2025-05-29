import random
import numpy as np
from config import *
from utils import calculate_repayment, calculate_max_mortgage

class House:
    def __init__(self, id, owner=None):
        self.id = id
        self.owner = owner
        self.price = 0
        self.rent = 0
        self.occupied = False
        self.for_rent = False
        self.for_sale = False
        self.tenants = None
        self.age = 0

    def decay_price(self):
        if self.for_sale:
            self.price *= PRICE_DROP_RATE

    def decay_rent(self):
        if self.for_rent:
            self.rent *= PRICE_DROP_RATE

    def demolish(self, age_threshold=100):
        return self.age > age_threshold

class Household:
    def __init__(self, id, is_owner, income):
        self.id = id
        self.is_owner = is_owner
        self.income = income
        self.capital = income * random.uniform(1.0, 3.0)
        self.owned_houses = []
        self.rented_house = None
        self.mortgages = []  # List of (house, val)
        self.repayment = []  # List of repayment values

    def affordability(self):
        return (self.income * AFFORDABILITY) / STEPS_PER_YEAR

    def calculate_mortgage(self):
        max_repay = self.affordability()
        max_mortgage = calculate_max_mortgage(max_repay)
        deposit = max_mortgage * (1/LOAN_TO_VALUE - 1)
        return max_mortgage, deposit

    def rent_threshold(self):
        return self.affordability()

    def evaluate_status(self):
        repay_total = sum(r for r in self.repayment)
        rental_income = sum(h.rent for h in self.owned_houses if h.tenants)
        total_income = self.income + rental_income
        median_price = np.median([h.price for h in self.owned_houses]) if self.owned_houses else 0
        median_repay = np.median(self.repayment) if self.repayment else 0
        
        is_rich = self.capital > OMEGA * median_price * (1 - LOAN_TO_VALUE) and total_income - repay_total > median_repay
        is_poor = repay_total > BETA * AFFORDABILITY * total_income
        
        return "rich" if is_rich else "poor" if is_poor else "normal"

class Realtor:
    def __init__(self, id):
        self.id = id
        self.listings = []
    
    def evaluate_price(self, house, all_prices):
        median_price = np.median(all_prices)
        if house.price == 0:
            return median_price
        if median_price > 2 * house.price:
            return 2 * house.price
        elif median_price < 0.5 * house.price:
            return 0.5 * house.price
        else:
            return median_price