import pygame
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
from statistics import median
import math

# ========== PARAMETERS with given values ==========

INTEREST_RATE_ANNUAL = 0.037        # 3.7%
PROPENSITY_THRESHOLD = 0.2          # For investment willingness
OCCUPANCY_RATIO = 0.95
OWNERS_TO_TENANTS = 0.5             # 50%
LTV = 0.9                          # Loan to value max 90%
MORTGAGE_DURATION_YEARS = 25
RATE_DURATION_M_RANGE = (2, 5)      # not extensively modeled here but reserved
RATE_DURATION_BTL_RANGE = (1, 5)
MEAN_INCOME = 30000
WAGE_INCREASE = 0                   # No wage increase per year
AFFORDABILITY = 0.33                # Max portion of income on housing
SAVINGS_M = 0.20                   # % yearly income savings for mortgage holders
SAVINGS_R = 0.05                   # % yearly income savings for renters
HOMELESS_PERIOD = 5                 # Max steps homeless before exit (5 quarters)
SEARCH_LENGTH = 5                   # Number of houses to consider in market search
CONSTRUCTION_RATE = 0.006           # 0.6% houses built per quarter
ENTRY_RATE = 0.04                  # 4% households enter system per quarter
EXIT_RATE = 0.02                   # 2% households exit system per quarter
REALTOR_TERRITORY = 16             # Realtor's neighborhood radius (abstract)
PRICE_DROP_RATE = 0.03             # 3% price/rent drop rate per unsold step
RENT_DROP_RATE = 0.03
SAVINGS_THRESHOLD_M = 2            # Factor for rich threshold mortgage
EVICT_THRESHOLD_M = 1              # Factor for poor threshold mortgage
SAVINGS_THRESHOLD_R = 2            # Factor for rich threshold renter
EVICT_THRESHOLD_R = 1              # Factor for poor threshold renter
STEPS_PER_YEAR = 4                 # 4 steps per year (quarterly)

GRID_WIDTH = 30                   # 900 houses total (30x30 grid)
GRID_HEIGHT = 30

CELL_SIZE = 16                   # Square for house drawing

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

GRAPH_AREA_WIDTH = SCREEN_WIDTH - (GRID_WIDTH * CELL_SIZE)

# Derived interest rate per step (quarter)
INTEREST_RATE = INTEREST_RATE_ANNUAL / STEPS_PER_YEAR

# Some colors
COLOR_BG = (30, 30, 30)
COLOR_MORTGAGE_HOUSE = (0, 150, 0)
COLOR_RENT_HOUSE = (150, 0, 0)
COLOR_FOR_SALE = (255, 255, 0)
COLOR_FOR_RENT = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)

# Initialize pygame font (default)
pygame.font.init()
FONT_SMALL = pygame.font.SysFont('Arial', 14)
FONT_LARGE = pygame.font.SysFont('Arial', 22)

# -------------------------------------
# Helper: mortgage repayment formula (fixed payment mortgage)
def calc_max_mortgage(a, rate_per_step, duration_steps):
    # From Kohn(1990), repayment formula inverted to get principal max M
    # a = payment per period, rate_per_step = interest rate per step, duration_steps = total steps
    if rate_per_step == 0:
        return a * duration_steps
    else:
        return a / rate_per_step * (1 - (1 + rate_per_step) ** (-duration_steps))

# -- End helper

# ================== Agent Classes ===================

class House:
    def __init__(self, grid_pos, age=None):
        self.grid_pos = grid_pos  # (x,y) on grid
        self.price = 0.0
        self.rent = 0.0
        self.my_owner = None     # Household owning it (mortgage household)
        self.my_occupier = None  # Household occupying it
        self.for_sale = False
        self.for_rent = False
        self.age = age if age is not None else np.random.exponential(scale=1000)
        self.demolish_age = int(np.random.exponential(scale=50) + 50)
        self.on_market_steps = 0  # how long on market
        self.is_new = True        # for initial construction

    def age_house(self):
        self.age += 1
        if self.age >= self.demolish_age:
            return True
        return False

    def update_price_rent_decay(self):
        # Decay prices/rents if offered for sale/rent but unsold
        if self.for_sale:
            self.price *= (1 - PRICE_DROP_RATE)
            if self.price < 10000:  # Floor price for realism
                self.price = 10000
        if self.for_rent:
            self.rent *= (1 - RENT_DROP_RATE)
            if self.rent < 100:
                self.rent = 100

class Household:
    def __init__(self, unique_id, initial_income, step):
        self.id = unique_id
        self.type = None  # 'mortgage' or 'rent'
        self.income = initial_income
        self.propensity = np.random.random()  # willingness to invest
        self.capital = 0.0
        self.my_house = None  # House they currently occupy
        self.my_ownership = set()  # Houses they own
        self.mortgage = dict()     # dict house -> remaining mortgage float
        self.paid_mortgage = dict()
        self.repayment = dict()    # dict house -> repayment per step
        self.rate_duration = 0
        self.rent = 0.0            # rent paid if renter
        self.market_status = None  # 'buying', 'selling', 'renting', 'cooldown', or None
        self.market_steps = 0      # How long listed for sale/rent or homeless duration
        self.homeless_steps = 0
        self.cooldown_steps = 0
        # Initialize capital at first step:
        self.init_capital(step)
        self.exit_flag = False     # to indicate exit from system

    def init_capital(self, step):
        # eta_i sampled from gamma distribution shape/scale fitted approx UK distribution
        # We assume income distributions gamma shaped, so eta_i similarly drawn approx (scale=1.5)
        eta_i = np.random.gamma(shape=2.0, scale=1.5)
        self.capital = self.income * eta_i

    def calculate_repayment(self, house_price):
        # Given house price, calculate repayment per step using mortgage formula
        deposit = house_price * (1 - LTV)
        mortgage_loan = house_price - deposit
        duration_steps = MORTGAGE_DURATION_YEARS * STEPS_PER_YEAR
        # amortized repayment formula:
        numerator = INTEREST_RATE * (1 + INTEREST_RATE) ** duration_steps
        denominator = (1 + INTEREST_RATE) ** duration_steps - 1
        repayment_per_step = mortgage_loan * numerator / denominator
        return repayment_per_step

    def financial_assessment(self, current_step):
        """
        Update rich/poor status, decide actions
        Returns:
            'rich', 'poor', or 'normal'
        """
        # Gather mortgages, repayments, rents for owned houses
        mortgages_remaining = [self.mortgage[h] for h in self.my_ownership if h in self.mortgage]
        if len(mortgages_remaining) == 0:
            median_mortgage = 0
        else:
            median_mortgage = median(mortgages_remaining)
        repayments = [self.repayment[h] for h in self.my_ownership if h in self.repayment]
        if len(repayments) == 0:
            median_repayment = 0
        else:
            median_repayment = median(repayments)

        rents_income = 0.0
        for h in self.my_ownership:
            if h.my_occupier is not None and h.my_occupier != self:
                rents_income += h.rent

        residual_income = self.income + rents_income * STEPS_PER_YEAR * AFFORDABILITY - sum(repayments)
        # Rich check:
        is_rich = (self.capital > SAVINGS_THRESHOLD_M * median_mortgage * (1 - LTV)) and (residual_income > median_repayment * STEPS_PER_YEAR)
        # Poor check:
        total_annual_repayment = sum(repayments) * STEPS_PER_YEAR
        total_annual_income = self.income + rents_income * STEPS_PER_YEAR
        is_poor = total_annual_repayment > EVICT_THRESHOLD_M * AFFORDABILITY * total_annual_income
        if is_rich:
            return 'rich'
        elif is_poor:
            return 'poor'
        else:
            return 'normal'

    def rent_house_assessment(self):
        """
        Check rich/poor status for rent households
        """
        if self.my_house is None:
            return 'homeless'
        c = self.capital
        p = self.my_house.price
        rent = self.my_house.rent
        y = self.income

        is_rich = c > SAVINGS_THRESHOLD_R * p * (1 - LTV)
        is_poor = rent * STEPS_PER_YEAR > EVICT_THRESHOLD_R * AFFORDABILITY * y

        if is_rich:
            return 'rich'
        elif is_poor:
            return 'poor'
        else:
            return 'normal'

    def update_income_after_rent(self):
        # Increase income to cover rent if rent = owner's repayment but tenant income insufficient
        if self.my_house is None:
            return
        a = 0
        # Owner's repayment of the house owner
        if self.my_house.my_owner is not None:
            owner = self.my_house.my_owner
            if self.my_house in owner.repayment:
                a = owner.repayment[self.my_house]

        if self.my_house.rent == a:
            self.income = max(self.income, self.my_house.rent * STEPS_PER_YEAR / AFFORDABILITY)

    def save_money(self):
        if self.type == 'mortgage':
            self.capital += self.income * SAVINGS_M / STEPS_PER_YEAR
        elif self.type == 'rent':
            self.capital += self.income * SAVINGS_R / STEPS_PER_YEAR

    def enter_market(self, market_type):
        self.market_status = market_type
        self.market_steps = 0

    def leave_market(self):
        self.market_status = None
        self.market_steps = 0
        self.cooldown_steps = COOL_DOWN_PERIOD

    def step(self, current_step, all_houses, all_households, realtors):
        """
        Main decision pipeline each timestep
        """
        # If in cooldown, decrement cooldown unless poor/homeless to rejoin market
        if self.cooldown_steps > 0:
            if self.market_status is None:
                self.cooldown_steps -= 1

        rich_status = None
        if self.type == 'mortgage':
            rich_status = self.financial_assessment(current_step)
            if rich_status == 'rich':
                # Possible investment if propensity > threshold
                if self.propensity > PROPENSITY_THRESHOLD:
                    self.enter_market('buying')
            elif rich_status == 'poor':
                # Evict and sell logic
                if len(self.my_ownership) == 1:
                    self.enter_market('selling')
                else:
                    # Evict house with lowest rental surplus or selling after no tenants
                    vacant_houses = [h for h in self.my_ownership if (h.my_occupier is None)]
                    rented_houses = [h for h in self.my_ownership if (h.my_occupier is not None)]
                    house_to_evict = None
                    # Choose house with lowest surplus rent - repayment
                    min_surplus = 1e11
                    for h in self.my_ownership:
                        owner_repayment = self.repayment.get(h, 0)
                        surplus = h.rent - owner_repayment if h.rent and owner_repayment else 0
                        if surplus < min_surplus:
                            min_surplus = surplus
                            house_to_evict = h

                    # Evict tenant if any
                    if house_to_evict:
                        if house_to_evict.my_occupier is not None:
                            tenant = house_to_evict.my_occupier
                            tenant.my_house = None
                            house_to_evict.my_occupier = None
                        house_to_evict.for_rent = True
                        house_to_evict.for_sale = False
                        house_to_evict.on_market_steps = 0
                    # After a while no tenants, sell highest profit house
                    self.enter_market('selling')
            else:
                # Normal or cooldown - do nothing or save money
                self.save_money()

        elif self.type == 'rent':
            rich_status = self.rent_house_assessment()
            if rich_status == 'rich':
                self.enter_market('buying')  # Try to become owner
            elif rich_status == 'poor':
                self.enter_market('renting')  # Try to find cheaper rent
            else:
                self.save_money()

        # Increment market step counters
        if self.market_status is not None:
            self.market_steps += 1

        # Handle homeless exit
        if self.my_house is None:
            self.homeless_steps += 1
            if self.homeless_steps > HOMELESS_PERIOD:
                self.exit_flag = True

        # Income increase per year (not enabled)
        # self.income *= (1 + WAGE_INCREASE / STEPS_PER_YEAR)

class Realtor:
    def __init__(self, idx):
        self.idx = idx
        self.locality = REALTOR_TERRITORY
        self.memory = 5  # years
        self.houses_for_sale = []
        self.houses_for_rent = []
        self.mean_price = 0
        self.mean_rent = 0

    def evaluate_prices(self):
        # Median price of houses for sale
        sale_prices = [h.price for h in self.houses_for_sale if h.price > 0]
        rent_prices = [h.rent for h in self.houses_for_rent if h.rent > 0]

        if len(sale_prices) == 0:
            self.mean_price = 0
        else:
            self.mean_price = median(sale_prices)

        if len(rent_prices) == 0:
            self.mean_rent = 0
        else:
            self.mean_rent = median(rent_prices)

        # Update prices (equation 16)
        for h in self.houses_for_sale:
            ratio = self.mean_price / h.price if h.price != 0 else 1
            if ratio > 2:
                h.price = min(2 * h.price, 2 * h.price)
            elif ratio < 0.5:
                h.price = max(h.price / 2, h.price / 2)
            else:
                h.price = self.mean_price

        # Update rents (equation 17)
        for h in self.houses_for_rent:
            ratio = self.mean_rent / h.rent if h.rent != 0 else 1
            if ratio > 2:
                h.rent = min(2 * h.rent, 2 * h.rent)
            elif ratio < 0.5:
                h.rent = max(h.rent / 2, h.rent / 2)
            else:
                h.rent = self.mean_rent

        # Decay prices and rents of unsold houses (equations 14 and 15)
        for h in self.houses_for_sale:
            h.price *= (1 - PRICE_DROP_RATE)
        for h in self.houses_for_rent:
            h.rent *= (1 - RENT_DROP_RATE)

# ================= MODEL CLASS ==================

class HousingMarketModel:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("UK Housing Market ABM")

        self.clock = pygame.time.Clock()

        # Initialize households, houses, realtors, records
        self.houses = []
        self.households = []
        self.realtors = []
        self.records = []  # Simple list of (house, price, rent, step)

        # Setup grid houses
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                self.houses.append(House((x, y)))

        # Initialize Realtors (divide grid in blocks)
        num_realtors = 4
        for i in range(num_realtors):
            self.realtors.append(Realtor(i))

        # Initialize households according to occupancy ratios and owner:tenant ratio
        total_households = int(len(self.houses) * OCCUPANCY_RATIO)
        owner_households = int(total_households * OWNERS_TO_TENANTS)
        rent_households = total_households - owner_households

        # Generate incomes by gamma distribution (shape=2 scale=MEAN_INCOME/2)
        incomes = gamma.rvs(a=2, scale=MEAN_INCOME/2, size=total_households)

        # Assign household types
        uid = 0
        for _ in range(owner_households):
            h = Household(uid, incomes[uid], 0)
            h.type = 'mortgage'
            self.households.append(h)
            uid += 1
        for _ in range(rent_households):
            h = Household(uid, incomes[uid], 0)
            h.type = 'rent'
            self.households.append(h)
            uid += 1

        # Assign households to houses randomly
        shuffled_houses = self.houses.copy()
        random.shuffle(shuffled_houses)
        for i, hh in enumerate(self.households):
            house = shuffled_houses[i]
            hh.my_house = house
            house.my_occupier = hh
            # Owner if mortgage
            if hh.type == 'mortgage':
                hh.my_ownership.add(house)
                house.my_owner = hh

        # Assign rent houses (mortgage households own rent houses)
        rent_houses = [h for h in self.houses if h.my_owner is None and h.my_occupier is not None and h.my_occupier.type == 'rent']
        mortgage_houses = [h for h in self.houses if h.my_owner is not None]

        # Randomly assign landlords (mortgage households) for rent houses
        owners_list = [hh for hh in self.households if hh.type == 'mortgage']
        for rh in rent_houses:
            landlord = random.choice(owners_list)
            rh.my_owner = landlord
            landlord.my_ownership.add(rh)

        # Initialize house prices and rents per household
        for hh in self.households:
            if hh.type == 'mortgage':
                # Calculate repayment and price
                a = hh.calculate_repayment(hh.my_house.price if hh.my_house.price > 0 else 150000)
                hh.repayment[hh.my_house] = a
                hh.mortgage[hh.my_house] = LTV * hh.my_house.price if hh.my_house.price > 0 else 135000
                hh.paid_mortgage[hh.my_house] = hh.my_house.price - hh.mortgage[hh.my_house]
                hh.my_house.price = hh.mortgage[hh.my_house] + (hh.mortgage[hh.my_house] * (1/LTV - 1))
                hh.my_house.for_sale = False
                hh.my_house.for_rent = False
            elif hh.type == 'rent':
                # Set rent as max affordable
                b = hh.income * AFFORDABILITY / STEPS_PER_YEAR
                if hh.my_house.my_owner is not None and hh.my_house.my_owner in self.households:
                    owner = hh.my_house.my_owner
                    repayment_owner = owner.repayment.get(hh.my_house, 0)
                    hh.my_house.rent = max(b, repayment_owner)
                else:
                    hh.my_house.rent = b
                hh.rent = hh.my_house.rent

        self.step_num = 0
        self.avg_price_list = []
        self.avg_rent_list = []

    def build_new_houses(self):
        # Build number proportional to total houses (construction rate)
        num_new = max(1, int(len(self.houses) * CONSTRUCTION_RATE))
        for _ in range(num_new):
            # Find empty plot in grid
            empty_plots = [h for h in self.houses if h.my_owner is None and h.my_occupier is None and not (h.for_sale or h.for_rent)]
            if empty_plots:
                h = random.choice(empty_plots)
                # New house on mortgage market for-sale
                h.for_sale = True
                # Assign initial price - arbitrary median price approx
                h.price = 150000
                h.age = 0

    def remove_demolished(self):
        # Remove houses past demolishe age
        for h in self.houses:
            if h.age_house():
                # unstuck ownerships and occupiers
                if h.my_owner is not None:
                    owner = h.my_owner
                    if h in owner.my_ownership:
                        owner.my_ownership.remove(h)
                    if h in owner.mortgage:
                        owner.mortgage.pop(h)
                    if h in owner.repayment:
                        owner.repayment.pop(h)
                    if h in owner.paid_mortgage:
                        owner.paid_mortgage.pop(h)
                if h.my_occupier is not None:
                    h.my_occupier.my_house = None
                self.houses.remove(h)

    def update_realtors(self):
        # Simple: all houses on market assigned to all realtors (for demo purposes)
        all_for_sale = [h for h in self.houses if h.for_sale]
        all_for_rent = [h for h in self.houses if h.for_rent]
        for realtor in self.realtors:
            realtor.houses_for_sale = all_for_sale
            realtor.houses_for_rent = all_for_rent
            realtor.evaluate_prices()

    def households_make_offers(self):
        # Simplified: households on buying or renting markets choose houses to bid/offers to make

        # Buyers prioritize mortgage market then buy-to-let investors (not fully separated here)
        buying_households = [hh for hh in self.households if hh.market_status == 'buying']

        # Generate sorted house list they can afford
        for hh in buying_households:
            # Consider subset of houses for sale within capital + mortgage
            affordable_houses = [h for h in self.houses if h.for_sale and h.price <= (hh.capital + hh.income * MORTGAGE_DURATION_YEARS)]
            if not affordable_houses:
                continue
            # Choose max price house in search subset
            search_houses = random.sample(affordable_houses, min(SEARCH_LENGTH, len(affordable_houses)))
            chosen_house = max(search_houses, key=lambda h: h.price)
            # Make offer:
            # Temporarily mark house sold subject to contract
            chosen_house.for_sale = False
            # Mark household to move here after chain cleared:
            hh.pending_purchase = chosen_house
            print(chosen_house)

        # Renters on rent market (simplified similarly)
        renting_households = [hh for hh in self.households if hh.market_status == 'renting']
        for hh in renting_households:
            # Houses for rent within affordability
            affordable_rentals = [h for h in self.houses if h.for_rent and h.rent <= hh.income * AFFORDABILITY / STEPS_PER_YEAR]
            if not affordable_rentals:
                continue
            search_rentals = random.sample(affordable_rentals, min(SEARCH_LENGTH, len(affordable_rentals)))
            chosen_house = max(search_rentals, key=lambda h: h.rent)
            chosen_house.for_rent = False
            hh.pending_rent = chosen_house

    def process_market_chains(self):
        # Confirm purchases after checking market chain(s)
        # Here simplified: no chains, confirm immediately
        for hh in self.households:
            if hasattr(hh, 'pending_purchase'):
                if hh.pending_purchase == None: continue
                house = hh.pending_purchase
                print(house)
                # If occupied, evict current occupant
                if house.my_occupier is not None:
                    occupant = house.my_occupier
                    occupant.my_house = None
                    occupant.market_status = 'renting'
                # Transfer ownership & occupation
                if house.my_owner:
                    # Previous owner loses house
                    prev_owner = house.my_owner
                    if house in prev_owner.my_ownership:
                        prev_owner.my_ownership.remove(house)
                    house.my_owner = hh
                    hh.my_ownership.add(house)
                else:
                    # New house
                    house.my_owner = hh
                    hh.my_ownership.add(house)

                house.my_occupier = hh
                if hh.my_house is not None:
                    # Leave old house
                    old_house = hh.my_house
                    old_house.my_occupier = None
                    if hh.type == 'mortgage':
                        old_house.for_sale = True
                hh.my_house = house
                hh.market_status = None
                hh.pending_purchase = None

                # Record transaction
                self.records.append({'house': house, 'price': house.price, 'rent': house.rent, 'step': self.step_num})

            if hasattr(hh, 'pending_rent'):
                house = hh.pending_rent
                if house.my_occupier is not None:
                    old_tenant = house.my_occupier
                    old_tenant.my_house = None
                    old_tenant.market_status = 'renting'
                house.my_occupier = hh
                hh.my_house = house
                hh.market_status = None
                hh.pending_rent = None
                # Record rent agreement
                self.records.append({'house': house, 'price': house.price, 'rent': house.rent, 'step': self.step_num})

    def remove_exiting_households(self):
        self.households = [hh for hh in self.households if not hh.exit_flag]

    def run_step(self):
        # Build houses
        self.build_new_houses()
        # Remove demolished houses
        self.remove_demolished()
        # Realtors update pricing
        self.update_realtors()

        # Households decide next moves
        for hh in self.households:
            hh.step(self.step_num, self.houses, self.households, self.realtors)

        # Households make offers
        self.households_make_offers()

        # Process market chains and finalize transactions
        self.process_market_chains()

        # Remove exited households
        self.remove_exiting_households()

        # Accumulate income, capital for households
        for hh in self.households:
            hh.save_money()

        # Update age of all houses
        for h in self.houses:
            h.age += 1

        # Collect mean price and rent for graphing
        sale_prices = [h.price for h in self.houses if h.price > 0]
        rent_prices = [h.rent for h in self.houses if h.rent > 0]

        mean_sale = np.mean(sale_prices) if sale_prices else 0
        mean_rent = np.mean(rent_prices) if rent_prices else 0

        self.avg_price_list.append(mean_sale)
        self.avg_rent_list.append(mean_rent)

        self.step_num += 1

    # ======================= Pygame Drawing ===========================

    def draw(self):
        self.screen.fill(COLOR_BG)
        # Draw grid houses
        for h in self.houses:
            x, y = h.grid_pos
            base_x = x * CELL_SIZE
            base_y = y * CELL_SIZE
            color = COLOR_BG
            if h.my_owner is not None:
                color = COLOR_MORTGAGE_HOUSE
            elif h.my_occupier is not None:
                color = COLOR_RENT_HOUSE

            pygame.draw.rect(self.screen, color, (base_x, base_y, CELL_SIZE - 1, CELL_SIZE - 1))
            # Overlay for sale/rent
            if h.for_sale:
                pygame.draw.rect(self.screen, COLOR_FOR_SALE, (base_x, base_y, CELL_SIZE - 1, 3))
            elif h.for_rent:
                pygame.draw.rect(self.screen, COLOR_FOR_RENT, (base_x, base_y, CELL_SIZE - 1, 3))

        # Draw separating line
        pygame.draw.line(self.screen, COLOR_TEXT, (GRID_WIDTH * CELL_SIZE, 0), (GRID_WIDTH * CELL_SIZE, SCREEN_HEIGHT), 2)

        # Draw graph at right
        graph_surf = self.get_graph_surface()
        self.screen.blit(graph_surf, (GRID_WIDTH * CELL_SIZE + 10, 10))

        # Display step and stats
        step_text = FONT_LARGE.render(f"Step: {self.step_num} (Quarter)", True, COLOR_TEXT)
        self.screen.blit(step_text, (GRID_WIDTH * CELL_SIZE + 10, SCREEN_HEIGHT - 70))
        if self.avg_price_list:
            price_text = FONT_SMALL.render(f"Latest Avg Sale Price: £{self.avg_price_list[-1]:,.0f}", True, COLOR_TEXT)
            self.screen.blit(price_text, (GRID_WIDTH * CELL_SIZE + 10, SCREEN_HEIGHT - 50))
        if self.avg_rent_list:
            rent_text = FONT_SMALL.render(f"Latest Avg Rent Price: £{self.avg_rent_list[-1]:,.0f}", True, COLOR_TEXT)
            self.screen.blit(rent_text, (GRID_WIDTH * CELL_SIZE + 10, SCREEN_HEIGHT - 35))

    def get_graph_surface(self):
        # Create matplotlib graph
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(self.avg_price_list, label='Avg Sale Price (£)')
        ax.plot(self.avg_rent_list, label='Avg Rent Price (£)')
        ax.set_xlabel('Time Steps (quarters)')
        ax.set_ylabel('Price (£)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()

        # Use tostring_argb for pygame (avoid tostring_rgb error)
        raw_data = canvas.tostring_argb()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(raw_data, size, "ARGB")
        plt.close(fig)
        return surf

    # ======================= Main Loop ================================

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.run_step()
            self.draw()

            pygame.display.flip()
            self.clock.tick(10)

        pygame.quit()


# ---- Running the Model -----
if __name__ == "__main__":
    COOL_DOWN_PERIOD = 1  # Quarter
    model = HousingMarketModel()
    model.run()