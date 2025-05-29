import pygame
import random
import math
import numpy as np

# --- Global Parameters (from your table and description) ---

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
GRID_SIZE = 30  # 30x30 grid, for example
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

STEPS_PER_YEAR = 4  # 3 months per step
TIME_STEP_DURATION_SEC = 1.0  # seconds per model step in UI

# Financial/Model Parameters (taken or derived from your description)
INTEREST_RATE_ANNUAL = 0.04  # 4% annual interest
INTEREST_RATE_STEP = INTEREST_RATE_ANNUAL / STEPS_PER_YEAR

MAX_MORTGAGE_DURATION_YEARS = 25
LOAN_TO_VALUE = 0.75

AFFORDABILITY = 0.3  # max proportion of income on housing (alpha)
SAVINGS_THRESHOLD_M = 1.5  # omega from eq 7
EVICT_THRESHOLD_M = 0.9  # beta from eq 8
SAVINGS_THRESHOLD_R = 1.3  # lambda eq 10
EVICT_THRESHOLD_R = 1.0  # gamma eq 11

PRICE_DROP_RATE = 0.98  # rho for decay eq 14 & 15

MAX_HOUSE_AGE = 100

# Propensity threshold (for rich households to invest)
PROPENSITY_THRESHOLD = 0.5

# Housing market parameters
SEARCH_LENGTH = 5  # number of houses a household considers when buying

# Income distribution parameters (gamma approx for UK income)
INCOME_SHAPE = 2.0
INCOME_SCALE = 30000

# Misc
MAX_HOUSEHOLDS = 500
INITIAL_OCCUPANCY_RATIO = 0.9
OWNERS_TO_TENANTS_RATIO = 0.6

CONSTRUCTION_RATE_PER_STEP = 0.01
ENTRY_RATE_PER_STEP = 0.005
EXIT_RATE_PER_STEP = 0.005

MAX_HOMELESS_PERIOD = 8  # steps
MAX_ON_MARKET_PERIOD = 6
COOL_DOWN_PERIOD = 2

# Colors for pygame
COLOR_BG = (20,20,30)
COLOR_MORTGAGE = (70,130,180)  # Steel blue (mortgage owners)
COLOR_RENT = (50,205,50)       # Lime green (rent households)
COLOR_VACANT = (100,100,100)   # Grey
COLOR_SOLD = (255, 215, 0)     # Gold
COLOR_TEXT = (230,230,230)

# ------------------------------
# Helper Functions
# ------------------------------

def mortgage_max_amount(repayment, i_step, d_years, steps_per_year):
    """Calculate maximum mortgage using amortization formula (Eq 3)."""
    n_periods = d_years * steps_per_year
    return repayment / i_step * (1 - (1 + i_step) ** (-n_periods))

def gamma_income(shape=INCOME_SHAPE, scale=INCOME_SCALE):
    return max(1000, np.random.gamma(shape, scale))

def choose_weighted(choices):
    """choices: list of (item, weight). Returns item based on weights."""
    total = sum(w for _, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for item, weight in choices:
        if upto + weight >= r:
            return item
        upto += weight
    return choices[-1][0]

def median(lst):
    if not lst:
        return 0
    s = sorted(lst)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid-1] + s[mid]) / 2
    else:
        return s[mid]

# ------------------------------
# Agent Classes
# ------------------------------

class House:
    def __init__(self, x, y, house_type='mortgage'):
        self.x = x
        self.y = y
        self.type = house_type  # 'mortgage' or 'rent'
        self.owner = None  # Household
        self.occupier = None  # Household
        self.price = 0.0
        self.rent = 0.0
        self.for_sale = False
        self.for_rent = False
        self.age = 0
        self.demolish_age = random.expovariate(1/50) + 50  # Mean 100 steps
        self.on_market_period = 0

    def step(self):
        self.age += 1

    def is_demolished(self):
        return self.age > self.demolish_age

class Record:
    def __init__(self, house, time_step, price, rent):
        self.house = house
        self.time = time_step
        self.price = price
        self.rent = rent

class Realtor:
    def __init__(self, id_, x, y, territory=5, memory=6):
        self.id = id_
        self.x = x
        self.y = y
        self.territory = territory
        self.memory = memory  # number of previous years aware of
        self.records = []  # Records known to realtor

    def houses_in_locality(self, houses_grid):
        houses_local = []
        for i in range(max(0, self.x - self.territory), min(GRID_SIZE, self.x + self.territory +1)):
            for j in range(max(0, self.y - self.territory), min(GRID_SIZE, self.y + self.territory +1)):
                house = houses_grid[j][i]
                if house is not None:
                    houses_local.append(house)
        return houses_local

    def update_price_rent(self, house, median_price, median_rent):
        # Equations 16 and 17
        if median_price == 0:
            median_price = house.price
        if median_rent == 0:
            median_rent = house.rent

        ratio_p = median_price / max(house.price,1e-3)
        if ratio_p > 2:
            house.price = 2 * house.price
        elif ratio_p < 0.5:
            house.price = house.price / 2
        else:
            house.price = median_price

        ratio_r = median_rent / max(house.rent, 1e-3)
        if ratio_r > 2:
            house.rent = 2 * house.rent
        elif ratio_r < 0.5:
            house.rent = house.rent / 2
        else:
            house.rent = median_rent

    def decay_prices(self, house):
        # Eq.14 and Eq.15
        house.price *= PRICE_DROP_RATE
        house.rent *= PRICE_DROP_RATE

    def record_transaction(self, record):
        self.records.append(record)

class Household:
    def __init__(self, id_, income=None, house_type=None):
        self.id = id_
        self.income = income if income else gamma_income()
        self.propensity = random.uniform(0,1)  # willingness to invest (=w_i)
        self.type = house_type  # 'mortgage' or 'rent'
        self.capital = 0.0  # initialized later: c = y * eta
        self.my_ownership = []  # Houses owned (list)
        self.my_house = None  # House currently occupied
        self.mortgage = {}  # house_id -> remaining mortgage
        self.paid_mortgage = {}  # house_id -> paid mortgage amount
        self.repayment = {}  # house_id -> repayment per step
        self.rate_duration = MAX_MORTGAGE_DURATION_YEARS  # fixed for now
        self.rent_amount = 0.0
        self.market = None  # 'mortgage', 'rental', 'btl' or None
        self.homeless_period = 0
        self.on_market_period = 0
        self.cool_down = 0
        self.exiting = False

    def initialize_finances(self):
        # Capital calculation: c = y * eta (eta random)
        eta = random.uniform(0.5, 3.0)
        self.capital = self.income * eta

    def assign_house(self, house):
        self.my_house = house
        if house.owner != self:
            house.owner = self
            self.my_ownership.append(house)
        house.occupier = self

    def calculate_max_repayment(self):
        return self.income * AFFORDABILITY / STEPS_PER_YEAR

    def calculate_max_mortgage(self):
        a = self.calculate_max_repayment()
        return mortgage_max_amount(a, INTEREST_RATE_STEP, MAX_MORTGAGE_DURATION_YEARS, STEPS_PER_YEAR)

    def set_mortgage_for_house(self, house):
        price = house.price
        mortgage_amount = LOAN_TO_VALUE * price
        self.mortgage[house] = mortgage_amount
        self.paid_mortgage[house] = 0.0
        repayment = self.calculate_max_repayment()
        self.repayment[house] = repayment

    def set_rent_for_house(self, house, tenant):
        # Rent as max of tenant willingness and owner's repayment
        b = tenant.calculate_max_repayment()
        a_owner = self.repayment.get(house, 0)
        if b > a_owner:
            rent = b
        else:
            rent = a_owner
            tenant.income = max(tenant.income, rent * STEPS_PER_YEAR / AFFORDABILITY)
        house.rent = rent
        self.rent_amount = rent
        house.for_rent = True
        tenant.rent_amount = rent
        house.price = house.price  # Keep price for ownership continuity

    def is_relatively_rich_mortgage(self):
        if not self.my_ownership:
            return False
        median_mortgage = median([self.mortgage.get(h, 0) for h in self.my_ownership])
        median_repayment = median([self.repayment.get(h, 0) for h in self.my_ownership])
        condition_1 = self.capital > SAVINGS_THRESHOLD_M * median_mortgage * (1 - LOAN_TO_VALUE)
        rental_income = sum([h.rent for h in self.my_ownership if h.for_rent]) * STEPS_PER_YEAR * AFFORDABILITY
        condition_2 = (self.income + rental_income - sum([self.repayment.get(h,0) for h in self.my_ownership])) > median_repayment * STEPS_PER_YEAR
        return condition_1 and condition_2

    def is_relatively_poor_mortgage(self):
        repayment_total = sum([self.repayment.get(h, 0) for h in self.my_ownership]) * STEPS_PER_YEAR
        rent_total = sum([h.rent for h in self.my_ownership]) * STEPS_PER_YEAR
        income_total = self.income
        return repayment_total > EVICT_THRESHOLD_M * AFFORDABILITY * (income_total + rent_total)

    def is_relatively_rich_rent(self):
        if not self.my_house:
            return False
        price = self.my_house.price
        return self.capital > SAVINGS_THRESHOLD_R * price * (1 - LOAN_TO_VALUE)

    def is_relatively_poor_rent(self):
        if not self.my_house:
            return False
        rent_total = self.my_house.rent * STEPS_PER_YEAR
        income_allow = EVICT_THRESHOLD_R * AFFORDABILITY * self.income
        return rent_total > income_allow

    def select_house_to_sell(self):
        vacant = [h for h in self.my_ownership if (not h.for_rent and not h.occupier)]
        if vacant:
            profits = [(h.price - self.mortgage.get(h, 0), h) for h in vacant]
            return max(profits, key=lambda x: x[0])[1]
        rented = [h for h in self.my_ownership if h.for_rent]
        if rented:
            profits = [(h.price - self.mortgage.get(h, 0), h) for h in rented]
            return max(profits, key=lambda x: x[0])[1]
        return None

    def select_house_to_evict(self):
        candidates = []
        for h in self.my_ownership:
            if h.for_rent and h.occupier:
                surplus = h.rent - self.repayment.get(h, 0)
                candidates.append((surplus, h))
        if candidates:
            # Lowest surplus
            return min(candidates, key=lambda x: x[0])[1]
        return None

    def update_on_market_period(self):
        # Increase time on market counters
        self.on_market_period +=1

    def enter_market(self, market_type):
        self.market = market_type
        self.on_market_period = 0

    def exit_market(self):
        self.market = None
        self.on_market_period = 0
        self.cool_down = COOL_DOWN_PERIOD

    def can_reenter_market(self):
        return self.cool_down <= 0

    def step(self, houses_grid, realtor_list, current_step, all_households):
        # Financial check and market entry/exit logic

        # Homeless check
        if self.my_house is None:
            self.homeless_period += 1
            if self.homeless_period > MAX_HOMELESS_PERIOD:
                self.exiting = True
            else:
                # Try to enter rental market if possible
                if self.can_reenter_market():
                    self.enter_market('rental')
            return

        else:
            self.homeless_period = 0

        # Cool down state update
        if self.cool_down > 0:
            self.cool_down -=1

        if self.market == 'cool_down':
            if self.can_reenter_market():
                self.market = None

        # Mortgage household logic
        if self.type == 'mortgage':
            if self.is_relatively_rich_mortgage():
                # Possibly invest (buy more houses)
                if self.propensity > PROPENSITY_THRESHOLD:
                    self.enter_market('btl')  # buy-to-let market

            if self.is_relatively_poor_mortgage():
                # Eviction or sell logic
                if len(self.my_ownership) <= 1:
                    if not self.market == 'mortgage':  # offer home for sale
                        if self.my_house:
                            self.my_house.for_sale = True
                        self.enter_market('mortgage')
                else:
                    to_evict = self.select_house_to_evict()
                    if to_evict:
                        to_evict.for_rent = True
                        to_evict.for_sale = False
                        if to_evict.occupier:
                            to_evict.occupier.my_house = None
                            to_evict.occupier = None
                        to_evict.on_market_period = 0
                    else:
                        to_sell = self.select_house_to_sell()
                        if to_sell:
                            to_sell.for_sale = True
                            self.enter_market('mortgage')
        # Rent household logic
        elif self.type == 'rent':
            if self.is_relatively_rich_rent():
                self.enter_market('mortgage')  # aim to buy home
            elif self.is_relatively_poor_rent():
                self.enter_market('rental')  # find cheaper rent

# ------------------------------
# Environment/Model
# ------------------------------

class ABMModel:
    def __init__(self):
        self.time_step = 0
        # Initialize Houses Grid
        self.houses_grid = [[None for _ in range(GRID_SIZE)] for __ in range(GRID_SIZE)]

        # Create houses
        self.houses = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # Randomly assign house type with 70% mortgage, 30% rent approx
                house_type = 'mortgage' if random.random() < 0.7 else 'rent'
                house = House(x, y, house_type)
                # Price and rent placeholders; will update after households assigned
                house.price = random.uniform(150_000, 350_000)
                house.rent = random.uniform(800, 1500)
                self.houses_grid[y][x] = house
                self.houses.append(house)

        # Initialize agents
        self.households = []
        num_households = int(MAX_HOUSEHOLDS * INITIAL_OCCUPANCY_RATIO)

        # Owners vs tenants ratio controls how many mortgage vs rent households
        num_own = int(num_households * OWNERS_TO_TENANTS_RATIO)
        num_rent = num_households - num_own

        # Create mortgage households (owners)
        for i in range(num_own):
            hh = Household(i, house_type='mortgage')
            hh.initialize_finances()
            self.households.append(hh)

        # Create rent households (tenants)
        for i in range(num_own, num_own + num_rent):
            hh = Household(i, house_type='rent')
            hh.initialize_finances()
            self.households.append(hh)

        # Assign houses to households randomly (based on availability and type)
        mortgage_houses = [h for h in self.houses if h.type == 'mortgage']
        rent_houses = [h for h in self.houses if h.type == 'rent']

        # Assign mortgage households to mortgage houses
        random.shuffle(mortgage_houses)
        mortgage_hindex = 0
        for hh in self.households:
            if hh.type == 'mortgage':
                if mortgage_hindex < len(mortgage_houses):
                    house = mortgage_houses[mortgage_hindex]
                    mortgage_hindex += 1
                    hh.assign_house(house)
                    house.for_sale = False
                    house.for_rent = False
                else:
                    break

        # Assign rent households to rent houses and landlords
        random.shuffle(rent_houses)
        rent_hindex = 0
        mortgage_owners = [hh for hh in self.households if hh.type == 'mortgage']
        for hh in self.households:
            if hh.type == 'rent':
                if rent_hindex < len(rent_houses):
                    rent_house = rent_houses[rent_hindex]
                    rent_hindex += 1
                    landlord = random.choice(mortgage_owners)
                    rent_house.owner = landlord
                    landlord.my_ownership.append(rent_house)
                    hh.assign_house(rent_house)
                    rent_house.for_rent = False
                    rent_house.for_sale = False
                else:
                    break

        # Initialize mortgages, rents, prices for owned houses
        for hh in self.households:
            if hh.type == 'mortgage':
                for h in hh.my_ownership:
                    # Price from Eq. 5, initial approximation
                    max_rep = hh.calculate_max_repayment()
                    max_mortgage = mortgage_max_amount(max_rep, INTEREST_RATE_STEP, MAX_MORTGAGE_DURATION_YEARS, STEPS_PER_YEAR)
                    deposit = max_mortgage * (1/LOAN_TO_VALUE -1)
                    h.price = max_mortgage + deposit
                    # mortgage and repayment per house
                    hh.set_mortgage_for_house(h)
            else:
                # rent households: set rent using Eq. 6 logic
                if hh.my_house:
                    landlord = hh.my_house.owner
                    if landlord:
                        landlord.set_rent_for_house(hh.my_house, hh)

        # Realtors init (assign to grid evenly)
        self.realtors = []
        num_realtors = 10
        spacing = GRID_SIZE // int(math.sqrt(num_realtors))
        idx = 0
        for i in range(0, GRID_SIZE, spacing):
            for j in range(0, GRID_SIZE, spacing):
                realtor = Realtor(idx, i, j)
                self.realtors.append(realtor)
                idx +=1
                if idx >= num_realtors:
                    break
            if idx >= num_realtors:
                break

        # Records stored globally
        self.records = []

    def step(self):
        # 1. Households financial status and decisions
        for hh in self.households:
            if hh.exiting:
                # Remove household and free up house
                if hh.my_house:
                    hh.my_house.occupier = None
                    hh.my_house.for_sale = True
                self.households.remove(hh)
                continue
            hh.step(self.houses_grid, self.realtors, self.time_step, self.households)

        # 2. Realtors update prices and rents for houses on market
        for realtor in self.realtors:
            local_houses = realtor.houses_in_locality(self.houses_grid)
            houses_for_sale = [h for h in local_houses if h.for_sale]
            houses_for_rent = [h for h in local_houses if h.for_rent]

            median_prices = median([h.price for h in houses_for_sale]) if houses_for_sale else 0
            median_rents = median([h.rent for h in houses_for_rent]) if houses_for_rent else 0

            for house in houses_for_sale:
                realtor.update_price_rent(house, median_prices, house.rent)
                realtor.decay_prices(house)

            for house in houses_for_rent:
                realtor.update_price_rent(house, house.price, median_rents)
                realtor.decay_prices(house)

        # 3. Market transactions
        # Housing market priorities: 'mortgage' market households buy first, then 'btl'

        # Collect buyers by market type
        mortgage_buyers = [hh for hh in self.households if hh.market == 'mortgage']
        btl_buyers = [hh for hh in self.households if hh.market == 'btl']
        rental_seekers = [hh for hh in self.households if hh.market == 'rental']

        # Helper function for market buying attempt
        def try_buy(hh, candidate_houses):
            affordable = [h for h in candidate_houses if h.price <= hh.income * 3]  # Simple affordability filter
            # Limit search length
            affordable = affordable[:SEARCH_LENGTH]
            if not affordable:
                return False
            # Choose highest price
            house_to_buy = max(affordable, key=lambda x: x.price)
            if house_to_buy.for_sale:
                # Complete market chain check (simplified: no chain, immediate purchase)
                house_to_buy.for_sale = False
                prev_owner = house_to_buy.owner
                if prev_owner:
                    if house_to_buy in prev_owner.my_ownership:
                        prev_owner.my_ownership.remove(house_to_buy)
                        if house_to_buy.occupier == prev_owner:
                            prev_owner.my_house = None
                house_to_buy.owner = hh
                hh.my_ownership.append(house_to_buy)
                hh.assign_house(house_to_buy)
                # Update mortgage for buyer
                hh.set_mortgage_for_house(house_to_buy)
                hh.exit_market()
                # Record transaction
                record = Record(house_to_buy, self.time_step, house_to_buy.price, house_to_buy.rent)
                self.records.append(record)
                return True
            return False

        # Run market transactions
        for buyer in mortgage_buyers:
            houses_on_sale = [h for h in self.houses if h.for_sale]
            try_buy(buyer, houses_on_sale)

        for buyer in btl_buyers:
            houses_on_sale = [h for h in self.houses if h.for_sale]
            try_buy(buyer, houses_on_sale)

        for seeker in rental_seekers:
            houses_for_rent = [h for h in self.houses if h.for_rent]
            affordable = [h for h in houses_for_rent if h.rent <= seeker.income * AFFORDABILITY / STEPS_PER_YEAR]
            if affordable:
                chosen_house = random.choice(affordable)
                # Rent transaction
                prev_tenant = chosen_house.occupier
                if prev_tenant:
                    prev_tenant.my_house = None
                chosen_house.occupier = seeker
                seeker.assign_house(chosen_house)
                chosen_house.for_rent = False
                seeker.exit_market()
                # Record rental
                record = Record(chosen_house, self.time_step, chosen_house.price, chosen_house.rent)
                self.records.append(record)

        # 4. Income and capital accumulation
        for hh in self.households:
            hh.capital += hh.income / STEPS_PER_YEAR  # income adds per step (quarterly approx)
            # Mortgage repayments reduce capital
            repayment_sum = sum(hh.repayment.values())
            hh.capital -= repayment_sum

        # 5. Construct new houses
        num_new_houses = int(len(self.houses) * CONSTRUCTION_RATE_PER_STEP)
        for _ in range(num_new_houses):
            x = random.randint(0, GRID_SIZE-1)
            y = random.randint(0, GRID_SIZE-1)
            if self.houses_grid[y][x] is None:
                house_type = 'mortgage' if random.random() < 0.7 else 'rent'
                new_house = House(x, y, house_type)
                new_house.price = random.uniform(150_000, 350_000)
                new_house.rent = random.uniform(800, 1500)
                self.houses.append(new_house)
                self.houses_grid[y][x] = new_house

        # 6. Demolish old houses
        for house in self.houses[:]:
            if house.is_demolished():
                if house.owner:
                    if house in house.owner.my_ownership:
                        house.owner.my_ownership.remove(house)
                    if house.occupier == house.owner:
                        house.owner.my_house = None
                if house.occupier:
                    house.occupier.my_house = None
                self.houses_grid[house.y][house.x] = None
                self.houses.remove(house)

        # 7. Immigration / Emigration
        # Exit households randomly
        exiting = [hh for hh in self.households if random.random() < EXIT_RATE_PER_STEP]
        for hh in exiting:
            if hh.my_house:
                hh.my_house.occupier = None
            self.households.remove(hh)

        # Enter households randomly
        num_entries = int(MAX_HOUSEHOLDS * ENTRY_RATE_PER_STEP)
        for _ in range(num_entries):
            income = gamma_income()
            hh_type = 'mortgage' if random.random() < OWNERS_TO_TENANTS_RATIO else 'rent'
            new_hh = Household(len(self.households), income, hh_type)
            new_hh.initialize_finances()
            if hh_type == 'mortgage':
                new_hh.enter_market('mortgage')
            else:
                new_hh.enter_market('rental')
            self.households.append(new_hh)

        self.time_step += 1

# ------------------------------
# Visualization with Pygame
# ------------------------------

pygame.init()
FONT = pygame.font.SysFont("Arial", 14)

class ABMView:
    def __init__(self, model):
        self.model = model
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Housing Market ABM")

    def draw(self):
        self.screen.fill(COLOR_BG)
        # Draw houses grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                house = self.model.houses_grid[y][x]
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1)
                if house:
                    color = COLOR_VACANT
                    if house.occupier:
                        if house.occupier.type == 'mortgage':
                            color = COLOR_MORTGAGE
                        else:
                            color = COLOR_RENT
                    pygame.draw.rect(self.screen, color, rect)
                    if house.for_sale:
                        pygame.draw.rect(self.screen, COLOR_SOLD, rect, 2)
                    if house.for_rent:
                        pygame.draw.rect(self.screen, (255,100,100), rect, 2)
                else:
                    pygame.draw.rect(self.screen, (10,10,10), rect)

        # Draw HUD
        info_lines = [
            f"Time Step: {self.model.time_step} (Year: {self.model.time_step / STEPS_PER_YEAR:.2f})",
            f"Households: {len(self.model.households)}",
            f"Houses: {len(self.model.houses)}",
            f"Houses for sale: {sum(h.for_sale for h in self.model.houses)}",
            f"Houses for rent: {sum(h.for_rent for h in self.model.houses)}",
        ]
        for i,line in enumerate(info_lines):
            text = FONT.render(line, True, COLOR_TEXT)
            self.screen.blit(text, (10, SCREEN_HEIGHT - 20*(len(info_lines)-i)))

        pygame.display.flip()

# ------------------------------
# Main Loop
# ------------------------------

def main():
    model = ABMModel()
    view = ABMView(model)
    clock = pygame.time.Clock()
    running = True
    last_step_time = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Advance model by one step per TIME_STEP_DURATION_SEC
        now = pygame.time.get_ticks()
        if now - last_step_time > TIME_STEP_DURATION_SEC*1000:
            model.step()
            last_step_time = now

        view.draw()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()