import pygame
import numpy as np
import random
import math
from statistics import mean
import matplotlib.pyplot as plt
from pygame.locals import *
from matplotlib.backends.backend_agg import FigureCanvasAgg

# === Global parameters matching your specification ===
STEPS_PER_YEAR = 4  # 3 months per step

INTEREST_RATE_ANNUAL = 0.04
INTEREST_RATE_STEP = INTEREST_RATE_ANNUAL / STEPS_PER_YEAR
LTV = 0.8
MORTGAGE_DURATION_YEARS = 25
AFFORDABILITY_RATIO = 0.3

SAVINGS_THRESHOLD_M = 1.2  # omega
EVICT_THRESHOLD_M = 0.6    # beta
SAVINGS_THRESHOLD_R = 1.0  # lambda
EVICT_THRESHOLD_R = 1.0    # gamma

PROPENSITY_THRESHOLD = 0.5
INVESTOR_RATIO = 0.3

PRICE_DROP_RATE = 0.98
RENT_DROP_RATE = 0.98

HOMELESS_PERIOD = STEPS_PER_YEAR * 5      # 5 years
ON_MARKET_PERIOD = STEPS_PER_YEAR * 3     # 3 years
COOL_DOWN_PERIOD = STEPS_PER_YEAR          # 1 year

SEARCH_LENGTH = 10
REALTOR_TERRITORY = 5

GRID_SIZE = 30
CELL_SIZE = 15
SCREEN_WIDTH = GRID_SIZE*CELL_SIZE + 300  # extra right panel for graph
SCREEN_HEIGHT = GRID_SIZE*CELL_SIZE

FONT_SIZE = 20

# === Financial helper function ===
def max_mortgage_from_repayment(a, interest_rate_step, duration_years, steps_per_year):
    """Eq (3)"""
    n = duration_years * steps_per_year
    if interest_rate_step == 0:
        return a * n
    return a / interest_rate_step * (1 - (1 + interest_rate_step) ** (-n))


# === Agents and Entities ===

class House:
    def __init__(self, x,y, house_id):
        self.id = house_id
        self.x = x
        self.y = y
        self.price = 0.0
        self.rent = 0.0
        self.owner = None          # Household object
        self.occupier = None       # Household object
        self.for_sale = False
        self.for_rent = False
        self.age = 0
        self.demolish_age = random.expovariate(1/(100*STEPS_PER_YEAR))
        self.type = None           # 'mortgage' or 'rent'

    def step_age(self):
        self.age += 1
        if self.age > self.demolish_age:
            return True
        return False

class Household:
    def __init__(self, hh_id, hh_type, income, propensity):
        self.id = hh_id
        self.type = hh_type  # 'mortgage' or 'rent'
        self.income = income
        self.propensity = propensity
        self.capital = 0.0

        self.my_ownership = []
        self.my_house = None

        self.mortgages = {}        # house_id -> mortgage amount remaining
        self.paid_mortgages = {}   # house_id -> mortgage paid off amount
        self.repayments = {}       # house_id -> repayment per step

        self.rate_duration = 0     # ignored for now
        self.rent_payment = 0.0
        self.market = None         # 'mortgage' or 'rent' or None

        self.on_market_since = None
        self.cool_down_until = None
        self.homeless_since = None

    def init_capital(self):
        eta = random.uniform(0.5, 3.0)  # capital-to-income ratio randomization
        self.capital = self.income * eta

    def affordability_repayment(self):
        # Eq (2)
        return (self.income * AFFORDABILITY_RATIO) / STEPS_PER_YEAR

    def max_mortgage(self):
        # Eq (3)
        a = self.affordability_repayment()
        return max_mortgage_from_repayment(a, INTEREST_RATE_STEP, MORTGAGE_DURATION_YEARS, STEPS_PER_YEAR)

    def deposit(self, mortgage_amount):
        # Eq (3b)
        return mortgage_amount * (1 / LTV - 1)

    def set_house_prices_and_rents(self):
        """
        Assign initial prices/rents per eq (5),(6), and tenant income update (eq n)
        """
        for h in self.my_ownership:
            if h.type == 'mortgage':
                a = self.affordability_repayment()
                M = self.max_mortgage()
                D = self.deposit(M)
                h.price = M + D
                h.rent = 0.0
                self.repayments[h.id] = a
                self.mortgages[h.id] = LTV * h.price
                self.paid_mortgages[h.id] = 0.0
            elif h.type == 'rent':
                tenant = h.occupier
                if tenant is None:
                    continue
                b = tenant.affordability_repayment()
                a = self.affordability_repayment()
                if b > a:
                    h.rent = b
                else:
                    h.rent = a
                    # tenant income update eq (n)
                    tenant.income = (h.rent * STEPS_PER_YEAR) / AFFORDABILITY_RATIO

    def is_relatively_rich(self, median_mortgage, median_repayment):
        # Eq (7)
        if len(self.mortgages) == 0:
            return False
        cond1 = self.capital > SAVINGS_THRESHOLD_M * median_mortgage * (1-LTV)
        rent_income = sum([h.rent for h in self.my_ownership if h.rent > 0])
        residual_income = self.income + (rent_income * STEPS_PER_YEAR * AFFORDABILITY_RATIO) - sum(self.repayments.values())
        cond2 = residual_income > median_repayment * STEPS_PER_YEAR
        return cond1 and cond2

    def is_relatively_poor(self):
        # Eq (8)
        total_repayment_annual = sum(self.repayments.values()) * STEPS_PER_YEAR
        rent_income_annual = sum([h.rent for h in self.my_ownership if h.rent > 0]) * STEPS_PER_YEAR
        income_annual = self.income + rent_income_annual
        return total_repayment_annual > EVICT_THRESHOLD_M * AFFORDABILITY_RATIO * income_annual

    def rent_house_relatively_rich(self):
        # Eq (10)
        if self.my_house is None:
            return False
        return self.capital > SAVINGS_THRESHOLD_R * self.my_house.price * (1-LTV)

    def rent_house_relatively_poor(self):
        # Eq (11)
        if self.my_house is None:
            return False
        rent_annual = self.my_house.rent * STEPS_PER_YEAR
        return rent_annual > EVICT_THRESHOLD_R * AFFORDABILITY_RATIO * self.income

class Realtor:
    def __init__(self, realtor_id, x, y):
        self.id = realtor_id
        self.x = x
        self.y = y
        self.locality_radius = REALTOR_TERRITORY

    def locality_houses(self, houses):
        result = []
        for h in houses:
            dist = math.sqrt((h.x - self.x)**2 + (h.y - self.y)**2)
            if dist <= self.locality_radius:
                result.append(h)
        return result

    def median_prices_rents(self, houses):
        prices = [h.price for h in houses if h.for_sale]
        rents = [h.rent for h in houses if h.for_rent]
        median_p = np.median(prices) if prices else 1.0
        median_r = np.median(rents) if rents else 1.0
        return median_p, median_r

    def update_prices_rents(self, houses):
        median_p, median_r = self.median_prices_rents(houses)
        for h in houses:
            if h.for_sale:
                ratio = median_p / (h.price if h.price>0 else 1)
                if ratio > 2:
                    h.price = 2 * h.price
                elif ratio < 0.5:
                    h.price = h.price / 2
                else:
                    h.price = median_p
            if h.for_rent:
                ratio = median_r / (h.rent if h.rent>0 else 1)
                if ratio > 2:
                    h.rent = 2 * h.rent
                elif ratio < 0.5:
                    h.rent = h.rent / 2
                else:
                    h.rent = median_r

    def decay_prices_rents(self, houses):
        for h in houses:
            if h.for_sale:
                h.price *= PRICE_DROP_RATE
                if h.price < 1:
                    h.price = 1.0
            if h.for_rent:
                h.rent *= RENT_DROP_RATE
                if h.rent < 1:
                    h.rent = 1.0

# === The Model ===

class HousingMarketModel:
    def __init__(self):
        self.time_step = 0
        self.houses = []
        self.households = []
        self.realtors = []
        self.mortgage_market = []
        self.rent_market = []
        self.transaction_records = []
        self.mean_prices_over_time = []
        self.mean_rents_over_time = []

        self.init_houses()
        self.init_households()
        self.init_realtors()
        self.assign_initial_houses()

    def init_houses(self):
        house_id = 0
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                house = House(x, y, house_id)
                house.type = 'mortgage' if random.random() < 0.7 else 'rent'
                self.houses.append(house)
                house_id += 1

    def init_households(self):
        num_households = int(0.85 * len(self.houses))
        for hh_id in range(num_households):
            hh_type = 'mortgage' if random.random() < 0.7 else 'rent'
            income = np.random.gamma(2., 25000)
            propensity = random.random()
            hh = Household(hh_id, hh_type, income, propensity)
            hh.init_capital()
            self.households.append(hh)

    def assign_initial_houses(self):
        unassigned_houses = self.houses.copy()
        random.shuffle(unassigned_houses)
        for hh in self.households:
            if len(unassigned_houses) == 0:
                break
            h = unassigned_houses.pop()
            h.occupier = hh
            hh.my_house = h
            if hh.type == 'mortgage':
                h.owner = hh
                hh.my_ownership.append(h)
        # Landlords assign for rent houses
        rent_houses = [h for h in self.houses if h.type == 'rent' and h.occupier]
        mortgage_owners = [hh for hh in self.households if hh.type == 'mortgage']
        for h in rent_houses:
            h.owner = random.choice(mortgage_owners)
            h.owner.my_ownership.append(h)
        for hh in self.households:
            hh.set_house_prices_and_rents()

    def init_realtors(self):
        for i in range(10):
            x = random.randint(0, GRID_SIZE-1)
            y = random.randint(0, GRID_SIZE-1)
            self.realtors.append(Realtor(i,x,y))

    def step(self):
        self.time_step += 1

        # Demolish old houses
        for house in self.houses[:]:
            if house.step_age():
                if house.owner and house in house.owner.my_ownership:
                    house.owner.my_ownership.remove(house)
                if house.occupier:
                    house.occupier.my_house = None
                self.houses.remove(house)

        # New houses
        num_new = int(len(self.houses) * (0.01 / STEPS_PER_YEAR))
        for _ in range(num_new):
            new_id = max(h.id for h in self.houses)+1
            x = random.randint(0, GRID_SIZE-1)
            y = random.randint(0, GRID_SIZE-1)
            new_house = House(x, y, new_id)
            new_house.type = 'mortgage'
            self.houses.append(new_house)

        # Median mortgage and repayment for mortgage households
        all_mortgages = [m for hh in self.households for m in hh.mortgages.values()]
        all_repayments = [r for hh in self.households for r in hh.repayments.values()]

        median_mtg = np.median(all_mortgages) if all_mortgages else 0
        median_rep = np.median(all_repayments) if all_repayments else 0

        self.mortgage_market.clear()
        self.rent_market.clear()

        # Households financial status and market participation
        for hh in self.households:
            if hh.type == 'mortgage':
                rich = hh.is_relatively_rich(median_mtg, median_rep)
                poor = hh.is_relatively_poor()
                if rich and hh.propensity > PROPENSITY_THRESHOLD:
                    self.mortgage_market.append(hh)
                    hh.market = 'mortgage'
                if poor:
                    if len(hh.my_ownership) <= 1:
                        if hh.my_house and not hh.my_house.for_sale:
                            hh.my_house.for_sale = True
                            self.mortgage_market.append(hh)
                            hh.market = 'mortgage'
                    else:
                        rents_surplus = [(h.rent - hh.repayments.get(h.id,0), h) for h in hh.my_ownership]
                        if rents_surplus:
                            eviction_house = min(rents_surplus, key=lambda x: x[0])[1]
                            if not eviction_house.for_rent:
                                eviction_house.for_rent = True
                                if eviction_house.occupier:
                                    eviction_house.occupier.my_house = None
                                    eviction_house.occupier = None
                                self.rent_market.append(hh)
                                hh.market = 'rent'
            elif hh.type == 'rent':
                rich = hh.rent_house_relatively_rich()
                poor = hh.rent_house_relatively_poor()
                if rich:
                    self.mortgage_market.append(hh)
                    hh.market = 'mortgage'
                elif poor:
                    self.rent_market.append(hh)
                    hh.market = 'rent'

        # Realtors update prices and rents
        for realtor in self.realtors:
            loc_houses = realtor.locality_houses(self.houses)
            realtor.update_prices_rents(loc_houses)
            realtor.decay_prices_rents(loc_houses)

        # Market transactions - mortgage buyers prioritized
        purchased = {}
        for buyer in self.mortgage_market:
            budget = buyer.capital + sum(buyer.mortgages.values())
            affordable_houses = [h for h in self.houses if h.for_sale and h.price <= budget and h.id not in purchased]
            if not affordable_houses:
                continue
            sample_set = random.sample(affordable_houses, min(len(affordable_houses), SEARCH_LENGTH))
            chosen_house = max(sample_set, key=lambda h: h.price)
            purchased[chosen_house.id] = buyer

        # Finalize purchases and update ownership
        for house_id, buyer in purchased.items():
            house = next(h for h in self.houses if h.id == house_id)
            old_owner = house.owner
            if old_owner:
                if house in old_owner.my_ownership:
                    old_owner.my_ownership.remove(house)
                old_owner.mortgages.pop(house.id, None)
                old_owner.repayments.pop(house.id, None)
                old_owner.paid_mortgages.pop(house.id, None)
            house.owner = buyer
            buyer.my_ownership.append(house)
            buyer.my_house = house
            buyer.type = 'mortgage'
            house.for_sale = False
            a = buyer.affordability_repayment()
            M = buyer.max_mortgage()
            D = buyer.deposit(M)
            house.price = M + D
            buyer.repayments[house.id] = a
            buyer.mortgages[house.id] = LTV * house.price
            buyer.paid_mortgages[house.id] = 0.0

            self.transaction_records.append({'house': house.id, 'time': self.time_step, 'price': house.price})

        # Incomes and payments
        for hh in self.households:
            hh.capital += hh.income / STEPS_PER_YEAR
            if hh.type == 'mortgage':
                hh.capital -= sum(hh.repayments.values())
            elif hh.type == 'rent' and hh.my_house:
                hh.capital -= hh.my_house.rent
        
        self.record_average_prices()

    def average_prices(self):
        prices = [h.price for h in self.houses if h.price > 0]
        rents = [h.rent for h in self.houses if h.rent > 0]
        return (np.mean(prices) if prices else 0.0, np.mean(rents) if rents else 0.0)

    def record_average_prices(self):
        all_prices = [h.price for h in self.houses if h.price > 0]
        all_rents = [h.rent for h in self.houses if h.rent > 0]

        avg_price = sum(all_prices) / len(all_prices) if all_prices else 0
        avg_rent = sum(all_rents) / len(all_rents) if all_rents else 0

        self.mean_prices_over_time.append(avg_price)
        self.mean_rents_over_time.append(avg_rent)


# === Pygame UI ===

def draw_text(surface, text, pos, font, color=(255,255,255)):
    img = font.render(text, True, color)
    surface.blit(img, pos)

def draw_grid(surface, model):
    for h in model.houses:
        x = h.x * CELL_SIZE
        y = h.y * CELL_SIZE
        rect = pygame.Rect(x,y,CELL_SIZE-1,CELL_SIZE-1)
        if h.occupier and h.occupier.type == 'mortgage':
            color = (0,150,0)
        elif h.occupier and h.occupier.type == 'rent':
            color = (150,0,0)
        elif h.for_sale:
            color = (150,255,150)
        elif h.for_rent:
            color = (255,150,150)
        else:
            color = (50,50,50)
        pygame.draw.rect(surface, color, rect)

# def draw_price_graph(surface, model, x_offset, width, height):
#     fig = plt.figure(figsize=(width/100, height/100), dpi=100)
#     ax = fig.add_subplot(111)

#     prices = [t['price'] for t in model.transaction_records]
#     rental_prices = [h.rent for h in model.houses if h.rent > 0]

#     if prices:
#         ax.plot(prices, label='Avg Sale Price')
#     if rental_prices:
#         ax.plot(rental_prices, label='Avg Rent Price')

#     ax.set_title("Housing Market Prices")
#     ax.set_xlabel("Transaction # or House Index")
#     ax.set_ylabel("Price (£)")
#     ax.legend(loc='upper left')
#     ax.grid(True)

#     if prices:
#         ax.annotate(f'{int(prices[-1]):,}', xy=(len(prices)-1, prices[-1]),
#                     xytext=(len(prices)-10, prices[-1]*1.1),
#                     arrowprops=dict(facecolor='blue', shrink=0.05),
#                     color='blue')
#     if rental_prices:
#         ax.annotate(f'{int(rental_prices[-1]):,}', xy=(len(rental_prices)-1, rental_prices[-1]),
#                     xytext=(len(rental_prices)-10, rental_prices[-1]*0.9),
#                     arrowprops=dict(facecolor='orange', shrink=0.05),
#                     color='orange')

#     canvas = FigureCanvasAgg(fig)
#     canvas.draw()
#     raw_data = canvas.tostring_rgb()
#     size = canvas.get_width_height()

#     surf = pygame.image.fromstring(raw_data, size, "RGB")
#     surface.blit(surf, (x_offset, 0))
#     plt.close(fig)

def draw_price_graph(surface, model, x_offset, width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)

    # Get time series data for each time step
    mean_prices = model.mean_prices_over_time  # e.g., [avg_price_t0, avg_price_t1, ..., avg_price_tN]
    mean_rents = model.mean_rents_over_time    # e.g., [avg_rent_t0, avg_rent_t1, ..., avg_rent_tN]

    time_steps = list(range(len(mean_prices)))

    # Plotting
    if mean_prices:
        ax.plot(time_steps, mean_prices, label='Avg House Price (£)', color='blue')
    if mean_rents:
        ax.plot(time_steps, mean_rents, label='Avg Rent (£)', color='orange')

    ax.set_title("Average Housing Prices and Rents Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price (£)")
    ax.legend(loc='upper left')
    ax.grid(True)

    # Optional annotations for last known values
    if mean_prices:
        ax.annotate(f'{int(mean_prices[-1]):,}', xy=(len(mean_prices)-1, mean_prices[-1]),
                    xytext=(len(mean_prices)-10, mean_prices[-1]*1.05),
                    arrowprops=dict(facecolor='blue', shrink=0.05),
                    color='blue')
    if mean_rents:
        ax.annotate(f'{int(mean_rents[-1]):,}', xy=(len(mean_rents)-1, mean_rents[-1]),
                    xytext=(len(mean_rents)-10, mean_rents[-1]*0.95),
                    arrowprops=dict(facecolor='orange', shrink=0.05),
                    color='orange')

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()

    surf = pygame.image.frombuffer(np.asarray(raw_data).tobytes(), size, "RGBA")
    surface.blit(surf, (x_offset, 0))
    plt.close(fig)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("UK Housing Market ABM")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)

    model = HousingMarketModel()

    paused = False
    step_delay_ms = 500
    last_step_time = pygame.time.get_ticks()

    running = True
    while running:
        screen.fill((15, 15, 15))

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_SPACE:
                paused = not paused

        current_time = pygame.time.get_ticks()
        if not paused and current_time - last_step_time > step_delay_ms:
            model.step()
            last_step_time = current_time

        draw_grid(screen, model)
        draw_price_graph(screen, model, GRID_SIZE*CELL_SIZE+2, 298, SCREEN_HEIGHT)

        draw_text(screen, f"Time step: {model.time_step} (3 months each). Press SPACE to Pause/Resume.", (10, SCREEN_HEIGHT - 25), font)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()