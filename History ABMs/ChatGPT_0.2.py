import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 20
TILE_SIZE = 32
SCREEN_WIDTH = GRID_SIZE * TILE_SIZE
GRAPH_HEIGHT = 100
SCREEN_HEIGHT = GRID_SIZE * TILE_SIZE + (3 * GRAPH_HEIGHT)
FPS = 5
INTEREST_RATE = 0.05  # 5% annual interest rate, simplified for simulation
LOAN_TO_VALUE_RATIO = 0.90  # 90% loan, 10% down payment

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
CYAN = (100, 200, 255)
GREEN = (0, 200, 100)
RED = (255, 50, 50)
PALERED = (255, 150, 150)
YELLOW = (255, 255, 100)
PURPLE = (200, 100, 255)

# Graph
GRAPH_WIDTH = SCREEN_WIDTH
GRAPH_POS = (0, SCREEN_HEIGHT - GRAPH_HEIGHT)
MAX_HISTORY = 100  # max data points to store for graph

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Housing ABM Simulation")
clock = pygame.time.Clock()
price_history = []
rent_history = []
income_history = []
capital_history = []
loan_history = []

step_count = 0

# Define agent types
class House:
    def __init__(self, x, y, owned=False, rented=False):
        self.x = x
        self.y = y
        self.owned = owned
        self.rented = rented
        self.owner = None
        self.tenant = None
        self.price = random.randint(100, 300)
        self.rent_price = random.randint(10, 30)
        self.occupancy_history = []

    def draw(self):
        color = CYAN if self.owned and not self.rented else PALERED if self.rented else WHITE
        pygame.draw.rect(screen, color, (self.x * TILE_SIZE, self.y * TILE_SIZE, TILE_SIZE, TILE_SIZE))


class Household:
    def __init__(self, x, y, is_owner):
        self.x = x
        self.y = y
        self.is_owner = is_owner
        self.income = random.randint(50, 150)
        self.capital = random.randint(50, 200)
        self.my_house = None
        self.rent = 0
        self.owned_houses = []
        self.loan = 0

    def move(self):
        self.x = max(0, min(GRID_SIZE - 1, self.x + random.choice([-1, 0, 1])))
        self.y = max(0, min(GRID_SIZE - 1, self.y + random.choice([-1, 0, 1])))

    def update(self, houses):
        self.capital += self.income // 10

        interest_payment = int(self.loan * INTEREST_RATE / 12)
        self.capital -= interest_payment

        if self.my_house:
            if self.my_house.owner == self:
                payment = self.my_house.price // 100
                self.capital -= payment
            else:
                rent = self.my_house.rent_price
                self.capital -= rent
                if self.my_house.owner:
                    self.my_house.owner.capital += rent

        if self.my_house and self.capital < 0:
            self.my_house.tenant = None
            self.my_house.rented = False
            self.my_house = None

        if not self.is_owner and self.my_house is None:
            for house in houses:
                if house.owned and not house.rented and house.owner != self and abs(house.x - self.x) <= 1 and abs(house.y - self.y) <= 1:
                    if self.capital >= house.rent_price:
                        self.my_house = house
                        house.tenant = self
                        house.rented = True
                        break

        if self.capital >= 100:
            for house in houses:
                if not house.owned and abs(house.x - self.x) <= 1 and abs(house.y - self.y) <= 1:
                    if self.capital >= house.price:
                        self.capital -= house.price
                        self.is_owner = True
                        self.my_house = house
                        house.owned = True
                        house.owner = self
                        self.owned_houses.append(house)
                        break
                    elif self.capital >= house.price * (1 - LOAN_TO_VALUE_RATIO):
                        down_payment = int(house.price * (1 - LOAN_TO_VALUE_RATIO))
                        self.loan += house.price - down_payment
                        self.capital -= down_payment
                        self.is_owner = True
                        self.my_house = house
                        house.owned = True
                        house.owner = self
                        self.owned_houses.append(house)
                        break

    def draw(self):
        if self.is_owner and self.my_house: color = BLUE
        elif not self.is_owner and self.my_house: color = RED
        else: color = BLACK

        pygame.draw.circle(screen, color, (self.x * TILE_SIZE + TILE_SIZE // 2, self.y * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 3)


class Realtor:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        pygame.draw.rect(screen, BLACK, (self.x * TILE_SIZE + 8, self.y * TILE_SIZE + 8, TILE_SIZE - 16, TILE_SIZE - 16))

    def adjust_prices(self, houses):
        for house in houses:
            house.occupancy_history.append(1 if house.rented else 0)
            if len(house.occupancy_history) > 10:
                house.occupancy_history.pop(0)

            avg_occupancy = sum(house.occupancy_history) / len(house.occupancy_history)

            neighborhood_prices = [h.price for h in houses if abs(h.x - house.x) <= 1 and abs(h.y - house.y) <= 1 and h != house]
            if neighborhood_prices:
                neighborhood_avg_price = sum(neighborhood_prices) / len(neighborhood_prices)
            else:
                neighborhood_avg_price = house.price

            if avg_occupancy < 0.3:
                house.rent_price = max(5, house.rent_price - 2)
                house.price = max(50, int((house.price + neighborhood_avg_price * 0.9) / 2))
            elif avg_occupancy > 0.7:
                house.rent_price = min(50, house.rent_price + 2)
                house.price = min(500, int((house.price + neighborhood_avg_price * 1.1) / 2))

    def maybe_build_house(self, houses):
        build_probability = random.randint(0, 100)
        if build_probability < 5:
            for _ in range(1):
                for _ in range(10):
                    new_x = random.randint(0, GRID_SIZE - 1)
                    new_y = random.randint(0, GRID_SIZE - 1)
                    if not any(h.x == new_x and h.y == new_y for h in houses):
                        new_house = House(new_x, new_y)
                        houses.append(new_house)
                        break

def maybe_add_immigrant(households, step_count):
    if step_count % 10 == 0:
        num_households = len(households)
        target_new = num_households // 10

        for _ in range(target_new):
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            is_owner = random.random() < 0.5
            new_household = Household(x, y, is_owner=is_owner)
            households.append(new_household)

def draw_graph():
    price_graph_top = SCREEN_HEIGHT - 3 * GRAPH_HEIGHT
    capital_graph_top = SCREEN_HEIGHT - 2 * GRAPH_HEIGHT
    loan_graph_top = SCREEN_HEIGHT - GRAPH_HEIGHT

    pygame.draw.rect(screen, (230, 230, 230), (0, price_graph_top, GRAPH_WIDTH, GRAPH_HEIGHT))
    pygame.draw.rect(screen, WHITE, (0, capital_graph_top, GRAPH_WIDTH, GRAPH_HEIGHT))
    pygame.draw.rect(screen, (230, 230, 230), (0, loan_graph_top, GRAPH_WIDTH, GRAPH_HEIGHT))

    if len(price_history) > 1:
        max_price = max(max(price_history), max(rent_history), 1)
        for i in range(1, len(price_history)):
            pygame.draw.line(screen, RED, (i - 1, price_graph_top + GRAPH_HEIGHT - int(rent_history[i - 1] / max_price * GRAPH_HEIGHT)),
                             (i, price_graph_top + GRAPH_HEIGHT - int(rent_history[i] / max_price * GRAPH_HEIGHT)))
            pygame.draw.line(screen, BLUE, (i - 1, price_graph_top + GRAPH_HEIGHT - int(price_history[i - 1] / max_price * GRAPH_HEIGHT)),
                             (i, price_graph_top + GRAPH_HEIGHT - int(price_history[i] / max_price * GRAPH_HEIGHT)))

        max_fin = max(max(income_history), max(capital_history), 1)
        x_offset = step_count - len(income_history)
        for i in range(1, len(income_history)):
            pygame.draw.line(screen, GREEN, (x_offset + i - 1, capital_graph_top + GRAPH_HEIGHT - int(income_history[i - 1] / max_fin * GRAPH_HEIGHT)),
                             (x_offset + i, capital_graph_top + GRAPH_HEIGHT - int(income_history[i] / max_fin * GRAPH_HEIGHT)))
            pygame.draw.line(screen, PURPLE, (x_offset + i - 1, capital_graph_top + GRAPH_HEIGHT - int(capital_history[i - 1] / max_fin * GRAPH_HEIGHT)),
                             (x_offset + i, capital_graph_top + GRAPH_HEIGHT - int(capital_history[i] / max_fin * GRAPH_HEIGHT)))
        
        max_loan = max(max(loan_history), 1)
        for i in range(1, len(loan_history)):
            pygame.draw.line(screen, YELLOW, (x_offset + i - 1, loan_graph_top + GRAPH_HEIGHT - int(loan_history[i - 1] / max_loan * GRAPH_HEIGHT)),
                             (x_offset + i, loan_graph_top + GRAPH_HEIGHT - int(loan_history[i] / max_loan * GRAPH_HEIGHT)))


houses = [House(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1), owned=False, rented=False) for _ in range(50)]
households = [Household(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1), is_owner=False) for _ in range(20)]
realtors = [Realtor(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) for _ in range(5)]

landlords = [Household(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1), is_owner=True) for _ in range(5)]
households.extend(landlords)

for landlord in landlords:
    for _ in range(random.randint(2, 4)):
        vacant = next((h for h in houses if not h.owned), None)
        if vacant:
            vacant.owned = True
            vacant.owner = landlord
            landlord.owned_houses.append(vacant)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    for house in houses:
        house.draw()

    for realtor in realtors:
        realtor.adjust_prices(houses)
        realtor.maybe_build_house(houses)
        realtor.draw()

    for household in households:
        household.move()
        household.update(houses)
        household.draw()
    
    # Track average prices and rents
    avg_price = sum(h.price for h in houses) / len(houses)
    avg_rent = sum(h.rent_price for h in houses) / len(houses)
    avg_capital = sum(hh.capital for hh in households) / len(households)
    avg_income = sum(hh.income for hh in households) / len(households)
    avg_loan = sum(hh.loan for hh in households) / len(households)

    price_history.append(avg_price)
    rent_history.append(avg_rent)
    capital_history.append(avg_capital)
    income_history.append(avg_income)
    loan_history.append(avg_loan)

    if len(price_history) > MAX_HISTORY:
        price_history.pop(0)
        rent_history.pop(0)
        capital_history.pop(0)
        income_history.pop(0)
        loan_history.pop(0)

    draw_graph()
    step_count += 1
    pygame.display.flip()
    # pygame.image.save(screen, f"snapshots/frame_{step_count:04d}.png")
    clock.tick(FPS)



pygame.quit()