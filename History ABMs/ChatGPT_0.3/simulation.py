import pygame
from ui import render
from model import House, Household, Realtor
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GRID_SIZE
import random

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("UK Housing Model Simulator")
    clock = pygame.time.Clock()

    step = 0
    houses = [House(id=i) for i in range(GRID_SIZE ** 2)]
    households = [Household(id=i, is_owner=bool(random.randint(0, 1)), income=random.randint(20_000, 80_000)) for i in range(100)]
    realtor = Realtor(0)

    running = True
    while running:
        screen.fill((255, 255, 255))
        render(screen, houses, step)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Advance Time Step
        for house in houses:
            house.age += 1
            house.decay_price()
            house.decay_rent()
        
        for agent in households:
            status = agent.evaluate_status()
            # More logic: join market, offer bids, BTL purchases...

        step += 1
        clock.tick(FPS)

    pygame.quit()