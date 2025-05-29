import pygame
from config import *

def draw_house(screen, house, x, y):
    color = (0, 200, 0) if house.occupied else (120, 120, 120)
    if house.for_sale:
        color = (200, 0, 0)
    elif house.for_rent:
        color = (0, 0, 200)
    pygame.draw.rect(screen, color, pygame.Rect(x, y, 20, 20))

def render(screen, houses, step):
    screen.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 24)
    label = font.render(f'Timestep: {step} (Q{step % STEPS_PER_YEAR + 1})', True, (0, 0, 0))
    screen.blit(label, (10, 10))

    # Draw houses on grid
    for idx, house in enumerate(houses):
        x = (idx % GRID_SIZE) * 22 + 10
        y = (idx // GRID_SIZE) * 22 + 40
        draw_house(screen, house, x, y)
    
    pygame.display.flip()