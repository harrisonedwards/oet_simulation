import pygame
from numpy import random
import numpy as np


class Cell(pygame.sprite.Sprite):

    def __init__(self, screen_width, screen_height, cell_type):
        super(Cell, self).__init__()
        self.image = pygame.Surface((4, 4))
        self.screen_width = screen_width
        self.screen_height = screen_height
        if cell_type == 'red':
            self.type = 'red'
            pygame.draw.circle(self.image, (255, 0, 0), (2, 2), 2)
        elif cell_type == 'green':
            self.type = 'green'
            pygame.draw.circle(self.image, (0, 255, 0), (2, 2), 2)
        self.rect = self.image.get_rect(
            center=(random.randint(0, self.screen_width - 10), random.randint(0, self.screen_height - 10)))
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        # cases for managing collision with the edge
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.screen_width:
            self.rect.right = self.screen_width
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= self.screen_height:
            self.rect.bottom = self.screen_height


class Sweeper(pygame.sprite.Sprite):

    def __init__(self, cell, screen_width, screen_height):
        super(Sweeper, self).__init__()
        self.closest, centerx, centery = self.get_cell_offset(cell, screen_width, screen_height)
        if int(self.closest) in [0, 1]:
            # we are closest to the left side
            self.image = pygame.Surface((2, 10))
            pygame.draw.rect(self.image, (255, 255, 0), (0, 0, 2, 10))
        elif int(self.closest) in [2,3]:
            self.image = pygame.Surface((10, 2))
            pygame.draw.rect(self.image, (255, 255, 0), (0, 0, 10, 2))
        if int(self.closest) == 0:
            self.rect = self.image.get_rect(centerx=centerx + 8, centery=centery)
        elif int(self.closest) == 1:
            self.rect = self.image.get_rect(centerx=centerx - 8, centery=centery)
        elif int(self.closest) == 2:
            self.rect = self.image.get_rect(centerx=centerx, centery=centery + 8)
        elif int(self.closest) == 3:
            self.rect = self.image.get_rect(centerx=centerx, centery=centery - 8)
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)

    def get_cell_offset(self, cell, screen_width, screen_height):
        x, y = cell.rect.centerx, cell.rect.centery
        closest = np.argmin([x, screen_width-x, y, screen_height-y])
        return closest, x, y

    def update(self):
        if int(self.closest) == 0:
            self.rect.centerx -= 1
        if int(self.closest) == 1:
            self.rect.centerx += 1
        if int(self.closest) == 2:
            self.rect.centery -= 1
        if int(self.closest) == 3:
            self.rect.centery += 1
        # self.rect.bottom += 1
        self.mask = pygame.mask.from_surface(self.image)


class DropZone(pygame.sprite.Sprite):

    def __init__(self, size, screen_width, screen_height):
        super(DropZone, self).__init__()
        self.image = pygame.Surface((size, size))
        pygame.draw.rect(self.image, (0, 0, 127), (0, 0, size, size))
        self.rect = self.image.get_rect(
            center=(
                random.randint(size // 2, screen_width - size // 2),
                random.randint(size // 2, screen_height - size // 2)))
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)


class OETField(pygame.sprite.Sprite):

    def __init__(self):
        super(OETField, self).__init__()
        size = 10
        self.image = pygame.Surface((size, size))
        pygame.draw.circle(self.image, (255, 255, 0), (5, 5), 3)
        self.rect = self.image.get_rect()
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)

    def update_position(self, position):
        # pygame.draw.circle(self.image, (0, 255, 0), position, 2)
        self.rect.center = position
        self.mask = pygame.mask.from_surface(self.image)
