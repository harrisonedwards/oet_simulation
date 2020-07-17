import pygame
from numpy import random


class Cell(pygame.sprite.Sprite):

    def __init__(self, screen_width, screen_height):
        super(Cell, self).__init__()
        self.image = pygame.Surface((4, 4))
        self.screen_width = screen_width
        self.screen_height = screen_height
        if random.randint(0, 2):
            self.type = 'red'
            pygame.draw.circle(self.image, (255, 0, 0), (2, 2), 2)
        else:
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
