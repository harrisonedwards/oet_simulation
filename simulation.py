import pygame, random
import argparse
from pygame.locals import *
import numpy as np
from oet_robot import Robot
from keras.utils import to_categorical


class Cell(pygame.sprite.Sprite):

    def __init__(self):
        super(Cell, self).__init__()
        self.image = pygame.Surface((4, 4))
        if random.randint(0, 1):
            self.type = 'red'
            pygame.draw.circle(self.image, (255, 0, 0), (2, 2), 2)
        else:
            self.type = 'green'
            pygame.draw.circle(self.image, (0, 255, 0), (2, 2), 2)
        self.rect = self.image.get_rect(
            center=(random.randint(0, SCREEN_WIDTH - 10), random.randint(0, SCREEN_HEIGHT - 10)))
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        # cases for managing collision with the edge
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT


class DropZone(pygame.sprite.Sprite):

    def __init__(self, size):
        super(DropZone, self).__init__()
        self.image = pygame.Surface((size, size))
        pygame.draw.rect(self.image, (0, 0, 127), (0, 0, size, size))
        self.rect = self.image.get_rect(
            center=(
                random.randint(size // 2, SCREEN_WIDTH - size // 2),
                random.randint(size // 2, SCREEN_HEIGHT - size // 2)))
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)


def handle_collision(rbt, cell):
    x = cell.rect.left - rbt.rect.left
    y = cell.rect.top - rbt.rect.top
    # overlap = np.sum(rbt.mask.overlap(cell.mask, [x, y]))
    dx = rbt.mask.overlap_area(cell.mask, (x + 1, y)) - rbt.mask.overlap_area(cell.mask, (x - 1, y))
    dy = rbt.mask.overlap_area(cell.mask, (x, y + 1)) - rbt.mask.overlap_area(cell.mask, (x, y - 1))
    # while overlap > 0:
    cell.rect.move_ip(-1 * np.sign(dx), -1 * np.sign(dy))


def update_all_collisions(cells, rbt):
    for cell in cells:
        if pygame.sprite.collide_mask(cell, rbt):
            handle_collision(rbt, cell)
        col_test = pygame.sprite.spritecollideany(cell, cells)
        if col_test != cell:
            handle_collision(col_test, cell)
        # lazy way of making sure the robot does not drop a cell if there are multiple in the basket
        if pygame.sprite.collide_mask(cell, rbt):
            handle_collision(rbt, cell)


def main(args):
    # initialize the pygame module and clock
    pygame.init()
    clock = pygame.time.Clock()

    pygame.display.set_caption("OET Simulation")

    # create a surface on screen that has the size of 300x300
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill((0))

    # set up our sprite groups
    cells = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()

    # create our robot right in the middle
    rbt = Robot([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], screen, SCREEN_WIDTH, SCREEN_HEIGHT)
    all_sprites.add(rbt)

    # generate a random number of cells
    for i in range(random.randint(args.min_cells, args.max_cells)):
        cell = Cell()
        all_sprites.add(cell)
        cells.add(cell)

    # create the designated drop zone
    dz = DropZone(50)
    all_sprites.add(dz)

    # draw all of the sprites initially
    for entity in all_sprites:
        screen.blit(entity.image, entity.rect)

    running = True
    while running:

        state_old = rbt.get_state(cells, dz)

        if np.random.randint(0, 2) > rbt.epsilon:
            final_move = to_categorical(np.random.randint(0, 3), num_classes=4)
            # print('RANDOM:', final_move)
        else:
            # predict action based on the old state
            prediction = rbt.model.predict(state_old.reshape((1, 13)))
            final_move = to_categorical(np.argmax(prediction[0]), num_classes=4)
            # print('PREDICT:', final_move)


        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    rbt.get_state(cells, dz)
            if event.type == pygame.QUIT:
                running = False

        # pressed_keys = pygame.key.get_pressed()

        # render background and drop zone, update robot, then all cells
        screen.fill(0)
        screen.blit(dz.image, dz.rect)
        # rbt.update(pressed_keys)
        rbt.update(final_move)
        for cell in cells:
            screen.blit(cell.image, cell.rect)

        # check for collision between robot and cells
        update_all_collisions(cells, rbt)

        state_new = rbt.get_state(cells, dz)

        reward = rbt.set_reward(state_new)

        # train short memory base on the new action and state
        rbt.train_short_memory(state_old, final_move, reward, state_new)
        # store the new data into a long term memory
        # rbt.remember(state_old, final_move, reward, state_new, game.crash)

        # update display
        pygame.display.flip()
        clock.tick(120)  # try to run at 120 fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optoelectronic Tweezer Simulation Environment')
    parser.add_argument('--min_cells', dest='min_cells', default=5)
    parser.add_argument('--max_cells', dest='max_cells', default=20)
    parser.add_argument('--screen_size', dest='screen_size', default=500)
    args = parser.parse_args()
    SCREEN_WIDTH = args.screen_size
    SCREEN_HEIGHT = args.screen_size
    main(args)
