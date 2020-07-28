import argparse
import datetime
import numpy as np
import pandas as pd
import pygame
import random, time
from pygame.locals import *
from characters import Cell, OETField, DropZone, Sweeper
from oet_robot import Robot
import gym
from gym import spaces


class OETEnvironment(gym.Env):

    def __init__(self, render=False):
        super(OETEnvironment, self).__init__()
        pygame.init()
        self.rendering = render

        # set up our sprite groups
        self.cells = pygame.sprite.Group()
        self.sweepers = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()

        # initialize variables
        self.screen = None
        self.rbt = None
        self.dz = None
        self.done = False

    def reset(self):
        # create a surface on screen
        if self.rendering:
            pygame.display.set_caption("OET Simulation")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        # add robot
        self.rbt = Robot((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), self.screen, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.all_sprites.add(self.rbt)

        # generate green cells
        for i in range(random.randint(3, 8)):
            cell = Cell(SCREEN_WIDTH, SCREEN_HEIGHT, 'green')
            self.all_sprites.add(cell)
            self.cells.add(cell)

        # generate red cells
        for i in range(random.randint(15, 300)):
            cell = Cell(SCREEN_WIDTH, SCREEN_HEIGHT, 'red')
            self.all_sprites.add(cell)
            self.cells.add(cell)

        # TODO generate the sweepers, and make sure their locations correspond to where the red cells are
        for cell in self.cells:
            if cell.type == 'red':
                sweeper = Sweeper(cell, SCREEN_WIDTH, SCREEN_HEIGHT)
                self.all_sprites.add(sweeper)
                self.sweepers.add(sweeper)

        # create the designated drop zone
        self.dz = DropZone(20, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.all_sprites.add(self.dz)

        # perform initial blit of all entities
        self.screen.fill(0)
        for entity in self.all_sprites:
            self.screen.blit(entity.image, entity.rect)

        # setup gym space
        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=4, shape=(50, 50), dtype=np.uint8)

        return self._get_obs()

    def _get_obs(self):
        # we want to return an array that has the identities of each entity in the grid
        obs = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
        # add all of the cell locations
        for cell in self.cells:
            x, y = cell.rect.center
            if -SCREEN_WIDTH < x < SCREEN_WIDTH and -SCREEN_HEIGHT < y < SCREEN_HEIGHT:
                if cell.type == 'green':
                    obs[x, y] = 2
                elif cell.type == 'red':
                    obs[x, y] = 3

        # add the dropzone location:
        x, y = self.dz.rect.center
        obs[x, y] = 4

        # add the robot's location
        x, y = self.rbt.rect.center
        obs[x, y] = 1

        # TODO add the sweeper locations

        return obs

    def step(self, action):
        # TODO: update robot action
        # self.rbt.update(action)

        for sweeper in self.sweepers:
            sweeper.update()

        # check for collisions
        if pygame.sprite.spritecollideany(self.rbt, self.cells):
            update_all_collisions(self.cells, self.rbt)
        for sweeper in self.sweepers:
            update_all_collisions(self.cells, sweeper)

        # update our observation (get a 500x500 array from pygame)
        obs = self._get_obs()

        # TODO: calculate reward
        # self.rbt.set_reward()
        reward = np.random.randint(0, 2)

        # TODO: determine whether or not to end the game

        return obs, reward, self.done, {}

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.done = True
            if event.type == pygame.QUIT:
                self.done = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
                print('MOUSE DOWN')
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
        self.screen.fill(0)
        for entity in self.all_sprites:
            self.screen.blit(entity.image, entity.rect)
        pygame.display.flip()


def handle_collision(rbt, cell):
    x = cell.rect.left - rbt.rect.left
    y = cell.rect.top - rbt.rect.top
    dx = rbt.mask.overlap_area(cell.mask, (x + 2, y)) - rbt.mask.overlap_area(cell.mask, (x - 2, y))
    dy = rbt.mask.overlap_area(cell.mask, (x, y + 2)) - rbt.mask.overlap_area(cell.mask, (x, y - 2))
    # while overlap > 0:
    cell.rect.move_ip(-1 * np.sign(dx), -1 * np.sign(dy))


def update_all_collisions(cells, rbt):
    for cell in cells:
        if pygame.sprite.collide_mask(cell, rbt):
            handle_collision(rbt, cell)



def write_results(results):
    now = datetime.datetime.now().strftime('%H_%M_%d_%m_%Y')
    results.to_csv(f'./results/{now}.csv')
    return


def main(args):
    render = True
    env = OETEnvironment(render=render)
    observation = env.reset()
    for t in range(1000):
        if render:
            env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(.01)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optoelectronic Tweezer Simulation Environment')
    parser.add_argument('--min_cells', dest='min_cells', default=5)
    parser.add_argument('--max_cells', dest='max_cells', default=20)
    parser.add_argument('--screen_size', dest='screen_size', default=500)
    args = parser.parse_args()
    SCREEN_WIDTH = args.screen_size
    SCREEN_HEIGHT = args.screen_size
    main(args)
