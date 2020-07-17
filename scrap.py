import argparse
import datetime
import numpy as np
import pandas as pd
import pygame
import random
from pygame.locals import *
from characters import Cell, OETField, DropZone
import gym
from gym import spaces

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
            print(cell.mask.overlap(rbt.mask, (0, 0)))
            handle_collision(rbt, cell)
        col_test = pygame.sprite.spritecollideany(cell, cells)
        if col_test != cell:
            handle_collision(col_test, cell)
        # lazy way of making sure the robot does not drop a cell if there are multiple in the basket
        if pygame.sprite.collide_mask(cell, rbt):
            handle_collision(rbt, cell)


def write_results(results):
    now = datetime.datetime.now().strftime('%H_%M_%d_%m_%Y')
    results.to_csv(f'./results/{now}.csv')
    return


def main(args):
    # initialize the pygame module and clock
    pygame.init()
    clock = pygame.time.Clock()

    pygame.display.set_caption("OET Simulation")

    # create a surface on screen that has the size of 300x300
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(0)

    # set up our sprite groups
    cells = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()

    # create our robot right in the middle
    # rbt = Robot([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], screen, SCREEN_WIDTH, SCREEN_HEIGHT)
    # all_sprites.add(rbt)

    # generate a random number of cells
    for i in range(random.randint(args.min_cells, args.max_cells)):
        cell = Cell(SCREEN_WIDTH, SCREEN_HEIGHT)
        all_sprites.add(cell)
        cells.add(cell)

    # create the designated drop zone
    dz = DropZone(50, SCREEN_WIDTH, SCREEN_HEIGHT)
    all_sprites.add(dz)

    # create oetfield
    oet_field = OETField()
    all_sprites.add(oet_field)

    # draw all of the sprites initially
    for entity in all_sprites:
        screen.blit(entity.image, entity.rect)

    running = True

    counter = 0

    mouse_down = False

    # distances, images, moves, rewards, ys = np.zeros(2), np.zeros((500, 500)), np.zeros(2), np.zeros(1), np.zeros(2)
    distance = np.zeros(2)
    # gray = np.zeros((500, 500))
    intepretation = {0: 'F', 1: 'B', 2: 'L', 3: 'R'}
    # where the rubber meets the road... our state vector: [angle, [wallx, wally, walld], [cellx, celly, cellg,
    # cellr], [dzx, dzy], [cells_left], [holding_green, holding_red], g_d_s, r_d_s]
    cols = ['angle', 'wallx', 'wally', 'walld', 'cellx', 'celly', 'cellg', 'cellr', 'dzx', 'dzy', 'cells_left',
            'holding_green', 'holding_red', 'g_d_s', 'r_d_s', 'decision', 'reward']
    results = pd.DataFrame(columns=cols)
    while running:
        # old_state = rbt.get_state(cells, dz, rbt)
        # model_output = rbt.model.predict(old_state.reshape(1, 15)).flatten()
        # action = np.zeros(4)
        # if np.random.randint(4) < rbt.epsilon:
        #     action[np.argmax(model_output)] = 1
        #     print('M', intepretation[np.argmax(model_output)])
        # else:
        #     choice = np.random.randint(4)
        #     action[choice] = 1
        #     print('R', intepretation[choice])

        # for i in range(6):
        #     screen.fill(0)
        #     screen.blit(dz.image, dz.rect)
        # rbt.update(action)
        # for cell in cells:
        #     screen.blit(cell.image, cell.rect)
        # update_all_collisions(cells, rbt)

        # new_state = rbt.get_state(cells, dz, rbt)
        # reward = rbt.set_reward(new_state)
        # rbt.train_short_memory(reward, old_state, new_state, action)

        # want to store the old state, network decision, and reward
        # r = np.concatenate((old_state.flatten(), np.array((np.amax(model_output), reward))), axis=0)
        # results.loc[counter] = r

        # store the new data into a long term memory
        # states = np.vstack((states, state))
        # distances = np.vstack((distances, [positive_distance_sum, negative_distance_sum]))
        # images = np.vstack((images.reshape(-1, 500,500), gray.reshape(1,500,500)))
        # moves = np.vstack((moves, action))
        # rewards = np.vstack((rewards, reward))
        # ys = np.vstack((ys, model_output))

        # render background and drop zone, update robot, then all cells
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    write_results(results)
                    running = False
            if event.type == pygame.QUIT:
                write_results(results)
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False

        if mouse_down:
            oet_field.update_position(pygame.mouse.get_pos())
        if pygame.sprite.spritecollideany(oet_field, cells):
            update_all_collisions(cells, oet_field)

        # image = pygame.surfarray.array3d(screen)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        screen.fill(0)
        for entity in all_sprites:
            screen.blit(entity.image, entity.rect)
        pygame.display.flip()
        # clock.tick(60)  # try to run at 120 fps
        counter += 1
        # rbt.epsilon = rbt.epsilon * np.exp(-counter / 10e10000)
        # print(rbt.epsilon)

        # if counter > 5:
        #     # train on 10 actions
        #     rbt.train_short_memory(reward, old_state, new_state, action)
        #     counter = 0
        #     distances, images, moves, rewards, ys = np.zeros(2), np.zeros((500, 500)), np.zeros(2), np.zeros(
        #         1), np.zeros(
        #         2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optoelectronic Tweezer Simulation Environment')
    parser.add_argument('--min_cells', dest='min_cells', default=5)
    parser.add_argument('--max_cells', dest='max_cells', default=20)
    parser.add_argument('--screen_size', dest='screen_size', default=500)
    args = parser.parse_args()
    SCREEN_WIDTH = args.screen_size
    SCREEN_HEIGHT = args.screen_size
    main(args)