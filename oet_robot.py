import pygame
from pygame.locals import *
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense


def get_percent_cells_not_in_drop_zone(cells, drop_zone):
    num_green = len([x for x in cells if x.type == 'green'])
    num_cells = num_green
    for cell in cells:
        if pygame.sprite.collide_mask(cell, drop_zone) and cell.type == 'green':
            num_cells -= 1
    return num_cells / num_green


def rotate(image, pos, angle):
    """
    image is the Surface which has to be rotated and blit
    pos is the position of the pivot on the target Surface surf (relative to the top left of surf)
    angle is the angle of rotation in degrees
    """
    # calculate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    origin_pos = (w // 2, h // 2)
    box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # calculate the translation of the pivot
    pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    return rotated_image, origin


class Robot(pygame.sprite.Sprite):

    def __init__(self, start_loc, screen, screen_width, screen_height):
        super(Robot, self).__init__()
        self.image = pygame.image.load("robot.png")
        self.rect = self.image.get_rect()
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)
        self.pos = start_loc
        self.screen = screen
        self.angle = 0
        self.speed = 1
        # specifying this arbitrarily as the threshold for cells that are being carried by the robot
        # (verified empirically)
        self.carrying_distance_threshold = np.sqrt(0.02 ** 2 * 2)
        self.carrying_cell_type = None
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.epsilon = .5
        self.model = self.network()
        self.reward = 0
        self.gamma = 0.9

    def network(self):
        model = Sequential()
        model.add(Dense(output_dim=64, activation='relu', input_dim=13))
        model.add(Dense(output_dim=128, activation='relu'))
        model.add(Dense(output_dim=256, activation='relu'))
        model.add(Dense(output_dim=4, activation='softmax'))
        opt = SGD()
        model.compile(loss='mse', optimizer=opt)
        return model

    def update(self, final_move):
        # if pressed_keys[K_UP]:
        #     self.pos[0] = self.pos[0] - np.sin(self.angle * np.pi / 180) * self.speed
        #     self.pos[1] = self.pos[1] - np.cos(self.angle * np.pi / 180) * self.speed
        # if pressed_keys[K_DOWN]:
        #     self.pos[0] = self.pos[0] + np.sin(self.angle * np.pi / 180) * self.speed
        #     self.pos[1] = self.pos[1] + np.cos(self.angle * np.pi / 180) * self.speed
        # if pressed_keys[K_LEFT]:
        #     self.angle += self.speed
        # if pressed_keys[K_RIGHT]:
        #     self.angle -= self.speed

        # replacing human controls with network predictions

        if final_move[0]:
            self.pos[0] = self.pos[0] - np.sin(self.angle * np.pi / 180) * self.speed
            self.pos[1] = self.pos[1] - np.cos(self.angle * np.pi / 180) * self.speed
        if final_move[1]:
            self.pos[0] = self.pos[0] + np.sin(self.angle * np.pi / 180) * self.speed
            self.pos[1] = self.pos[1] + np.cos(self.angle * np.pi / 180) * self.speed
        if final_move[2]:
            self.angle += self.speed
        if final_move[3]:
            self.angle -= self.speed

        self.angle = self.angle % 360

        rotated_image, origin = rotate(self.image, self.pos, self.angle)
        self.rect.topleft = origin
        self.mask = pygame.mask.from_surface(rotated_image)
        # cases for managing collision with the edge
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.screen_width:
            self.rect.right = self.screen_width
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= self.screen_height:
            self.rect.bottom = self.screen_height
        self.screen.blit(rotated_image, self.rect.topleft)

    def get_closest_wall(self):
        # we want to get a vector to the closest piece of wall
        # its always just going to be a single digit (up down left right)
        # probably best to just make it digital and have a value associated with it (distance)
        rbt_center = self.rect.center
        screen_center = (self.screen_width // 2, self.screen_height // 2)
        x_dir = np.sign(screen_center[0] - rbt_center[0])
        y_dir = np.sign(screen_center[1] - rbt_center[1])
        if np.abs(screen_center[0] - rbt_center[0]) > np.abs(screen_center[1] - rbt_center[1]):
            #     we are further from the center in the x direction than the y direction
            distance = np.abs(screen_center[0] - rbt_center[0]) / screen_center[0]
            y_dir = 0
        else:
            distance = np.abs(screen_center[1] - rbt_center[1]) / screen_center[1]
            x_dir = 0
        return [x_dir, y_dir, distance]

    def get_cell_awareness(self, cells):
        # updates whether or not the robot is carrying a cell, and also gets the closest cell
        prev_mag = np.inf
        closest_cell = None
        self.carrying_cell_type = None
        for cell in cells:
            dx = (cell.rect.center[0] - self.rect.center[0]) / self.screen_width
            dy = (cell.rect.center[1] - self.rect.center[1]) / self.screen_height
            mag = np.sqrt(dx ** 2 + dy ** 2)
            if self.carrying_distance_threshold > mag:
                self.carrying_cell_type = cell.type
            if self.carrying_distance_threshold < mag < prev_mag:
                closest_cell = cell
                prev_mag = mag
        vx = (closest_cell.rect.center[0] - self.rect.center[0]) / self.screen_width
        vy = (closest_cell.rect.center[1] - self.rect.center[1]) / self.screen_height
        return [vx, vy, int(closest_cell.type == 'green'), int(closest_cell.type == 'red')]

    def get_drop_zone_vector(self, drop_zone):
        dx = drop_zone.rect.center[0] - self.rect.center[0]
        dy = drop_zone.rect.center[1] - self.rect.center[1]
        return [dx / self.screen_width, dy / self.screen_height]

    def get_is_holding_cell(self, cells):
        return [int(self.carrying_cell_type == 'green'), int(self.carrying_cell_type == 'red')]

    def get_state(self, cells, drop_zone):
        # print('getting state')
        # print(f'angle:{self.angle / 360}\tvector to closest wall:{self.get_closest_wall()}\tvector to closest cell:'
        #       f'{self.get_cell_awareness(cells)}\t vector to drop zone:{self.get_drop_zone_vector(drop_zone)}\t'
        #       f'percentage of cells not yet collected:{get_percent_cells_not_in_drop_zone(cells, drop_zone)}\t'
        #       f'currently holding cell:{self.get_is_holding_cell(cells)}')

        state = [self.angle / 360] + self.get_closest_wall() + self.get_cell_awareness(cells) + \
                self.get_drop_zone_vector(drop_zone) + [get_percent_cells_not_in_drop_zone(cells, drop_zone)] + \
                self.get_is_holding_cell(cells)
        # print('STATE:', state)
        return np.array(state)

    def set_reward(self, state):
        # where the rubber meets the road... our state vector: [angle, [wallx, wally, walld], [cellx, celly, cellg,
        # cellr], [dzx, dzy], [cells_left], [holding_green, holding_red]]
        self.reward = 0

        # penalize for getting close to a wall
        if state[3] > .8:
            self.reward -= state[3] * .2

        # penalize for getting close to a red cell
        if state[7]:
            self.reward -= np.sqrt(state[4]**2 + state[5]**2)*10

        # penalize for picking up a red cell
        if state[-1]:
            self.reward -= 1

        # reward for getting close to a green cell:
        if state[6]:
            self.reward += np.sqrt(state[4]**2 + state[5]**2)*10

        # reward for picking up a green cell
        if state[-2]:
            self.reward += 10

        # reward for putting a green cell in the drop zone
        if state[10] < 1:
            self.reward += (-state[10] + 1) * 5

        # reward for getting close to the drop zone with a green cell
        if state[-2]:
            self. reward += np.sqrt(state[8]**2 + state[9]**2)

        print(self.reward)
        return self.reward

    def train_short_memory(self, state_old, final_move, reward, state_new):
        target = reward

        target = reward + self.gamma * np.amax(self.model.predict(state_new.reshape((1, 13)))[0])

        target_f = self.model.predict(state_old.reshape((1, 13)))

        target_f[0][np.argmax(final_move)] = target

        self.model.fit(state_old.reshape((1, 13)), target_f, epochs=1, verbose=0)
