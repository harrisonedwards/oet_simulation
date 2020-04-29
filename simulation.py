import pygame, random
import argparse
from pygame.locals import *
import numpy as np


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

    def __init__(self, start_loc, screen):
        super(Robot, self).__init__()
        self.image = pygame.image.load("robot.png")
        self.rect = self.image.get_rect()
        self.image.set_colorkey((0, 0, 0))
        self.mask = pygame.mask.from_surface(self.image)
        self.pos = start_loc
        self.screen = screen
        self.angle = 0
        self.speed = 1

    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.pos[0] = self.pos[0] - np.sin(self.angle * np.pi / 180) * self.speed
            self.pos[1] = self.pos[1] - np.cos(self.angle * np.pi / 180) * self.speed
        if pressed_keys[K_DOWN]:
            self.pos[0] = self.pos[0] + np.sin(self.angle * np.pi / 180) * self.speed
            self.pos[1] = self.pos[1] + np.cos(self.angle * np.pi / 180) * self.speed
        if pressed_keys[K_LEFT]:
            self.angle += self.speed
        if pressed_keys[K_RIGHT]:
            self.angle -= self.speed

        rotated_image, origin = rotate(self.image, self.pos, self.angle)
        self.rect.topleft = origin
        self.mask = pygame.mask.from_surface(rotated_image)
        # cases for managing collision with the edge
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

        self.screen.blit(rotated_image, self.rect.topleft)


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


def handle_collision(rbt, cell):
    x = cell.rect.left - rbt.rect.left
    y = cell.rect.top - rbt.rect.top
    overlap = np.sum(rbt.mask.overlap(cell.mask, [x, y]))
    dx = rbt.mask.overlap_area(cell.mask, (x + 1, y)) - rbt.mask.overlap_area(cell.mask, (x - 1, y))
    dy = rbt.mask.overlap_area(cell.mask, (x, y + 1)) - rbt.mask.overlap_area(cell.mask, (x, y - 1))
    # while overlap > 0:
    cell.rect.move_ip(-1 * np.sign(dx), -1 * np.sign(dy))


def check_for_any_collisions(all_sprites):
    for entity in all_sprites:
        col_test = pygame.sprite.spritecollideany(entity, all_sprites)
        if col_test != entity:
            return True


def main(args):
    # initialize the pygame module and clock
    pygame.init()
    clock = pygame.time.Clock()

    # pygame.display.set_icon(logo)
    pygame.display.set_caption("OET Simulation")

    # create a surface on screen that has the size of 300x300
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill((0))

    # set up our sprite groups
    cells = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()

    # create our robot right in the middle
    rbt = Robot([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], screen)
    all_sprites.add(rbt)

    # generate a random number of cells
    for i in range(random.randint(args.min_cells, args.max_cells)):
        cell = Cell()
        all_sprites.add(cell)
        cells.add(cell)

    # draw all of the sprites initially
    for entity in all_sprites:
        screen.blit(entity.image, entity.rect)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
            if event.type == pygame.QUIT:
                running = False
        pressed_keys = pygame.key.get_pressed()

        # render background, update robot, then all cells
        screen.fill(0)
        rbt.update(pressed_keys)
        for cell in cells:
            screen.blit(cell.image, cell.rect)

        # check for collision between robot and cells
        # while check_for_any_collisions(all_sprites):
        for cell in cells:
            if pygame.sprite.collide_mask(cell, rbt):
                handle_collision(rbt, cell)
            col_test = pygame.sprite.spritecollideany(cell, cells)
            if col_test != cell:
                handle_collision(col_test, cell)
            # lazy way of making sure the robot does not drop a cell if there are multiple in the basket
            if pygame.sprite.collide_mask(cell, rbt):
                handle_collision(rbt, cell)
                # col_test = pygame.sprite.spritecollideany(cell, cells)
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
