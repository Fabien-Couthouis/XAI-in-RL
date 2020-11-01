import pygame
import math
import os
from abc import ABC
from .weapons import Arrow, Sword

WIDTH = 1280
HEIGHT = 720
ARCHER_SPEED = 25
KNIGHT_SPEED = 25
ARCHER_X, ARCHER_Y = 400, 610
KNIGHT_X, KNIGHT_Y = 800, 610
ANGLE_RATE = 10
IMG_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'img'))


class Player(pygame.sprite.Sprite, ABC):

    def __init__(self, agent_name, health, power):
        super().__init__()

        self.org_image = self.image.copy()
        self.angle = 0
        self.pos = pygame.Vector2(self.rect.center)

        self.attacking = False  # disable movement during attacking
        self.score = 0
        self.agent_name = agent_name
        self.health = health
        self.max_health = health
        self.is_archer = False
        self.is_knight = False
        self.power = power

    def offset(self, x_offset, y_offset):
        self.rect.x += x_offset
        self.rect.y += y_offset

    def is_done(self):
        return not self.is_alive()

    def remove_life(self):
        self.health -= 1

    def is_alive(self):
        return self.health > 0

    def draw_health(self):
        r = min(255, 255 - (255 * ((self.health -
                                    (self.max_health - self.health)) / self.max_health)))
        g = min(255, 255 * (self.health / (self.max_health / 2)))
        color = (r, g, 0)
        width = int(self.rect.width * self.health / self.max_health)
        self.health_bar = pygame.Rect(0, 0, width, 7)
        if self.health <= self.max_health:
            pygame.draw.rect(self.image, color, self.health_bar)


class Archer(Player):

    def __init__(self, agent_name, health=10, power=1):
        self.image = pygame.image.load(
            os.path.join(IMG_PATH, 'archer.png'))
        self.rect = self.image.get_rect(center=(ARCHER_X, ARCHER_Y))
        super().__init__(agent_name, health, power)
        self.direction = pygame.Vector2(0, -1)
        self.weapon = Arrow(self)
        self.is_archer = True

    def update(self, action):
        went_out_of_bounds = False

        if not self.attacking:
            move_angle = math.radians(self.angle + 90)
            # Up and Down movement
            if action == 1 and self.rect.y > 20:
                self.rect.x += math.cos(move_angle) * KNIGHT_SPEED
                self.rect.y -= math.sin(move_angle) * KNIGHT_SPEED
            elif action == 2 and self.rect.y < HEIGHT - 40:
                self.rect.x += math.cos(move_angle) * KNIGHT_SPEED
                self.rect.y += math.sin(move_angle) * KNIGHT_SPEED
            # Turn CCW & CW
            elif action == 3:
                self.angle += ANGLE_RATE
            elif action == 4:
                self.angle -= ANGLE_RATE
            elif action == 5 and self.is_alive():
                self.weapon.fired = True
                # self.attacking = True # gets reset to False in weapon attack
            elif action == 6:
                pass

            # Clamp to stay inside the screen
            if self.rect.y < 0 or self.rect.y > (HEIGHT - 40):
                went_out_of_bounds = True

            self.rect.x = max(min(self.rect.x, WIDTH - 132), 100)
            self.rect.y = max(min(self.rect.y, HEIGHT - 40), 0)

        self.direction = pygame.Vector2(0, -1).rotate(-self.angle)
        self.image = pygame.transform.rotate(self.org_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.draw_health()

        return went_out_of_bounds


class Knight(Player):
    def __init__(self, agent_name, health=10, power=1):
        self.image = pygame.image.load(
            os.path.join(IMG_PATH, 'knight.png'))
        self.rect = self.image.get_rect(center=(KNIGHT_X, KNIGHT_Y))
        super().__init__(agent_name, health, power)
        self.direction = pygame.Vector2(1, 0)
        self.attack_phase = -5
        self.weapon = Sword(self)
        self.action = -1
        self.is_knight = True

    def update(self, action):
        self.action = action
        went_out_of_bounds = False
        if not self.attacking:
            move_angle = math.radians(self.angle + 90)
            # Up and Down movement
            if action == 1 and self.rect.y > 20:
                self.rect.x += math.cos(move_angle) * KNIGHT_SPEED
                self.rect.y -= math.sin(move_angle) * KNIGHT_SPEED
            elif action == 2 and self.rect.y < HEIGHT - 40:
                self.rect.x += math.cos(move_angle) * KNIGHT_SPEED
                self.rect.y += math.sin(move_angle) * KNIGHT_SPEED
            # Turn CCW & CW
            elif action == 3:
                self.angle += ANGLE_RATE
            elif action == 4:
                self.angle -= ANGLE_RATE
            elif action == 5:
                self.attacking = True  # gets reset to False in weapon attack
            elif action == 6:
                pass

            # Clamp to stay inside the screen
            if self.rect.y < 0 or self.rect.y > (HEIGHT - 40):
                went_out_of_bounds = True

            self.rect.x = max(min(self.rect.x, WIDTH - 132), 100)
            self.rect.y = max(min(self.rect.y, HEIGHT - 40), 0)

        self.direction = pygame.Vector2(0, -1).rotate(-self.angle)
        self.image = pygame.transform.rotate(self.org_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.draw_health()

        return went_out_of_bounds
