"""
file: forest.py
"""

import random
import pygame

from collections import namedtuple
from perlin_noise import PerlinNoise

class ForestMap:
    Rect = namedtuple("Rect", ["x", "y", "width", "height", "center"])
    def __init__(self, num_organisms: int = 10, screen_width: int = 800, screen_height: int = 600):
        self.num_organisms = num_organisms
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.screen = None
        self._pygame_init()

        self.organisms = list()
        self.picture = None
        self.road = list()

    def _pygame_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Random Forest Map Generator")

    def generate_map(self):
        """
        Generate a random forest.
        """
        noise = PerlinNoise(octaves=6, seed=random.randint(0, 100000))
        xpix, ypix = self.screen_width, self.screen_height
        pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]

        self.picture = pic

    def generate_organisms(self):
        """
        Generate organisms on the map.
        """
        def make_tree(x, y, size):
            tree_x, tree_y = x, y
            tree_rect = pygame.Rect(tree_x - size // 2, tree_y - size // 2, size, size)

            overlapping = False
            # for other_tree in self.organisms:
            #     other_rect = pygame.Rect(other_tree[0].x, other_tree[0].y, other_tree[0].width, other_tree[0].height)
            #     if tree_rect.colliderect(other_rect):
            #         overlapping = True
            #         break

            if not overlapping:
                tree_center_x, tree_center_y = tree_rect.center
                tree_rect = self.Rect(tree_center_x, tree_center_y, size, size, (tree_x, tree_y))
                self.organisms.append((tree_rect, (random.randint(200, 255), 0, 0)))

        margin = 10
        for i in range(0, len(self.picture) - margin, margin):
            for j in range(0, len(self.picture[i]) - margin, margin):
                chance = random.random()
                if self.picture[i][j]>=0.2 and chance < 0.3:
                    make_tree(j, i, 4)
                elif self.picture[i][j]>=0.09 and chance < 0.2:
                    make_tree(j, i, 3)
                elif self.picture[i][j]>=0.009 and chance < 0.1:
                    make_tree(j, i, 3)
                elif self.picture[i][j]>=-0.3 and chance < 0.05:
                    make_tree(j, i, 3)

        self.organisms = tuple(self.organisms)

    def _update_picture(self):
        """
        Update the picture of the forest.
        """
        screen = self.screen
        for i, row in enumerate(self.picture):
            for j, column in enumerate(row):
                if column>=0.2:
                    # There will be generated animals
                    pygame.draw.rect(screen, (80, 80, 80), pygame.Rect(j, i, 1, 1))             
                elif column>=0.09:
                    pygame.draw.rect(screen, (30, 90, 30), pygame.Rect(j, i, 1, 1))
                elif column >=0.009:
                    pygame.draw.rect(screen, (10, 100, 10), pygame.Rect(j, i, 1, 1))
                elif column >=0.002:
                    pygame.draw.rect(screen, (100, 150, 0), pygame.Rect(j, i, 1, 1))
                elif column >=-0.06:
                    pygame.draw.rect(screen, (30, 190, 0), pygame.Rect(j, i, 1, 1))
                elif column >=-0.02:
                    pygame.draw.rect(screen, (40, 200, 0), pygame.Rect(j, i, 1, 1))
                elif column >=-0.3:
                    pygame.draw.rect(screen, (10, 210, 0), pygame.Rect(j, i, 1, 1))
                elif column >=-0.8 and column <-0.3:
                    pygame.draw.rect(screen, (0, 0, 200), pygame.Rect(j, i, 1, 1))

    def _update_organisms(self):
        """
        Update the organisms positions in the forest.
        """
        for tree, color in self.organisms:
            pygame.draw.rect(self.screen, color, (tree.x, tree.y, tree.width, tree.height))

    def _update_path(self):
        """
        Draw a path on the screen.
        """
        assert self.screen is not None, "Must call pygame_init() before drawing a path."

        for i in range(len(self.road) - 1):
            pygame.draw.line(self.screen, (255, 255, 255), self.road[i], self.road[i + 1], 2)

    def run(self):
        """
        Run the simulation.
        """
        assert self.screen is not None, "Must call pygame_init() before running the simulation."
        pygame.display.update()

        # Wait for the user to close the window
        running = True
        while running:
            self.screen.fill((0, 0, 0))

            self._update_picture()
            self._update_organisms()
            self._update_path()

            pygame.time.delay(1000)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()


if __name__ == "__main__":
    # Test the forest map

    NUM_TREES = 300
    MIN_TREE_SIZE = 20
    MAX_TREE_SIZE = 60
    TREE_MARGIN = 10

    forest = ForestMap(NUM_TREES)
    forest.generate_map()
    forest.generate_organisms()

    forest.run()
