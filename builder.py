"""
file: builder.py
"""

import math
from functools import cache

import neat

from forest import ForestMap


class NeatForestRoadBuilder:
    """
    A class that uses NEAT to build a road through a forest.
    """
    def __init__(self, forest_map: ForestMap, num_generations: int = 100):
        self.forest_map = forest_map

        self.forest_organisms: list = forest_map.organisms
        self.screen_width = forest_map.screen_width
        self.screen_height = forest_map.screen_height

        self.road_smoothness = 250
        self.winner = None

        self.num_generations = num_generations
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  "config-feedforward.txt")

    def _build_path(self, net):
        ...

    @staticmethod
    def distance(point1, point2):
        """
        Calculate the distance between two points.
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) **2)

    @staticmethod
    @cache
    def nearest_organism_cached(point, organisms, n_first=5):
        """
        Find the nearest organism to a point.
        """
        return sorted(organisms, key=lambda x: __class__.distance(x[0].center, point))[:n_first]

    def is_valid(self, point):
        """
        Check if a point is valid.
        """
        if point[0] < 0 or point[1] < 0 \
            or point[0] >= self.screen_width or point[1] >= self.screen_height:
            return False

        return True

    def eval_genome(self, genome, config):
        """
        Evaluate the fitness of a genome.
        """
        ...

    def eval_genomes(self, genomes, config):
        """
        Evaluate the fitness of a list of genomes.
        """
        for _, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)

    def _find_winner(self):
        pop = neat.Population(self.config)

        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        winner = pop.run(self.eval_genomes, self.num_generations)
        return winner

    def build(self):
        """
        Build a road through the forest.
        """
        self.winner = self._find_winner() if self.winner is None else self.winner
        net = neat.nn.FeedForwardNetwork.create(self.winner, self.config)

        self.forest_map.road = self._build_path(net)
