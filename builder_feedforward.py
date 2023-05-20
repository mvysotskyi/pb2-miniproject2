"""
file: builder_feedforward.py
"""

import math
import neat

from forest import ForestMap
from builder import NeatForestRoadBuilder

class NeatForestRoadBuilderFeedForward(NeatForestRoadBuilder):
    """
    A class that uses NEAT to build a road through a forest.
    """
    def _build_path(self, net: neat.nn.FeedForwardNetwork):
        """
        'net' instance builds a road through the forest.
        """
        path = [(0, 0), (3, 3)]

        for _ in range(1, self.road_smoothness):
            if self.distance(path[-1], (self.screen_width, self.screen_height)) < 30:
                break

            delta_dest = (
                (self.screen_width - path[-1][0]) / 1,
                (self.screen_height - path[-1][1]) / 1
            )

            current_pos = (
                path[-1][0] / 1,
                path[-1][1] / 1
            )

            nearest = self.nearest_organism_cached(path[-1], self.forest_organisms, 5)
            nearest = filter(lambda x: self.distance(path[-1], x[0].center) < 30, nearest)

            delta_nearest = []
            for point in nearest:
                delta_nearest.append((point[0].center[0] - path[-1][0]) / 1)
                delta_nearest.append((point[0].center[1] - path[-1][1]) / 1)

            odd_point = [current_pos[0] - 30, current_pos[1] - 30]
            inputs = list(delta_dest) + list(current_pos) + list(delta_nearest) + odd_point * int((10 - len(delta_nearest)) / 2)

            step = 8
            output = net.activate(inputs)

            output = list(map(lambda x: 2 * (x - 0.5), output))

            path.append((
                path[-1][0] + output[0] * step,
                path[-1][1] + output[1] * step
            ))

        return path

    def eval_genome(self, genome, config):
        """
        Evaluate the fitness of a genome.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        path = self._build_path(net)

        fitness = 0

        for point in path:
            if not self.is_valid(point):
                continue

            nearest = self.nearest_organism_cached(point, self.forest_organisms, 5)
            nearest = tuple(filter(lambda x: self.distance(point, x[0].center) < 30, nearest))

            delta_fitness = 1
            for organism in nearest:
                delta_fitness *= self.distance(point, organism[0].center)

            delta_fitness = delta_fitness ** (1.0 / max(len(nearest), 1)) * 0.033 + 1
            fitness += 30 * delta_fitness

        path_length = sum(self.distance(path[i], path[i + 1]) for i in range(len(path) - 1))
        length_prop = 1 + math.fabs(path_length - self.distance((0, 0), (self.screen_height, self.screen_width)))

        delta_dest = self.distance(path[-1], (self.screen_width, self.screen_height))
        return fitness * (1 / (1 + delta_dest * 0.1)) ** 0.8 * (1 / length_prop ** 0.8)

if __name__ == "__main__":
    forest = ForestMap(900, 600, 400)

    forest.generate_map()
    forest.generate_organisms()

    builder = NeatForestRoadBuilderFeedForward(forest, 70)
    builder.build()

    print(forest.road)
    forest.run()
