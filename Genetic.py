import Prey
import Predator

from random import shuffle

class Genetic:

    def __init__(self, mutation_rate) -> None:
        self.mutation_rate = mutation_rate

    # mutate the stats of the boids proportional to the mutation_rate
    def mutation(self, boid):
        pass

    # crossover the stats of the two parents
    def crossover(self, parent1, parent2):
        pass

    # create children from the two parents
    def make_children(self, parent1, parent2):
        pass

    # randomly pair parents from the population
    def pair_random(self, population):
        shuffle(population)
        group1 = population[len(population / 2):]
        group2 = population[:len(population / 2)]

        return zip(group1, group2)

    # create the next generation from the population
    def next_generation(self, population):
        velocities = []
        positions  = []
        traits = {}

        for parent1, parent2 in self.pair_random(population):
            children = self.make_children(parent1, parent2)