import Prey
import Predator
import Boids

from random import shuffle
import numpy as np
from math import ceil

class Genetic:

    def __init__(self, simulation, mutation_rate) -> None:
        self.simulation = simulation
        self.mutation_rate = mutation_rate

    # mutate the stats of the boids proportional to the mutation_rate
    def mutation(self, child):
        mutate = np.random.choice([True,False], p=[self.mutation_rate, 1-self.mutation_rate])
        if mutate:
            pass

    # crossover the stats of the two parents
    def crossover(self, parent1, parent2, boidclass : Boids.Boids):
        position = (boidclass.positions[parent1] + boidclass.positions[parent2]) / 2
        velocity = (boidclass.velocities[parent1] + boidclass.velocities[parent2]) / 2
        traits = {}

        for trait, traitlist in boidclass.traits.items():
            traits[trait] = (traitlist[parent1] + traitlist[parent2]) / 2

        return (position, velocity, traits)
            

    # create children from the two parents
    def make_children(self, parent1, parent2, boidclass : Boids.Boids):
        num_children = ceil(np.random.normal(2))
        children = []

        for i in range(num_children):
            child = self.crossover(parent1, parent2, boidclass)
            self.mutation(child)
            children.append(child)
        
        return children

    # randomly pair parents from the population
    def pair_random(self, population):
        shuffle(population)
        group1 = population[len(population / 2):]
        group2 = population[:len(population / 2)]

        return zip(group1, group2)

    # create the next generation from the population
    def next_generation(self, population, boidclass):
        velocities = []
        positions  = []
        traits = {}

        for parent1, parent2 in self.pair_random(population):
            children = self.make_children(parent1, parent2, boidclass)