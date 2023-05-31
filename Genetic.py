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
        
        #self.next_positions = []
        #self.next_velocity = []
        self.next_alignment_distance = []
        self.next_cohesion_distance = []
        self.next_separation_distance = []
        self.next_vision_distance = []
        self.next_alignment_strength = []
        self.next_cohesion_strength = []
        self.next_separation_strength = []
        self.next_noise_strength = []
        self.next_max_velocity = []
        self.next_traits = {}

    # mutate the stats of the boids proportional to the mutation_rate
    def mutation(self, child):
        mutate = np.random.choice([True,False], p=[self.mutation_rate, 1-self.mutation_rate])
        if mutate:
            pass

    # crossover the stats of the two parents
    def crossover(self, parent1, parent2, boidclass : Boids.Boids):
        #position = [(boidclass.positions[parent1][0] + boidclass.positions[parent2][0]) / 2,  (boidclass.positions[parent1][1] + boidclass.positions[parent2][1]) / 2]
        #velocity = [ (boidclass.velocities[parent1][0] + boidclass.velocities[parent2][0]) / 2, (boidclass.velocities[parent1][1] + boidclass.velocities[parent2][1]) / 2]

        return boidclass.crossover(parent1, parent2)           

    # create children from the two parents
    def make_children(self, parent1, parent2, boidclass : Boids.Boids):
        num_children = ceil(np.random.normal(2))
        children = []

        for _ in range(num_children):
            child = self.crossover(parent1, parent2, boidclass)
            self.mutation(child)
            children.append(child)
        
        return children

    # randomly pair parents from the population
    def pair_random(self, population):
        shuffle(population)
        group1 = population[int(len(population) / 2):]
        group2 = population[:int(len(population) / 2)]
        return zip(group1, group2)


    # create the next generation from the population
    def next_generation(self, population, boidclass : Boids.Boids):
        children = []
        for _, (parent1, parent2) in enumerate(self.pair_random(population)):
            children = children + self.make_children(parent1, parent2, boidclass)

        reshaped_list = [list(x) for x in zip(*children)]
        arguments = [len(children), 0, boidclass.width, boidclass.height] + reshaped_list

        # for trait in range(len(children[1])-1):
        #     test = children[trait][:]
        #     print("trait")
        #     print(test)

        next_generation_boidclass =  boidclass.__class__(*arguments)