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
        
        self.next_positions = []
        self.next_velocity = []
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
        position = (boidclass.positions[parent1] + boidclass.positions[parent2]) / 2
        velocity = (boidclass.velocities[parent1] + boidclass.velocities[parent2]) / 2
        alignment_distance = (boidclass.alignment_distance[parent1] + boidclass.alignment_distance[parent2]) / 2
        cohesion_distance = (boidclass.cohesion_distance[parent1] + boidclass.cohesion_distance[parent2]) / 2
        separation_distance = (boidclass.separation_distance[parent1] + boidclass.separation_distance[parent2]) / 2
        vision_distance = (boidclass.vision_distance[parent1] + boidclass.vision_distance[parent2]) / 2
        alignment_strength = (boidclass.alignment_strength[parent1] + boidclass.alignment_strength[parent2]) / 2
        cohesion_strength = (boidclass.cohesion_strength[parent1] + boidclass.cohesion_strength[parent2]) / 2
        separation_strength = (boidclass.separation_strength[parent1] + boidclass.separation_strength[parent2]) / 2
        noise_strength = (boidclass.noise_strength[parent1] + boidclass.noise_strength[parent2]) / 2
        max_velocity =(boidclass.max_velocity[parent1] + boidclass.max_velocity[parent2]) / 2
        traits = {}

        for trait, traitlist in boidclass.traits.items():
            traits[trait] = (traitlist[parent1] + traitlist[parent2]) / 2

        return (position, velocity, alignment_distance, cohesion_distance, separation_distance, vision_distance,
                alignment_strength, cohesion_strength, separation_strength, noise_strength, max_velocity, traits)
            

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
        group1 = population[int(len(population) / 2):]
        group2 = population[:int(len(population) / 2)]

        return zip(group1, group2)


    # create the next generation from the population
    def next_generation(self, population, boidclass : Boids.Boids):

        for parent1, parent2 in self.pair_random(population):
            children = self.make_children(parent1, parent2, boidclass)
            for child in children:
                self.next_positions.append(child[0])
                self.next_velocity.append(child[1])
                self.next_alignment_distance.append(child[2])
                self.next_cohesion_distance.append(child[3])
                self.next_separation_distance.append(child[4])
                self.next_vision_distance.append(child[5])
                self.next_alignment_strength.append(child[6])
                self.next_cohesion_strength.append(child[7])
                self.next_separation_strength.append(child[8])
                self.next_noise_strength.append(child[9])
                self.next_max_velocity.append(child[10])
                self.next_traits = child[11]

        next_generation_boidclass =  boidclass.__new__(len(children), boidclass.width, boidclass.height, np.array(self.next_alignment_distance))