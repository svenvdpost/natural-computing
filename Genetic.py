import Prey
import Predator
import Boids
import Simulation

from random import shuffle
import numpy as np
from math import ceil
import copy

class Genetic:

    def __init__(self, simulation : Simulation.Simulation, mutation_rate_prey, mutation_rate_predator, mutation_scale) -> None:
        self.simulation = simulation
        self.mutation_rate_prey = mutation_rate_prey # probability of prey mutating
        self.mutation_rate_predator = mutation_rate_predator # probability of predatoro mutating
        self.mutation_scale = mutation_scale # the mutation range (how much mutated values deviate from the inherited ones)
        self.crossover_method = None # method of crossover for a generation

    # mutate the stats of the boids proportional to the mutation_rate
    def mutation(self, child, boidclass : Boids.Boids):

        rate = self.mutation_rate_prey if isinstance(boidclass, Prey.Prey) else self.mutation_rate_predator

        mutate = np.random.choice([True,False], p=[rate, 1-rate])
        if mutate:
            boidclass.mutate(child, self.mutation_scale)

    # crossover the stats of the two parents
    def crossover(self, parents, boidclass : Boids.Boids):
        return boidclass.crossover(parents, self.crossover_method)
    
    # create children from the two parents
    def make_children(self, parents, boidclass : Boids.Boids):
        num_children = ceil(np.random.normal(2, 1))
        children = []

        for _ in range(num_children):
            child = self.crossover(parents, boidclass)
            self.mutation(child, boidclass)
            children.append(child)
        
        return children

    # randomly pair parents from the population
    def pair_random(self, population):
        shuffle(population)
        group1 = population[int(len(population) / 2):]
        group2 = population[:int(len(population) / 2)]
        return zip(group1, group2)

    # Select prey and predator boids for crossover
    def crossover_selection(self, elimination_order, predator_kill_counts, prey_survival_times, time_step):
        
        sim = self.simulation # easier readability

        # Predator crossover selection (select the predators that eliminated most boids)
        predator_selection_weights = predator_kill_counts**sim.predator_selection_weight
        predator_selection_weights = np.where(predator_selection_weights == 0, 0.01, predator_selection_weights) # make sure there are no zero values in the probabilities
        predator_selection_probabilities = predator_selection_weights / np.sum( predator_selection_weights)

        size = min(sim.num_predator_crossover, sim.num_predator) # in case the new population is smaller than the last generation

        predator_crossover_idx = list(np.random.choice(range(sim.num_predator),size=size,replace=False, p=predator_selection_probabilities))        
        
        # Prey crossover selection (select the prey that survived the longest)
        prey_selection_weights = list(np.asarray(prey_survival_times).astype('float64')**sim.prey_selection_weight)
        prey_selection_probabilities =  prey_selection_weights / np.sum( prey_selection_weights)

        size = min(sim.num_prey_crossover, sim.num_prey)

        if len(elimination_order) >= sim.num_prey: # if all the prey is eaten
            prey_crossover_idx = list(np.random.choice(elimination_order,size=size,replace=False, p=prey_selection_probabilities))

        elif time_step >= sim.max_time_steps: # if the maximum time steps has been reached (meaning some prey hase survived)
            survivors = list(set(range(sim.num_prey)) - set(elimination_order))
            num_select_survivors = np.min([size, len(survivors)])

            # Less survivors than desired number of prey boids for cross over?
            num_remaining = size - num_select_survivors
            prey_crossover_idx = list(np.random.choice(survivors,size=num_select_survivors,replace=False))

            # Select the remaining boids for crossover depending on their survival time
            if num_remaining:
                prey_crossover_idx.extend(np.random.choice(elimination_order,size=num_remaining,replace=False, p=prey_selection_probabilities))

        return prey_crossover_idx, predator_crossover_idx

    # find the best performing boid of a generation
    def alpha_boids(self, elimination_order, predator_kill_counts):
        survivors = list(set(range(self.simulation.num_prey)) - set(elimination_order))

        if survivors:
            alpha_prey = np.random.choice(survivors,size=1)[0] # one of the surviving prey
        else:
            alpha_prey = elimination_order[-1] # the last eaten prey

        alpa_predator = np.argmax(predator_kill_counts) # predator with the most kills

        return alpha_prey, alpa_predator

    # generate next generation in a natural setting: pair boid with probability of offspring
    def natural_procreation(self, population, boidclass : Boids.Boids):
        children = []
        for _, (parent1, parent2) in enumerate(self.pair_random(population)):
            children = children + self.make_children([parent1, parent2], boidclass)
        return children

    # generate next generation in a fixed setting: extract mean child and fill population with copies
    def fixed_procreation(self, population, boidclass : Boids.Boids):
        mean_child = self.crossover(population, boidclass)
        children = [copy.deepcopy(mean_child) for _ in range(boidclass.num_boids)]
        for child in children:
            self.mutation(child, boidclass)

        return children

    # create the next generation from the population
    def next_generation(self, population, boidclass : Boids.Boids, procreation, crossover_method):
        children = []

        self.crossover_method = crossover_method # set the crossover method for this generation

        if procreation == "natural":
            children = self.natural_procreation(population, boidclass)
        elif procreation == "fixed":
            children = self.fixed_procreation(population, boidclass)

        # store the children traits in a dictionary
        traits_dic = {}
        for child in children:
            for trait, value in child.items():
                try:
                    traits_dic[trait].append(value)
                except:
                    traits_dic[trait] = [value]

        next_generation_boidclass =  boidclass.__class__(*[len(children), boidclass.attributes, boidclass.environment, boidclass.width, boidclass.height] ) # innit dummy class to overwrite later
        next_generation_boidclass.set_traits(traits_dic)

        return next_generation_boidclass
