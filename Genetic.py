import Prey
import Predator
import Boids
import Simulation

from random import shuffle
import numpy as np
from math import ceil
import copy

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
    def crossover(self, parents, boidclass : Boids.Boids):
        return boidclass.crossover(parents)
    
    # create children from the two parents
    def make_children(self, parents, boidclass : Boids.Boids):
        num_children = ceil(np.random.normal(5, 4))
        children = []

        for _ in range(num_children):
            child = self.crossover(parents, boidclass)
            self.mutation(child)
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
        predator_selection_weights = predator_kill_counts**sim.kill_counts_scaling_factor
        predator_selection_weights = np.where(predator_selection_weights == 0, 0.01, predator_selection_weights)
        predator_selection_probabilities = predator_selection_weights / np.sum( predator_selection_weights)

        size = min(sim.num_predator_crossover, sim.num_predator)

        predator_crossover_idx = list(np.random.choice(range(sim.num_predator),size=size,replace=False, p=predator_selection_probabilities))
        
        
        # Prey crossover selection
        prey_selection_weights = list(np.array(prey_survival_times)**sim.survival_time_scaling_factor)
        prey_selection_probabilities =  prey_selection_weights / np.sum( prey_selection_weights)

        size = min(sim.num_prey_crossover, sim.num_prey)

        if len(elimination_order) >= sim.num_prey:
            prey_crossover_idx = list(np.random.choice(elimination_order,size=size,replace=False, p=prey_selection_probabilities))

        elif time_step >= sim.max_time_steps:
            survivors = list(set(range(sim.num_prey)) - set(elimination_order))
            num_select_survivors = np.min([size, len(survivors)])

            # Less survivors than desired number of prey boids for cross over?
            num_remaining = size - num_select_survivors
            prey_crossover_idx = list(np.random.choice(survivors,size=num_select_survivors,replace=False))

            # Select the remaining boids for crossover depending on their survival time
            if num_remaining:
                prey_crossover_idx.extend(np.random.choice(elimination_order,size=num_remaining,replace=False, p=prey_selection_probabilities))

        return prey_crossover_idx, predator_crossover_idx


    def natural_procreation(self, population, boidclass : Boids.Boids):
        children = []
        for _, (parent1, parent2) in enumerate(self.pair_random(population)):
            children = children + self.make_children([parent1, parent2], boidclass)
        return children

    def fixed_procreation(self, population, boidclass : Boids.Boids):
        mean_child = self.crossover(population, boidclass)
        children = [copy.deepcopy(mean_child) for _ in range(boidclass.num_boids)]
        for child in children:
            self.mutation(child)

        return children

    # create the next generation from the population
    def next_generation(self, population, boidclass : Boids.Boids, procreation):
        children = []

        while children == []:
            if procreation == "natural":
                children = self.natural_procreation(population, boidclass)
            elif procreation == "fixed":
                children = self.fixed_procreation(population, boidclass)

        # reshape list from child orented to trait oriented
        reshaped_list = [list(x) for x in zip(*list(map(lambda x: x.values(), children)))]

        # store the children traits in a dictionary
        traits_dic = {}
        for child in children:
            for trait, value in child.items():
                try:
                    traits_dic[trait].append(value)
                except:
                    traits_dic[trait] = [value]

        next_generation_boidclass =  boidclass.__class__(*([len(children), 0, boidclass.width, boidclass.height, boidclass.environment] + list(np.ones(len(reshaped_list))))) # innit dummy class to overwrite later
        next_generation_boidclass.set_traits(traits_dic)

        return next_generation_boidclass
