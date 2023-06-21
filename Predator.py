# Working version
import numpy as np
import Boids

class Predators(Boids.Boids):
    def __init__(self, 
                 num_predator, 
                 coefficient_of_variation,
                 width, 
                 height, 
                 environment,
                 avoid_border_distance,
                 avoid_border_strength,
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance,
                 hunting_distance, 
                 elimination_distance,
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength,
                 hunting_strength, 
                 noise_strength,
                 max_velocity):
        super().__init__(num_predator, 
                        coefficient_of_variation,
                        width, 
                        height, 
                        environment,
                        avoid_border_distance,
                        avoid_border_strength,
                        alignment_distance, 
                        cohesion_distance, 
                        separation_distance,
                        alignment_strength, 
                        cohesion_strength, 
                        separation_strength,
                        noise_strength,
                        max_velocity)
    
        self.hunting_distance  = np.random.normal(hunting_distance, self.coefficient_of_variation*hunting_distance, num_predator) #  separation_distance num_predator
        self.hunting_strength = np.random.normal(hunting_strength, self.coefficient_of_variation*hunting_strength, num_predator) # separation_strength num_prey
        self.num_predator = num_predator
        self.elimination_distance = np.random.normal(elimination_distance, self.coefficient_of_variation*elimination_distance, num_predator) #  separation_distance num_predator

        self.trait_names = super().get_trait_names() + ['hunting_distance', 'hunting_strength', 'elimination_distance']

    
    def step_pygame(self, prey_positions, prey_velocities):
        predator_distances = self.get_distances(self.positions)
        

        prey_distances = self.get_distances(prey_positions)


        alignment = self.alignment_rule(predator_distances)
        cohesion = self.cohesion_rule(predator_distances)
        separation = self.separation_rule(predator_distances)
        hunting = self.hunting_rule(prey_distances, prey_positions)

        eliminate, predator_kill_counts = self.elimination(prey_distances)

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength[:, np.newaxis]) * self.alignment_strength[:, np.newaxis]
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength[:, np.newaxis]) * self.cohesion_strength[:, np.newaxis]
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.separation_strength[:, np.newaxis]
        hunting_correction = (hunting + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.hunting_strength[:, np.newaxis]

        self.velocities += alignment_correction + cohesion_correction + separation_correction + hunting_correction

        if self.environment == 'hard_borders':
            avoid_border = self.avoid_border_rule(self.positions)
            avoid_border_correction = (avoid_border + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.avoid_border_strength[:, np.newaxis]
            self.velocities += avoid_border_correction

            
        self.limit_velocity()

        self.positions = self.update_positions(self.positions + self.velocities)

        return self.positions, self.velocities, eliminate, predator_kill_counts
    
    def hunting_rule(self, distances, positions_2):
        hunting = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = self.get_close_boids(self.hunting_distance[i], distances[i])
            if any(neighbors):
                deltapos = (positions_2[neighbors] - self.positions[i])
                minpos = np.argmin(np.sqrt(np.sum(np.square(positions_2[neighbors] - self.positions[i]), axis=1)))
                hunting[i] = deltapos[minpos]
        return hunting    


    def elimination(self, distances):
        predator_kill_counts = np.zeros(self.num_predator)
        caught_prey_idx = []
        for i in range(len(self.positions)):
            caught_prey = self.get_close_boids(self.elimination_distance[i], distances[i])
            predator_kill_counts[i] = np.sum(caught_prey)
            if any(caught_prey):
                caught_prey_idx.extend(caught_prey.nonzero()[0])
        return list( dict.fromkeys(caught_prey_idx)), predator_kill_counts 
    

    def crossover(self, parents, method):
        genes = super().crossover(parents, method)

        if method == "choice":
            for trait_name in self.trait_names:
                genes[trait_name] = np.random.choice(getattr(self, trait_name)[parents], 1)[0]

        elif method == "max" or method == "mean":
            if method == "max":
                func = np.max
            elif method == "mean":
                func = np.mean

            genes["hunting_distance"] = func(self.hunting_distance[parents])
            genes["hunting_strength"] = func(self.hunting_strength[parents], axis=0)
            genes["elimination_distance"] = func(self.elimination_distance[parents])

        return genes
    
    def mutate(self, child, scale):
        super().mutate(child, scale)

    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.hunting_distance  = np.array(trait_dic["hunting_distance"])
        self.hunting_strength = np.array(trait_dic["hunting_strength"])
        self.elimination_distance = np.array(trait_dic["elimination_distance"])

