# Working version
import numpy as np
import Boids

class Prey(Boids.Boids):
    def __init__(self, 
                 num_prey,
                 coefficient_of_variation,
                 width, 
                 height, 
                 environment,
                 avoid_border_distance,
                 avoid_border_strength,
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance, 
                 dodging_distance,
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength, 
                 dodging_strength,
                 noise_strength,
                 max_velocity):
        super().__init__(num_prey,
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
        
        #TODO implement traits
        self.coefficient_of_variation = coefficient_of_variation
        self.dodging_distance  = np.random.normal(dodging_distance, self.coefficient_of_variation*dodging_distance, num_prey) #  separation_distance 
        self.dodging_strength = np.random.normal(dodging_strength, self.coefficient_of_variation*dodging_strength, num_prey) # separation_strength 
        
        self.trait_names = super().get_trait_names() + ['dodging_distance', 'dodging_strength']

    
    def step_pygame(self, predator_positions, predator_velocities):

        self.num_predator = len(predator_positions)

        prey_distances = self.get_distances(self.positions)
        predator_distances = self.get_distances(predator_positions)

        alignment = self.alignment_rule(prey_distances)
        cohesion = self.cohesion_rule(prey_distances)
        separation = self.separation_rule(prey_distances)
        dodging = self.dodging_rule(predator_distances, predator_positions)
        

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength[:, np.newaxis]) * self.alignment_strength[:, np.newaxis]
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength[:, np.newaxis]) * self.cohesion_strength[:, np.newaxis]
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.separation_strength[:, np.newaxis]
        dodging_correction = (dodging + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.dodging_strength[:, np.newaxis]

        self.velocities += alignment_correction + cohesion_correction + separation_correction + dodging_correction


        if self.environment == 'hard_borders':
            avoid_border = self.avoid_border_rule(self.positions)
            avoid_border_correction = (avoid_border + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.avoid_border_strength[:, np.newaxis]
            self.velocities += avoid_border_correction



        self.limit_velocity()

        self.positions = self.update_positions(self.positions + self.velocities)

        return self.positions, self.velocities
    
    def dodging_rule(self, distances, positions_2):

        dodging_distance_step = np.repeat(self.dodging_distance[:, np.newaxis], self.num_predator, axis=1)

        close_boids = self.get_close_boids(dodging_distance_step, distances)
        dodging = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                dodging[i] = np.sum(positions_2[neighbors] - self.positions[i], axis=0)
        return - dodging
    
    def crossover(self, parents):
        genes = super().crossover(parents)
        
        genes["dodging_distance"] = np.mean(self.dodging_distance[parents])
        genes["dodging_strength"] = np.mean(self.dodging_strength[parents])

        return genes

    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.dodging_distance = np.array(trait_dic["dodging_distance"])
        self.dodging_strength = np.array(trait_dic["dodging_strength"])
