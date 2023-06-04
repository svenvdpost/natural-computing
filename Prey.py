# Working version
import numpy as np
import Boids

class Prey(Boids.Boids):
    def __init__(self, 
                 num_prey,
                 num_predator, 
                 width, 
                 height, 
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance, 
                 dodging_distance,
                 #vision_distance,
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength, 
                 dodging_strength,
                 noise_strength,
                 max_velocity):
        super().__init__(num_prey,
                        #num_predator, 
                        width, 
                        height, 
                        alignment_distance, 
                        cohesion_distance, 
                        separation_distance, 
                        #dodging_distance,
                        #vision_distance,
                        alignment_strength, 
                        cohesion_strength, 
                        separation_strength, 
                        #dodging_strength,
                        noise_strength,
                        max_velocity)
        
        #TODO implement traits
        scale = 0.001
        self.dodging_distance  = np.random.normal(dodging_distance, scale, num_predator) #  separation_distance 
        self.dodging_strength = np.random.normal(dodging_strength, scale, (num_prey, 2)) # separation_strength 

    
    def step_pygame(self, predator_positions, predator_velocities):
        #print(predator_positions)
        prey_distances = self.get_distances(self.positions)
        #print(f'prey_distances {prey_distances.shape}')
        #print(prey_distances)

        predator_distances = self.get_distances(predator_positions)
        #print(f'predator_distances {predator_distances.shape}')
        #print(predator_distances)

        alignment = self.alignment_rule(prey_distances)
        cohesion = self.cohesion_rule(prey_distances)
        separation = self.separation_rule(prey_distances)
        dodging = self.dodging_rule(predator_distances, predator_positions)

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength
        dodging_correction = (dodging + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.dodging_strength

        self.velocities += alignment_correction + cohesion_correction + separation_correction + dodging_correction
        self.limit_velocity()
        self.positions = self.wrap(self.positions + self.velocities)

        return self.positions, self.velocities
    
    def dodging_rule(self, distances, positions_2):
        close_boids = self.get_close_boids(self.dodging_distance, distances)
        dodging = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                #separation[i] = np.sum(self.positions[neighbors] - self.positions[i], axis=0)
                dodging[i] = np.sum(positions_2[neighbors] - self.positions[i], axis=0)
        return - dodging
    
    def crossover(self, parent1, parent2):
        genes = super().crossover(parent1, parent2)
        
        return genes
