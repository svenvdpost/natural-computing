# Working version
import numpy as np
import Boids

class Prey(Boids.Boids):
    def __init__(self, 
                 num_boids, 
                 width, 
                 height, 
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance, 
                 dodging_distance,
                 vision_distance,
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength, 
                 dodging_strength,
                 noise_strength,
                 max_velocity):
        super().__init__(num_boids, 
                        width, 
                        height, 
                        alignment_distance, 
                        cohesion_distance, 
                        separation_distance, 
                        dodging_distance,
                        vision_distance,
                        alignment_strength, 
                        cohesion_strength, 
                        separation_strength, 
                        dodging_strength,
                        noise_strength,
                        max_velocity)
        
        #TODO implement traits

        vision_trait = [10]*num_boids
        speed_trait = [10]*num_boids
        self.traits['vision', 'speed'] = vision_trait, speed_trait
    
    def step_pygame(self, predator_positions, predator_velocities):
        #print(predator_positions)
        prey_distances = self.get_prey_distances()

        predator_distances = self.get_predator_distances(predator_positions)

        alignment = self.alignment_rule(prey_distances)
        cohesion = self.cohesion_rule(prey_distances)
        separation = self.separation_rule(prey_distances)

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength

        self.velocities += alignment_correction + cohesion_correction + separation_correction
        self.limit_velocity()
        self.positions = self.wrap(self.positions + self.velocities)

        return self.positions, self.velocities
    

