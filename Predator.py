# Working version
import numpy as np
import Boids

class Predators(Boids.Boids):
    def __init__(self, 
                 num_predator, 
                 num_prey,
                 width, 
                 height, 
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance,
                 hunting_distance, 
                 #vision_distance,
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength,
                 hunting_strength, 
                 noise_strength,
                 max_velocity):
        super().__init__(num_predator, 
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
        self.hunting_distance  = np.random.normal(hunting_distance, scale, num_prey) #  separation_distance num_predator
        self.hunting_strength = np.random.normal(hunting_strength, scale, (num_predator, 2)) # separation_strength num_prey

        vision_trait = [10]*num_predator
        speed_trait = [10]*num_predator
        self.traits['vision', 'speed'] = vision_trait, speed_trait
    
    #TODO: implent hunting component
    def step_pygame(self, prey_positions, prey_velocities):
        predator_distances = self.get_distances(self.positions)

        prey_distances = self.get_distances(prey_positions)

        alignment = self.alignment_rule(predator_distances)
        cohesion = self.cohesion_rule(predator_distances)
        separation = self.separation_rule(predator_distances)
        hunting = self.hunting_rule(prey_distances, prey_positions)

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength
        hunting_correction = (hunting + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.hunting_strength

        self.velocities += alignment_correction + cohesion_correction + separation_correction + hunting_correction
        self.limit_velocity()
        self.positions = self.wrap(self.positions + self.velocities)

        return self.positions, self.velocities
    
    def hunting_rule(self, distances, positions_2):
        close_boids = self.get_close_boids(self.hunting_distance, distances)
        hunting = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                #separation[i] = np.sum(self.positions[neighbors] - self.positions[i], axis=0)
                hunting[i] = np.sum(positions_2[neighbors] - self.positions[i], axis=0)
        return hunting