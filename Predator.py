# Working version
import numpy as np
import Boids

class Predators(Boids.Boids):
    def __init__(self, 
                 num_predator, 
                 scale,
                 width, 
                 height, 
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
                        scale,
                        width, 
                        height, 
                        alignment_distance, 
                        cohesion_distance, 
                        separation_distance,
                        alignment_strength, 
                        cohesion_strength, 
                        separation_strength,
                        noise_strength,
                        max_velocity)
    
        scale = 0.001
        self.hunting_distance  = np.random.normal(hunting_distance, scale, num_predator) #  separation_distance num_predator
        self.hunting_strength = np.random.normal(hunting_strength, scale, (num_predator, 2)) # separation_strength num_prey
        self.num_predator = num_predator
        self.hunting_strength = np.random.normal(hunting_strength, self.scale, (num_predator, 2)) # separation_strength num_prey
        self.elimination_distance = np.random.normal(elimination_distance, self.scale, num_predator) #  separation_distance num_predator

    
    def step_pygame(self, prey_positions, prey_velocities):
        predator_distances = self.get_distances(self.positions)
        

        prey_distances = self.get_distances(prey_positions)


        alignment = self.alignment_rule(predator_distances)
        cohesion = self.cohesion_rule(predator_distances)
        separation = self.separation_rule(predator_distances)
        hunting = self.hunting_rule(prey_distances, prey_positions)

        eliminate, predator_kill_counts = self.elimination(prey_distances)
        #print(eliminate)

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength
        hunting_correction = (hunting + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.hunting_strength

        self.velocities += alignment_correction + cohesion_correction + separation_correction + hunting_correction
        self.limit_velocity()
        self.positions = self.wrap(self.positions + self.velocities)

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
    

    def crossover(self, parents):
        genes = super().crossover(parents)

        hunting_distance = np.mean(np.take(self.hunting_distance, parents))
        hunting_strength = np.mean(np.take(self.hunting_strength, parents))
        elimination_distance = np.mean(np.take(self.elimination_distance, parents))

        return genes[:3] + [hunting_distance, elimination_distance] + genes[3:6] + [hunting_strength] + genes[6:]
    
    def set_traits(self, trait_matrix):
        pass