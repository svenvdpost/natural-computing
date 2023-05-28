# Working version
import numpy as np
import Boid

class Prey(Boid.Boids):
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
      
        def step(self):

            distances = self.get_distances()

            alignment = self.alignment_rule(distances)
            cohesion = self.cohesion_rule(distances)
            separation = self.separation_rule(distances)

            alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
            cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
            separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength

            self.velocities += alignment_correction + cohesion_correction + separation_correction
            self.limit_velocity()
            self.positions = self.wrap(self.positions + self.velocities)
            self.positions_over_time.append(self.positions.copy())
            self.velocities_over_time.append(self.velocities.copy())
    
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

    def get_prey_distances(self):
        return np.sqrt(np.sum((self.positions[:, np.newaxis] - self.positions) ** 2, axis=2))
    
    def get_predator_distances(self, predator_positions):
        print('Call function')
        return np.sqrt(np.sum((predator_positions[:, np.newaxis] - self.positions) ** 2, axis=2))

    def get_close_boids(self, rule_distance, distances):
        return (distances < rule_distance) & (distances > 0)

    def alignment_rule(self, distances):
        close_boids = self.get_close_boids(self.alignment_distance, distances)
        alignment = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                alignment[i] = np.mean(self.velocities[neighbors], axis=0)
        return alignment 

    def cohesion_rule(self, distances):
        close_boids = self.get_close_boids(self.cohesion_distance, distances)
        cohesion = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                cohesion[i] = np.mean(self.positions[neighbors], axis=0) - self.positions[i]
        return cohesion 

    def separation_rule(self, distances):
        close_boids = self.get_close_boids(self.separation_distance, distances)
        separation = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                separation[i] = np.sum(self.positions[neighbors] - self.positions[i], axis=0)
        return - separation 
    
    def dodging_rule(self, distances):
        close_boids = self.get_close_boids(self.dodging_distance, distances)
        separation = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):
                separation[i] = np.sum(self.positions[neighbors] - self.positions[i], axis=0)
        return - separation 

    def limit_velocity(self):
        speed = np.sqrt(np.sum(self.velocities ** 2, axis=1))
        too_fast = speed > self.max_velocity
        self.velocities[too_fast] = self.velocities[too_fast] / speed[too_fast, np.newaxis] * self.max_velocity

    def wrap(self, positions):
        positions[:, 0] = np.where(positions[:, 0] > self.width, positions[:, 0] - self.width, positions[:, 0])
        positions[:, 0] = np.where(positions[:, 0] < 0, positions[:, 0] + self.width, positions[:, 0])
        positions[:, 1] = np.where(positions[:, 1] > self.height, positions[:, 1] - self.height, positions[:, 1])
        positions[:, 1] = np.where(positions[:, 1] < 0, positions[:, 1] + self.height, positions[:, 1])
        return positions
