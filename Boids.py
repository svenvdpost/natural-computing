# Working version
import numpy as np

class Boids:
    def __init__(self, 
                 num_boids, 
                 width, 
                 height, 
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance, 
                 vision_distance,
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength, 
                 noise_strength,
                 max_velocity #Trait for predator and prey TODO: implement this as trait
                 ):
      
        self.num_boids = num_boids
        self.positions = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))        
        self.velocities = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))
        self.width = width
        self.height = height


        # TODO suggestion: instead of drawing different samples for the x and y strenght, make them the same. Maybe not necessary however.
        scale = 0.001

        self.alignment_distance = np.random.normal(alignment_distance, scale, num_boids) # alignment_distance 
        self.cohesion_distance = np.random.normal(cohesion_distance, scale, num_boids) # cohesion_distance 
        self.separation_distance = np.random.normal(separation_distance, scale, num_boids) #  separation_distance 
        self.alignment_strength = np.random.normal(alignment_strength, scale, (num_boids, 2)) # alignment_strength 
        self.cohesion_strength = np.random.normal(cohesion_strength, scale, (num_boids, 2)) # cohesion_strength 
        self.separation_strength = np.random.normal(separation_strength, scale, (num_boids, 2)) # separation_strength 
        self.noise_strength = np.random.normal(noise_strength, scale, (num_boids, 2)) # noise_strength
        
        if vision_distance != None:
            self.alignment_distance = vision_distance
            self.cohesion_distance = vision_distance
            self.separation_distance = vision_distance
            self.alignment_strength = vision_distance
            self.cohesion_strength = vision_distance
            self.separation_strength = vision_distance


        self.max_velocity = np.random.normal(max_velocity, scale, num_boids) # max_velocity 
        self.positions_over_time = [self.positions]
        self.velocities_over_time = [self.velocities]

        # Optimization        
        #self.distances = sorted([("alignment", self.alignment_distance), ("cohesion", self.cohesion_distance), ("separation", self.separation_distance)], key=lambda x: x[1])

        self.traits = {}

    # def step(self):

    #     distances = self.get_distances()

    #     alignment = self.alignment_rule(distances)
    #     cohesion = self.cohesion_rule(distances)
    #     separation = self.separation_rule(distances)

    #     alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
    #     cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
    #     separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength

    #     self.velocities += alignment_correction + cohesion_correction + separation_correction
    #     self.limit_velocity()
    #     self.positions = self.wrap(self.positions + self.velocities)
    #     self.positions_over_time.append(self.positions.copy())
    #     self.velocities_over_time.append(self.velocities.copy())
    
    # def step_pygame(self):
    #     distances = self.get_distances()

    #     alignment = self.alignment_rule(distances)
    #     cohesion = self.cohesion_rule(distances)
    #     separation = self.separation_rule(distances)

    #     alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength) * self.alignment_strength
    #     cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength) * self.cohesion_strength
    #     separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength) * self.separation_strength

    #     self.velocities += alignment_correction + cohesion_correction + separation_correction
    #     self.limit_velocity()
    #     self.positions = self.wrap(self.positions + self.velocities)

    #     return self.positions, self.velocities

    def get_distances(self):
        return np.sqrt(np.sum((self.positions[:, np.newaxis] - self.positions) ** 2, axis=2))

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

    def limit_velocity(self):
        speed = np.sqrt(np.sum(self.velocities ** 2, axis=1))
        too_fast = speed > self.max_velocity
        self.velocities[too_fast] = self.velocities[too_fast] / speed[too_fast, np.newaxis] * self.max_velocity[too_fast, np.newaxis]

    def wrap(self, positions):
        positions[:, 0] = np.where(positions[:, 0] > self.width, positions[:, 0] - self.width, positions[:, 0])
        positions[:, 0] = np.where(positions[:, 0] < 0, positions[:, 0] + self.width, positions[:, 0])
        positions[:, 1] = np.where(positions[:, 1] > self.height, positions[:, 1] - self.height, positions[:, 1])
        positions[:, 1] = np.where(positions[:, 1] < 0, positions[:, 1] + self.height, positions[:, 1])
        return positions

    def run_simulation(self, num_steps):
        for i in range(num_steps):
            self.step()
        return self.positions_over_time, self.velocities_over_time
