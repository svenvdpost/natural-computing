# Working version
import numpy as np
import Boids

class Prey(Boids.Boids):
    def __init__(self, 
                 num_prey,
                 scale,
                 width, 
                 height, 
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
        
        #TODO implement traits
        self.scale = scale
        self.dodging_distance  = np.random.normal(dodging_distance, self.scale, num_prey) #  separation_distance 
        self.dodging_strength = np.random.normal(dodging_strength, self.scale, (num_prey, 2)) # separation_strength 

    
    def step_pygame(self, predator_positions, predator_velocities):

        self.num_predator = len(predator_positions)

        prey_distances = self.get_distances(self.positions)
        predator_distances = self.get_distances(predator_positions)

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
        genes["dodging_strength"] = np.mean(self.dodging_strength[parents], axis=0)

        return genes
    
    def mutate(self, child, scale):
        super().mutate(child, scale)

    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.dodging_distance = np.array(trait_dic["dodging_distance"])
        self.dodging_strength = np.array(trait_dic["dodging_strength"])

    def show_boid(self, boid_id):
        output = super().show_boid(boid_id)

        dd = self.dodging_distance[boid_id]
        ds = self.dodging_strength[boid_id]

        return output + f",hd={dd}, cd={ds}"
