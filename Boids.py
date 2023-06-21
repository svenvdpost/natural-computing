# Working version
import numpy as np

class Boids:
    def __init__(self, 
                 num_boids,
                 coefficient_of_variation, 
                 width, 
                 height, 
                 alignment_distance, 
                 cohesion_distance, 
                 separation_distance, 
                 alignment_strength, 
                 cohesion_strength, 
                 separation_strength,
                 noise_strength,
                 max_velocity #Trait for predator and prey TODO: implement this as trait
                 ):
      
        self.num_boids = num_boids
        self.positions = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))        
        self.velocities = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))
        self.coefficient_of_variation = coefficient_of_variation
        self.width = width
        self.height = height
        
        self.trait_names = ['alignment_distance', 'cohesion_distance', 'separation_distance', 'alignment_strength', 'cohesion_strength', 'separation_strength', 'noise_strength', 'max_velocity']


        # TODO suggestion: instead of drawing different samples for the x and y strenght, make them the same. Maybe not necessary however.
        #coefficient_of_variation = 0.001

        self.alignment_distance = np.random.normal(alignment_distance, self.coefficient_of_variation*alignment_distance, num_boids) # alignment_distance 
        self.cohesion_distance = np.random.normal(cohesion_distance, self.coefficient_of_variation*cohesion_distance, num_boids) # cohesion_distance 
        self.separation_distance = np.random.normal(separation_distance, self.coefficient_of_variation*separation_distance, num_boids) #  separation_distance 
        self.alignment_strength = np.random.normal(alignment_strength, self.coefficient_of_variation*alignment_strength, num_boids) # alignment_strength 
        self.cohesion_strength = np.random.normal(cohesion_strength, self.coefficient_of_variation*cohesion_strength, num_boids) # cohesion_strength 
        self.separation_strength = np.random.normal(separation_strength, self.coefficient_of_variation*separation_strength, num_boids) # separation_strength 
        self.noise_strength = np.random.normal(noise_strength, self.coefficient_of_variation*noise_strength, num_boids) # noise_strength  


        '''
        self.alignment_strength = np.random.normal(alignment_strength, self.coefficient_of_variation, (num_boids, 2)) # alignment_strength 
        self.cohesion_strength = np.random.normal(cohesion_strength, self.coefficient_of_variation, (num_boids, 2)) # cohesion_strength 
        self.separation_strength = np.random.normal(separation_strength, self.coefficient_of_variation, (num_boids, 2)) # separation_strength 
        self.noise_strength = np.random.normal(noise_strength, self.coefficient_of_variation, (num_boids, 2)) # noise_strength  
        '''
    

        self.max_velocity = np.random.normal(max_velocity, coefficient_of_variation, num_boids) # max_velocity

        self.positions_over_time = [self.positions]
        self.velocities_over_time = [self.velocities]

        # Optimization        
        #self.distances = sorted([("alignment", self.alignment_distance), ("cohesion", self.cohesion_distance), ("separation", self.separation_distance)], key=lambda x: x[1])


    def get_distances(self, positions_2):
        return np.sqrt(np.sum((self.positions[:, np.newaxis] - positions_2) ** 2, axis=2))

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

    def crossover(self, parents):
        trait_dic = {}

        trait_dic["alignment_distance"] = np.max(self.alignment_distance[parents])
        trait_dic["cohesion_distance"] = np.max(self.cohesion_distance[parents])
        trait_dic["separation_distance"] = np.max(self.separation_distance[parents])
        trait_dic["alignment_strength"] = np.max(self.alignment_strength[parents], axis=0)
        trait_dic["cohesion_strength"] = np.max(self.cohesion_strength[parents], axis=0)
        trait_dic["separation_strength"] = np.max(self.separation_strength[parents], axis=0)
        trait_dic["noise_strength"] = np.max(self.noise_strength[parents], axis=0)
        trait_dic["max_velocity"] = np.max(self.max_velocity[parents])

        return trait_dic
    
    def mutate(self, child, scale):
        for trait, value in child.items():
            if isinstance(value, list):
                for v in value:
                    v = np.random.normal(v, scale)
            else:
                value = np.random.normal(value, scale)
    
    def set_traits(self, trait_dic):
        self.alignment_distance = np.array(trait_dic["alignment_distance"])
        self.cohesion_distance = np.array(trait_dic["cohesion_distance"])
        self.separation_distance = np.array(trait_dic["separation_distance"])
        self.alignment_strength = np.array(trait_dic["alignment_strength"])
        self.cohesion_strength = np.array(trait_dic["cohesion_strength"])
        self.separation_strength = np.array(trait_dic["separation_strength"])
        self.noise_strength = np.array(trait_dic["noise_strength"])
        self.max_velocity = np.array(trait_dic["max_velocity"])

    def show_boid(self, boid_id):
        ad = self.alignment_distance[boid_id]
        cd = self.cohesion_distance[boid_id]
        sd = self.separation_distance[boid_id]
        als = self.alignment_strength[boid_id]
        cs = self.cohesion_strength[boid_id]
        ss = self.separation_strength[boid_id]
        ns = self.noise_strength[boid_id]
        mv = self.max_velocity[boid_id]

        return f"ad={ad}, cd={cd}, sd={sd}, als={als}, cs={cs}, ss={ss}, ns={ns}, mv={mv}"
    
    def get_trait_names(self):
        return self.trait_names
