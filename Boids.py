# Working version
import numpy as np

class Boids:
    def __init__(self, 
                 num_boids,
                 attributes,
                 environment,
                 width,
                 height 
                 ):
      
        #num_boids = attributes["num_boids"]
        # self.attributes = attributes
        self.positions = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))        
        self.velocities = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))
        coefficient_of_variation = attributes["coefficient_of_variation"]
        scale = attributes["scale"]
        self.environment = environment
        self.width = width
        self.height = height
        
        
        #self.trait_names = ['avoid_border_distance', 'avoid_border_strength', 'alignment_distance', 'cohesion_distance', 'separation_distance', 'alignment_strength', 'cohesion_strength', 'separation_strength', 'noise_strength', 'max_velocity']


        # TODO suggestion: instead of drawing different samples for the x and y strenght, make them the same. Maybe not necessary however.
        #coefficient_of_variation = 0.001

        for key, value in attributes.items():
            if key not in ["coefficient_of_variation", "scale"]:
                if value == 0:
                    standard_deviation = scale
                else:
                    standard_deviation = coefficient_of_variation * value
                    
                attribute = np.random.normal(value, standard_deviation, num_boids)
                setattr(self, key, attribute)
                
                # Process the key-value pair here

                #print(key, value)

        """
        self.alignment_distance = np.random.normal(attributes["alignment_distance"],    coefficient_of_variation * attributes["alignment_distance"],    num_boids) # alignment_distance 
        self.cohesion_distance = np.random.normal(attributes["cohesion_distance"],      coefficient_of_variation * attributes["cohesion_distance"],     num_boids) # cohesion_distance 
        self.separation_distance = np.random.normal(attributes["separation_distance"],  coefficient_of_variation * attributes["separation_distance"],   num_boids) #  separation_distance 
        self.alignment_strength = np.random.normal(attributes["alignment_strength"],    coefficient_of_variation * attributes["alignment_strength"],    num_boids) # alignment_strength 
        self.cohesion_strength = np.random.normal(attributes["cohesion_strength"],      coefficient_of_variation * attributes["cohesion_strength"],     num_boids) # cohesion_strength 
        self.separation_strength = np.random.normal(attributes["separation_strength"],  coefficient_of_variation * attributes["separation_strength"],   num_boids) # separation_strength 
        self.noise_strength = np.random.normal(attributes["noise_strength"],            coefficient_of_variation * attributes["noise_strength"],        num_boids) # noise_strength 
        self.avoid_border_distance = np.random.normal(attributes["avoid_border_distance"], coefficient_of_variation *avoid_border_distance, num_boids) 
        self.avoid_border_strength = np.random.normal(attributes["avoid_border_strength"], coefficient_of_variation *avoid_border_strength, num_boids) 
        """
        

        #self.trait_names = self.trait_names + ['avoid_border_distance', 'avoid_border_strength']
            



        '''
        self.alignment_strength = np.random.normal(alignment_strength, self.coefficient_of_variation, (num_boids, 2)) # alignment_strength 
        self.cohesion_strength = np.random.normal(cohesion_strength, self.coefficient_of_variation, (num_boids, 2)) # cohesion_strength 
        self.separation_strength = np.random.normal(separation_strength, self.coefficient_of_variation, (num_boids, 2)) # separation_strength 
        self.noise_strength = np.random.normal(noise_strength, self.coefficient_of_variation, (num_boids, 2)) # noise_strength  
        '''
    

        #self.max_velocity = np.random.normal(max_velocity, coefficient_of_variation, num_boids) # max_velocity

        self.positions_over_time = [self.positions]
        self.velocities_over_time = [self.velocities]

    
        # Optimization        
        #self.distances = sorted([("alignment", self.alignment_distance), ("cohesion", self.cohesion_distance), ("separation", self.separation_distance)], key=lambda x: x[1])     

    def avoid_border_rule(self, positions):
        avoid_force = np.zeros((len(positions), 2))

        # Calculate distance from each boid to the border
        distance_to_border = np.minimum(np.minimum(positions[:, 0], self.width - positions[:, 0]),
                                        np.minimum(positions[:, 1], self.height - positions[:, 1]))

        # Calculate steering force to avoid the border
        too_close = distance_to_border < self.avoid_border_distance
        avoid_force[too_close, 0] = np.sign(self.width / 2 - positions[too_close, 0])
        avoid_force[too_close, 1] = np.sign(self.height / 2 - positions[too_close, 1])

        # Scale the steering force by the avoid_border_strength
        #avoid_force *= self.avoid_border_strength[:, np.newaxis]

        return avoid_force


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

    def wrapped_borders(self, positions):
        positions[:, 0] = np.where(positions[:, 0] > self.width, positions[:, 0] - self.width, positions[:, 0])
        positions[:, 0] = np.where(positions[:, 0] < 0, positions[:, 0] + self.width, positions[:, 0])
        positions[:, 1] = np.where(positions[:, 1] > self.height, positions[:, 1] - self.height, positions[:, 1])
        positions[:, 1] = np.where(positions[:, 1] < 0, positions[:, 1] + self.height, positions[:, 1])
        return positions
    
    def hard_borders(self, positions):
        positions[:, 0] = np.where(positions[:, 0] > self.width, self.width, positions[:, 0])
        positions[:, 0] = np.where(positions[:, 0] < 0, 0, positions[:, 0])
        positions[:, 1] = np.where(positions[:, 1] > self.height, self.height, positions[:, 1])
        positions[:, 1] = np.where(positions[:, 1] < 0, 0, positions[:, 1])
        return positions
    
    def update_positions(self, positions):
        if self.environment == 'wrapped_borders':
            return self.wrapped_borders(positions)
        elif self.environment == 'hard_borders':
            return self.hard_borders(positions)

    def run_simulation(self, num_steps):
        for i in range(num_steps):
            self.step()
        return self.positions_over_time, self.velocities_over_time

    def crossover(self, parents, method):
        trait_dic = {}

        if method == "choice":
            for trait_name in self.trait_names:
                trait_dic[trait_name] = np.random.choice(getattr(self, trait_name)[parents], 1)[0]

        elif method == "max" or method == "mean":
            if method == "max":
                func = np.max
            elif method == "mean":
                func = np.mean

            for key, _ in self.attributes.items(): #, value
                if key not in ["coefficient_of_variation", "scale"]:
                    trait = getattr(self, key)
                    trait_dic[key] = func(trait[parents])
                    #setattr(self, key, np.array(trait_dic[key]))

            '''
                        trait_dic["avoid_border_distance"] = func(self.avoid_border_distance[parents])
            trait_dic["alignment_distance"] = func(self.alignment_distance[parents])
            trait_dic["cohesion_distance"] = func(self.cohesion_distance[parents])
            trait_dic["separation_distance"] = func(self.separation_distance[parents])
            trait_dic["avoid_border_strength"] = func(self.avoid_border_strength[parents])
            trait_dic["alignment_strength"] = func(self.alignment_strength[parents], axis=0)
            trait_dic["cohesion_strength"] = func(self.cohesion_strength[parents], axis=0)
            trait_dic["separation_strength"] = func(self.separation_strength[parents], axis=0)
            trait_dic["noise_strength"] = func(self.noise_strength[parents], axis=0)
            trait_dic["max_velocity"] = func(self.max_velocity[parents])
            
            '''


        return trait_dic
    
    def mutate(self, child, scale):
        for trait, value in child.items():
            if isinstance(value, list):
                child[trait] = list(map(lambda x: np.random.normal(x, abs(x*scale)), value))
            else:
                child[trait] = np.random.normal(value, abs(value*scale))
    
    def set_traits(self, trait_dic):
        for key, _ in self.attributes.items(): #, value
            if key not in ["coefficient_of_variation", "scale"]:
                setattr(self, key, np.array(trait_dic[key]))


        '''
        self.alignment_distance = np.array(trait_dic["alignment_distance"])
        self.cohesion_distance = np.array(trait_dic["cohesion_distance"])
        self.separation_distance = np.array(trait_dic["separation_distance"])
        self.alignment_strength = np.array(trait_dic["alignment_strength"])
        self.cohesion_strength = np.array(trait_dic["cohesion_strength"])
        self.separation_strength = np.array(trait_dic["separation_strength"])
        self.noise_strength = np.array(trait_dic["noise_strength"])
        self.max_velocity = np.array(trait_dic["max_velocity"])
        '''
        

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
    
