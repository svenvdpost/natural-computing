# Working version
import numpy as np
import Boids

class Prey(Boids.Boids):
    def __init__(self,
                 num_boids,
                 attributes,
                 environment,
                 width,
                 height,
                 ):
        super().__init__(num_boids,
                         attributes,
                         environment,
                         width,
                         height)
        
        #TODO implement traits

        # (self.prey_attributes, self.environment, self.width, self.height)
        self.num_boids = num_boids 
        self.attributes = attributes
        coefficient_of_variation = attributes["coefficient_of_variation"]
        scale = attributes["scale"]

        prey_specific_attributes = ["dodging_strength", "dodging_distance"]

        for key, value in attributes.items():
            if key  in prey_specific_attributes:
                if value == 0:
                    standard_deviation = scale
                else:
                    standard_deviation = coefficient_of_variation * value
                    
                attribute = np.random.normal(value, standard_deviation, num_boids)
                setattr(self, key, attribute)

        #self.dodging_distance  = np.random.normal(attributes["dodging_distance"], coefficient_of_variation * attributes["dodging_distance"], num_boids) #  separation_distance 
        #self.dodging_strength = np.random.normal(attributes["dodging_strength"], coefficient_of_variation * attributes["dodging_strength"], num_boids) # separation_strength 
        
        #self.trait_names = super().get_trait_names() + ['dodging_distance', 'dodging_strength']

    
    def step_pygame(self, predator_positions):

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
    
    def crossover(self, parents, method):
        genes = super().crossover(parents, method)

        if method == "choice":
            for trait_name in self.trait_names:
                genes[trait_name] = np.random.choice(getattr(self, trait_name)[parents], 1)[0]

        elif method == "max" or method == "mean":
            if method == "max":
                func = np.max
            elif method == "mean":
                func = np.mean
        
            genes["dodging_distance"] = func(self.dodging_distance[parents])
            genes["dodging_strength"] = func(self.dodging_strength[parents], axis=0)

        return genes
    
    def mutate(self, child, scale):
        super().mutate(child, scale)

    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.dodging_distance = np.array(trait_dic["dodging_distance"])
        self.dodging_strength = np.array(trait_dic["dodging_strength"])

