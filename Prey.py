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
        
        # Get the number of prey boids and their corresponding initial attributes
        self.num_boids = num_boids 
        self.attributes = attributes

        # Initialize the prey attributes
        prey_specific_attributes = ["dodging_strength", "dodging_distance"]
        coefficient_of_variation = attributes["coefficient_of_variation"]
        scale = attributes["scale"]

        # For all prey specific attributes draw a random sample for each boid
        for key, value in attributes.items():
            if key in prey_specific_attributes:
                if value == 0:
                    # If the mean value is zero use a negative normal distribution to draw a sample
                    standard_deviation = scale
                    attribute = scale - np.random.normal(value, standard_deviation, num_boids)
                else:
                    # If the mean is not zero compute the standard deviation using a predefined coefficient of variation to draw samples from a normal distribution
                    standard_deviation = coefficient_of_variation * value
                    attribute = np.random.normal(value, standard_deviation, num_boids) 

                setattr(self, key, attribute)

    # Compute the positions and velocities of prey boids on the next simulation step
    def step_pygame(self, predator_positions):

        self.num_predator = len(predator_positions)

        # Get the distances to prey and predator boids
        prey_distances = self.get_distances(self.positions)
        predator_distances = self.get_distances(predator_positions)

        # Compute the forces for the different rules
        alignment_force = self.alignment_rule(prey_distances)
        cohesion_force = self.cohesion_rule(prey_distances)
        separation_force = self.separation_rule(prey_distances)
        dodging_force = self.dodging_rule(predator_distances, predator_positions)
        
        # Compute the rule specific correction of the boids velocities
        alignment_correction  = (alignment_force + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength[:, np.newaxis]) * self.alignment_strength[:, np.newaxis]
        cohesion_correction   = (cohesion_force + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength[:, np.newaxis]) * self.cohesion_strength[:, np.newaxis]
        separation_correction = (separation_force + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.separation_strength[:, np.newaxis]
        dodging_correction = (dodging_force + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.dodging_strength[:, np.newaxis]

        # Update the boids velocities according to the different rules
        self.velocities += alignment_correction + cohesion_correction + separation_correction + dodging_correction

        # If the environment uses hard borders, additionally compute and apply the force to avoid borders
        if self.environment == 'hard_borders':
            avoid_border = self.avoid_border_rule(self.positions)
            avoid_border_correction = (avoid_border + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.avoid_border_strength[:, np.newaxis]
            self.velocities += avoid_border_correction


        # Limit the velocity and update the prey positions
        self.limit_velocity()
        self.positions = self.update_positions(self.positions + self.velocities)

        return self.positions, self.velocities
    
    # Compute the force to dodge predators
    def dodging_rule(self, distances, positions_2):

        # Get close predator boids
        dodging_distance_step = np.repeat(self.dodging_distance[:, np.newaxis], self.num_predator, axis=1)
        close_boids = self.get_close_boids(dodging_distance_step, distances)

        # Calculate steering force to dodge predators
        dodging_force = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):

                # For each prey boid compute the vector pointing in the opposite direction of the closest predator
                dodging_force[i] = - np.sum(positions_2[neighbors] - self.positions[i], axis=0)
        return dodging_force
    
    def crossover(self, parents, method):
        genes = super().crossover(parents, method)

        # Use the specified crossover method
        if method == "choice":
            # Randomly choose traits traits of the parents to produce new children
            for trait_name in self.trait_names:
                genes[trait_name] = np.random.choice(getattr(self, trait_name)[parents], 1)[0]

        # Use either the maximum or the average trait value of the parents for their children
        elif method == "max" or method == "mean":
            if method == "max":
                func = np.max
            elif method == "mean":
                func = np.mean

            # Set the new attributes for the next generation of boids
            genes["dodging_distance"] = func(self.dodging_distance[parents])
            genes["dodging_strength"] = func(self.dodging_strength[parents], axis=0)

        return genes
    
    # Mutate genes (traits) of the new generation of prey boids
    def mutate(self, child, scale):
        super().mutate(child, scale)

    # Helper function to set the traits for prey boids
    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.dodging_distance = np.array(trait_dic["dodging_distance"])
        self.dodging_strength = np.array(trait_dic["dodging_strength"])

