# Working version
import numpy as np
import Boids

class Predators(Boids.Boids):
    def __init__(self, 
                 num_boids,
                 attributes,
                 environment,
                 width,
                 height
                 ):
        super().__init__(num_boids,
                         attributes,
                         environment,
                         width,
                         height
                         )

        # Get the number of prey boids and their corresponding initial attributes
        self.num_boids = num_boids
        self.attributes = attributes

        # Initialize the predator attributes
        predator_specific_attributes = ["hunting_strength", "hunting_distance", "elimination_distance"]
        coefficient_of_variation = attributes["coefficient_of_variation"]
        scale = attributes["scale"]

         # For all predator specific attributes draw a random sample for each boid
        for key, value in attributes.items():
            if key  in predator_specific_attributes:
                if value == 0:
                    # If the mean value is zero use a negative normal distribution to draw a sample
                    standard_deviation = scale
                    attribute = scale - np.random.normal(value, standard_deviation, num_boids)
                else:
                    # If the mean is not zero compute the standard deviation using a predefined coefficient of variation to draw samples from a normal distribution
                    standard_deviation = coefficient_of_variation * value
                    attribute = np.random.normal(value, standard_deviation, num_boids)

                setattr(self, key, attribute)

    
    def step_pygame(self, prey_positions, prey_velocities):
        
        # Get the distances to prey and predator boids
        predator_distances = self.get_distances(self.positions)
        prey_distances = self.get_distances(prey_positions)

        # Compute the forces for the different rules
        alignment_force = self.alignment_rule(predator_distances)
        cohesion_force = self.cohesion_rule(predator_distances)
        separation_force = self.separation_rule(predator_distances)
        hunting_force = self.hunting_rule(prey_distances, prey_positions)

        # Compute the rule specific correction of the boids velocities
        alignment_correction  = (alignment_force + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength[:, np.newaxis]) * self.alignment_strength[:, np.newaxis]
        cohesion_correction   = (cohesion_force + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength[:, np.newaxis]) * self.cohesion_strength[:, np.newaxis]
        separation_correction = (separation_force + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.separation_strength[:, np.newaxis]
        hunting_correction = (hunting_force + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.hunting_strength[:, np.newaxis]

        # Update the boids velocities according to the different rules
        self.velocities += alignment_correction + cohesion_correction + separation_correction + hunting_correction

        # If the environment uses hard borders, additionally compute and apply the force to avoid borders
        if self.environment == 'hard_borders':
            avoid_border = self.avoid_border_rule(self.positions)
            avoid_border_correction = (avoid_border + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.avoid_border_strength[:, np.newaxis]
            self.velocities += avoid_border_correction

            

        # Limit the velocity and update the prey positions
        self.limit_velocity()
        self.positions = self.update_positions(self.positions + self.velocities)

        # Eliminate close prey boids and update the predator specific kill count
        eliminate, predator_kill_counts = self.elimination(prey_distances)

        return self.positions, self.velocities, eliminate, predator_kill_counts
    
    # Compute the force to hunt prey boids
    def hunting_rule(self, distances, positions_2):
        hunting = np.zeros((len(self.positions), 2))

        # Calculate steering force to hunt prey boids
        for i in range(len(self.positions)):
            neighbors = self.get_close_boids(self.hunting_distance[i], distances[i])
            if any(neighbors):

                # For each predator boid compute the vector pointing in the direction of the closest prey boid 
                deltapos = (positions_2[neighbors] - self.positions[i])
                minpos = np.argmin(np.sqrt(np.sum(np.square(positions_2[neighbors] - self.positions[i]), axis=1)))
                hunting[i] = deltapos[minpos]
        return hunting    

    # Calculate whether a predator hunted down a prey boid
    def elimination(self, distances):
        predator_kill_counts = np.zeros(self.num_boids)
        caught_prey_idx = []
        for i in range(len(self.positions)):

            # Compute if prey boids are within the specified elimination distance and eliminate them
            caught_prey = self.get_close_boids(self.elimination_distance[i], distances[i])
            predator_kill_counts[i] = np.sum(caught_prey)
            if any(caught_prey):
                caught_prey_idx.extend(caught_prey.nonzero()[0])
        return list( dict.fromkeys(caught_prey_idx)), predator_kill_counts 
    

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
            genes["hunting_distance"] = func(self.hunting_distance[parents])
            genes["hunting_strength"] = func(self.hunting_strength[parents], axis=0)
            genes["elimination_distance"] = func(self.elimination_distance[parents])

        return genes
    
    # Mutate genes (traits) of the new generation of prey boids
    def mutate(self, child, scale):
        super().mutate(child, scale)

    # Helper function to set the traits for prey boids
    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.hunting_distance  = np.array(trait_dic["hunting_distance"])
        self.hunting_strength = np.array(trait_dic["hunting_strength"])
        self.elimination_distance = np.array(trait_dic["elimination_distance"])

