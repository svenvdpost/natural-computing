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

        # Initialize the initial positions and velocities of the boids
        self.positions = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))        
        self.velocities = np.random.uniform(low=[0,0], high=[width, height], size=(num_boids, 2))

        # Initialize the environment parameter
        self.environment = environment
        self.width = width
        self.height = height

        # Initialize the boid attributes
        coefficient_of_variation = attributes["coefficient_of_variation"]
        scale = attributes["scale"]

        # For all attributes draw a random sample for each boid
        for key, value in attributes.items():
            if key not in ["coefficient_of_variation", "scale"]:
                if value == 0:
                    # If the mean value is zero use a negative normal distribution to draw a sample
                    standard_deviation = scale
                    attribute = scale - np.random.normal(value, standard_deviation, num_boids)
                else:
                    # If the mean is not zero compute the standard deviation using a predefined coefficient of variation to draw samples from a normal distribution
                    standard_deviation = coefficient_of_variation * value
                    attribute = np.random.normal(value, standard_deviation, num_boids)
                    
                setattr(self, key, attribute)
                
        self.positions_over_time = [self.positions]
        self.velocities_over_time = [self.velocities]

    # Compute the force to avoid borders
    def avoid_border_rule(self, positions):
        avoid_force = np.zeros((len(positions), 2))

        # Calculate distance from each boid to the border
        distance_to_border = np.minimum(np.minimum(positions[:, 0], self.width - positions[:, 0]),
                                        np.minimum(positions[:, 1], self.height - positions[:, 1]))

        # Calculate steering force to avoid the border
        too_close = distance_to_border < self.avoid_border_distance
        avoid_force[too_close, 0] = np.sign(self.width / 2 - positions[too_close, 0])
        avoid_force[too_close, 1] = np.sign(self.height / 2 - positions[too_close, 1])

        return avoid_force

    # Get the distances between boids
    def get_distances(self, positions_2):
        return np.sqrt(np.sum((self.positions[:, np.newaxis] - positions_2) ** 2, axis=2))

    # Get the boids within a distance specified by the rule at hand
    def get_close_boids(self, rule_distance, distances):
        return (distances < rule_distance) & (distances > 0)

    # Compute the alignment force
    def alignment_rule(self, distances):
        # Get boids within the alignment distance
        close_boids = self.get_close_boids(self.alignment_distance, distances)

        # Calculate steering force for boid alignment
        alignment_force = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)): 
            neighbors = close_boids[i]
            if any(neighbors):

                # For each boid compute the vector pointing in the mean steering direction of close boids
                alignment_force[i] = np.mean(self.velocities[neighbors], axis=0)
        return alignment_force 

    # Compute the cohesion force
    def cohesion_rule(self, distances):
        # Get boids within the cohesion distance
        close_boids = self.get_close_boids(self.cohesion_distance, distances)

        cohesion_force = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):

                # For each boid compute the vector pointing to the mean position of close boids
                cohesion_force[i] = np.mean(self.positions[neighbors], axis=0) - self.positions[i]
        return cohesion_force 

    # Compute the separation force
    def separation_rule(self, distances):
        # Get boids within the separation distance
        close_boids = self.get_close_boids(self.separation_distance, distances)

        separation_force = np.zeros((len(self.positions), 2))
        for i in range(len(self.positions)):
            neighbors = close_boids[i]
            if any(neighbors):

                # For each boid compute the vector pointing in the opposite direction of the nearest boid
                separation_force[i] = - np.sum(self.positions[neighbors] - self.positions[i], axis=0)

        return separation_force 
    
    # Limit the velocity of boids
    def limit_velocity(self):
        # Compute the current velocity of each boid
        speed = np.sqrt(np.sum(self.velocities ** 2, axis=1))

        # If boids are moving faster than the specified maximum velocity limit their speed
        too_fast = speed > self.max_velocity
        self.velocities[too_fast] = self.velocities[too_fast] / speed[too_fast, np.newaxis] * self.max_velocity[too_fast, np.newaxis]

    # Wrap the environment border such that if a boid leaves the environment it reappears on the other side
    def wrapped_borders(self, positions):
        positions[:, 0] = np.where(positions[:, 0] > self.width, positions[:, 0] - self.width, positions[:, 0])
        positions[:, 0] = np.where(positions[:, 0] < 0, positions[:, 0] + self.width, positions[:, 0])
        positions[:, 1] = np.where(positions[:, 1] > self.height, positions[:, 1] - self.height, positions[:, 1])
        positions[:, 1] = np.where(positions[:, 1] < 0, positions[:, 1] + self.height, positions[:, 1])
        return positions
    
    # Compute whether a boid hit a border/wall
    def hard_borders(self, positions):
        positions[:, 0] = np.where(positions[:, 0] > self.width, self.width, positions[:, 0])
        positions[:, 0] = np.where(positions[:, 0] < 0, 0, positions[:, 0])
        positions[:, 1] = np.where(positions[:, 1] > self.height, self.height, positions[:, 1])
        positions[:, 1] = np.where(positions[:, 1] < 0, 0, positions[:, 1])
        return positions
    
    # Update the position of boids depending on the environment setup
    def update_positions(self, positions):
        if self.environment == 'wrapped_borders':
            return self.wrapped_borders(positions)
        elif self.environment == 'hard_borders':
            return self.hard_borders(positions)

    # Perform a simulation step
    def run_simulation(self, num_steps):
        for i in range(num_steps):
            self.step()
        return self.positions_over_time, self.velocities_over_time

    # Define the crossover function for boids
    def crossover(self, parents, method):
        trait_dic = {}

        # Use the specified crossover method
        if method == "choice":
            # Randomly choose traits traits of the parents to produce new childs
            for trait_name in self.trait_names:
                trait_dic[trait_name] = np.random.choice(getattr(self, trait_name)[parents], 1)[0]

        # Use either the maximum or the average trait value of the parents for their children
        elif method == "max" or method == "mean":
            if method == "max":
                func = np.max
            elif method == "mean":
                func = np.mean

            # Set the new attributes for the next generation of boids
            for key, _ in self.attributes.items(): 
                if key not in ["coefficient_of_variation", "scale"]:
                    trait = getattr(self, key)
                    trait_dic[key] = func(trait[parents])

        return trait_dic
    
    # Mutate genes (traits) of the new generation of boids
    def mutate(self, child, scale):
        for trait, value in child.items():
            if isinstance(value, list):
                child[trait] = list(map(lambda x: np.random.normal(x, abs(x*scale)), value))
            else:
                child[trait] = np.random.normal(value, abs(value*scale))
    
    # Helper function to set the traits for boids
    def set_traits(self, trait_dic):
        for key, _ in self.attributes.items(): #, value
            if key not in ["coefficient_of_variation", "scale"]:
                setattr(self, key, np.array(trait_dic[key]))

    # Helper function to get the trait names
    def get_trait_names(self):
        return self.trait_names
    
