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

        self.attributes = attributes
        coefficient_of_variation = attributes["coefficient_of_variation"]
        self.num_boids =num_boids
        scale = attributes["scale"]

        predator_specific_attributes = ["hunting_strength", "hunting_distance", "elimination_distance"]

        for key, value in attributes.items():
            if key  in predator_specific_attributes:
                if value == 0:
                    standard_deviation = scale
                else:
                    standard_deviation = coefficient_of_variation * value
                    
                attribute = np.random.normal(value, standard_deviation, num_boids)
                setattr(self, key, attribute)
        #self.trait_names = super().get_trait_names() + ['hunting_distance', 'hunting_strength', 'elimination_distance']

    
    def step_pygame(self, prey_positions, prey_velocities):
        predator_distances = self.get_distances(self.positions)
        

        prey_distances = self.get_distances(prey_positions)


        alignment = self.alignment_rule(predator_distances)
        cohesion = self.cohesion_rule(predator_distances)
        separation = self.separation_rule(predator_distances)
        hunting = self.hunting_rule(prey_distances, prey_positions)

        eliminate, predator_kill_counts = self.elimination(prey_distances)

        alignment_correction  = (alignment + np.random.uniform(-1,1, (self.num_boids, 2))   * self.noise_strength[:, np.newaxis]) * self.alignment_strength[:, np.newaxis]
        cohesion_correction   = (cohesion + np.random.uniform(-1,1, (self.num_boids, 2))    * self.noise_strength[:, np.newaxis]) * self.cohesion_strength[:, np.newaxis]
        separation_correction = (separation + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.separation_strength[:, np.newaxis]
        hunting_correction = (hunting + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.hunting_strength[:, np.newaxis]

        self.velocities += alignment_correction + cohesion_correction + separation_correction + hunting_correction

        if self.environment == 'hard_borders':
            avoid_border = self.avoid_border_rule(self.positions)
            avoid_border_correction = (avoid_border + np.random.uniform(-1,1, (self.num_boids, 2)) * self.noise_strength[:, np.newaxis]) * self.avoid_border_strength[:, np.newaxis]
            self.velocities += avoid_border_correction

            
        self.limit_velocity()

        self.positions = self.update_positions(self.positions + self.velocities)

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
        predator_kill_counts = np.zeros(self.num_boids)
        caught_prey_idx = []
        for i in range(len(self.positions)):
            caught_prey = self.get_close_boids(self.elimination_distance[i], distances[i])
            predator_kill_counts[i] = np.sum(caught_prey)
            if any(caught_prey):
                caught_prey_idx.extend(caught_prey.nonzero()[0])
        return list( dict.fromkeys(caught_prey_idx)), predator_kill_counts 
    

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

            genes["hunting_distance"] = func(self.hunting_distance[parents])
            genes["hunting_strength"] = func(self.hunting_strength[parents], axis=0)
            genes["elimination_distance"] = func(self.elimination_distance[parents])

        return genes
    
    def mutate(self, child, scale):
        super().mutate(child, scale)

    def set_traits(self, trait_dic):
        super().set_traits(trait_dic)

        self.hunting_distance  = np.array(trait_dic["hunting_distance"])
        self.hunting_strength = np.array(trait_dic["hunting_strength"])
        self.elimination_distance = np.array(trait_dic["elimination_distance"])

    def show_boid(self, boid_id):
        output = super().show_boid(boid_id)

        hd = self.hunting_distance[boid_id]
        hs = self.hunting_strength[boid_id]
        ed = self.elimination_distance[boid_id]

        return output + f",hd={hd}, cd={hs}, ed={ed}"
