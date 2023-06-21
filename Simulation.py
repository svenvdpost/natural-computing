import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.ticker as mtick
import time
import random

import Prey
import Predator
import Genetic

class Simulation:



    def __init__(self, num_prey, num_predator, coefficient_of_variation, width, height, environment, num_prey_crossover, num_predator_crossover, max_time_steps, max_generations, survival_time_scaling_factor, kill_counts_scaling_factor, render_sim_verbosity) -> None:

        self.num_prey = num_prey
        self.num_predator = num_predator
        self.coefficient_of_variation = coefficient_of_variation
        self.width = width
        self.height = height
        self.environment = environment

        self.num_prey_crossover = num_prey_crossover
        self.num_predator_crossover = num_predator_crossover
        self.max_time_steps = max_time_steps
        self.max_generations = max_generations
        self.survival_time_scaling_factor = survival_time_scaling_factor
        self.kill_counts_scaling_factor = kill_counts_scaling_factor

        self.render_sim_verbosity = render_sim_verbosity

        self.prey = self.init_prey()        
        self.predators = self.init_predators()
        self.canvas = self.init_pygame()
        self.font = pygame.font.SysFont('arial', 15)

        self.genetic = None
        self.fig, self.axs = plt.subplots(3, figsize=(8, 12), gridspec_kw={'height_ratios': [1, 3, 3]})
        self.traits = []


    # ---- PREY ------
    def init_prey(self):
        # Define model parameters
        num_prey = self.num_prey
        coefficient_of_variation = self.coefficient_of_variation
        avoid_border_distance = 50
        avoid_border_strength = 0.4
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25 #25
        dodging_distance = 100
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        dodging_strength = 0.1
        noise_strength = 0.1
        max_velocity = 5    

        # Create Boids object
        boids = Prey.Prey(num_prey, coefficient_of_variation, width, height, environment, avoid_border_distance, avoid_border_strength, alignment_distance, cohesion_distance, separation_distance, dodging_distance,
                            alignment_strength, cohesion_strength, separation_strength, dodging_strength, noise_strength,  max_velocity) # vision_distance,
        
        return boids

    def draw_prey(self, positions, velocities):
        shape = pygame.Surface([20,20])

        for pos, vel in zip(positions, velocities):
            pygame.draw.circle(self.canvas, (255,0,0), pos, 3)
            pygame.draw.circle(self.canvas, (0,255,0), pos + vel, 3)

    # ---- PREDATORS ------
    def init_predators(self):
        # Define model parameters
        num_predator = self.num_predator
        #num_prey = self.num_prey
        coefficient_of_variation = self.coefficient_of_variation
        avoid_border_distance = 50
        avoid_border_strength = 0.4
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25
        hunting_distance = 100
        elimination_distance = 10
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        hunting_strength = 0.5 
        noise_strength = 0.1
        max_velocity = 6   

        # Create Predator object
        boids = Predator.Predators(num_predator, coefficient_of_variation, width, height, environment, avoid_border_distance, avoid_border_strength,alignment_distance, cohesion_distance, separation_distance, hunting_distance, elimination_distance,
                                   alignment_strength, cohesion_strength, separation_strength, hunting_strength, noise_strength, max_velocity) #vision_distance, dodging_strength,
        
        return boids

    def draw_predators(self, positions, velocities):
        for pos, vel in zip(positions, velocities):

            pygame.draw.circle(self.canvas, (0,0,255), pos, 5)
            pygame.draw.circle(self.canvas, (0,255,0), pos + vel, 5)
    

    # ---- CANVAS -----
    def init_pygame(self):
        pygame.init()

        if self.render_sim_verbosity > 1:
            canvas = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Boid simulation")

            pygame.font.init()     

            return canvas


    # ---- GENETIC ----
    def init_genetic(self, mutation_rate_prey, mutation_rate_predator, mutation_scale):
        self.genetic = Genetic.Genetic(self, mutation_rate_prey, mutation_rate_predator, mutation_scale)

    # ---- RUNING THE SIMULATION -----
    def render_and_run(self, steps):

        prey_positions, prey_velocities = self.prey.run_simulation(steps)
        pred_positions, pred_velocities = self.predators.run_simulation(steps)
        exit = False

        while not exit:

            for prey_pos, prey_velo, pred_pos, pred_velo in zip(prey_positions, prey_velocities, pred_positions, pred_velocities):

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True
                        
                if exit == True:
                    break

                self.canvas.fill((255,255,255))

                simulation.draw_prey(prey_pos, prey_velo)
                simulation.draw_predators(pred_pos, pred_velo)

                pygame.display.update()

                time.sleep(0.05)
  
    
    def create_trait_dict(self,boid_class):
        traits = dict()
        for trait in boid_class.trait_names:
            traits[trait] = [np.mean(getattr(boid_class, trait))]
        return traits
    
    def update_trait_dict(self, boid_class, crossover_idx, traits):
        mean_traits = boid_class.crossover(crossover_idx, "mean")
        #print(f'mean_traits: {mean_traits}')
        for trait in boid_class.trait_names:
            traits[trait].append(mean_traits[trait])

        return traits

    def plot_evolution_of_traits(self, survival_times, prey_traits, predator_traits):

        for i, ax in enumerate(self.axs):
            if i == 0:
                # Plot the fitness proxy
                ax.plot(survival_times, c = 'k')
                ax.axhline(y = survival_times[0], color = 'Gray', linestyle = '--', label = 'Baseline')
                ax.set_ylim([0, 1])
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['Apex Predator', 'Apex Prey'])
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness')
                ax.set_title(f'Fitness Proxy')

            # Plot the evolution of traits    
            else:
            
                if i == 1:
                    traits = prey_traits
                    name = 'Prey'
                elif i == 2:
                    traits = predator_traits
                    name = 'Predator'

                x = range(len(traits[next(iter(traits))]))
                color = iter(cm.jet(np.linspace(0, 1, len(traits))))

                # Get and plot the evolution of each trait
                ax.axhline(y = 0, color = 'Gray', linestyle = '--', label = 'Baseline') 
                for _, trait in enumerate(traits):

                    if self.environment == 'wrapped_borders':
                        if trait == ('avoid_border_distance' or 'avoid_border_strength'):
                            continue

                    c = next(color)
                    values = traits[trait]
                    normalized_trait = (np.array(values) - values[0]) / np.abs(values[0])  * 100

                    if trait not in self.traits:
                        ax.plot(x, normalized_trait,  label=trait, c=c)
                        self.traits.append(trait)
                    else:
                        # Update the plot using the new data
                        line, = ax.plot(x, normalized_trait, label=trait, c=c)
                        line.set_ydata(normalized_trait)

                ax.set_xlabel('Generation')
                ax.set_ylabel('Normalized Trait Value')
                ax.set_title(f'Evolution of {name} Traits')
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            if ax.get_legend() == None:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

            ax.relim()
            ax.autoscale_view()
            

        plt.tight_layout()
        plt.pause(0.01)
    
    
    def run_forever(self):

            exit = False

            prey_positions = self.prey.positions
            prey_velocities = self.prey.velocities

            predators_positions = self.predators.positions
            predators_velocities = self.predators.velocities

            elimination_order = []
            prey_survival_times = []
            normalized_mean_prey_survival_times = []
            time_step = 1
            predator_kill_counts = np.zeros(self.num_predator)

            generation = 0

            # Get and store the average boid traits
            mean_prey_traits = self.create_trait_dict(self.prey)
            mean_predator_traits = self.create_trait_dict(self.predators)

 
            while not exit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True
                # let the prey do simulation step
                prey_positions, prey_velocities = self.prey.step_pygame(predators_positions, predators_velocities) 

                # let the predators do a simulatino step
                predators_positions, predators_velocities, eliminated_prey, predator_kills = self.predators.step_pygame(prey_positions, prey_velocities)   # prey_positions, prey_velocities   

                # remove/deactivate eliminated prey
                elimination_order.extend(eliminated_prey)
                prey_positions[eliminated_prey] = None

                # Store how long the eliminated prey boids survived within the simulation
                for _ in range(len(eliminated_prey)):
                    prey_survival_times.append(time_step)

                # Update kill count
                predator_kill_counts += predator_kills
                
                # Draw prey and predators
                if (self.render_sim_verbosity == 4  
                    or (self.render_sim_verbosity > 1 and (generation == self.max_generations-1 or self.num_prey <= 1 or self.num_predator <= 1)) 
                    or (self.render_sim_verbosity == 3 and (generation == 0 or (generation == self.max_generations-1 or self.num_prey <= 1 or self.num_predator <= 1)))): #or generation == 0
                     
                    # reset the canvas
                    self.canvas.fill((255,255,255))
                    simulation.draw_prey(prey_positions, prey_velocities)                     
                    simulation.draw_predators(predators_positions, predators_velocities)

                    # Display some stats
                    text_surface = self.font.render(f' generation: {generation}', False, (0, 0, 0)) 
                    self.canvas.blit(text_surface, (0,0))
                    text_surface = self.font.render(f' num_prey: {self.num_prey - len(elimination_order)}', False, (0, 0, 0)) 
                    self.canvas.blit(text_surface, (0,20))
                    text_surface = self.font.render(f' kill_counts: {predator_kill_counts}', False, (0, 0, 0)) 
                    self.canvas.blit(text_surface, (0,40))
                    text_surface = self.font.render(f' time_step: {time_step}', False, (0, 0, 0)) 
                    self.canvas.blit(text_surface, (0,60))

                    pygame.display.update()
                    time.sleep(0.01)


                # Stop simulation if all prey was hunted down or max steps reached
                if len(elimination_order) >= self.num_prey or time_step >= self.max_time_steps:

                    # Select the fittest parents
                    prey_crossover_idx, predator_crossover_idx = self.genetic.crossover_selection(elimination_order, predator_kill_counts, prey_survival_times, time_step)

                    # Get the average traits for both classes and plot their evolution over generations
                    mean_prey_traits = self.update_trait_dict(self.prey, prey_crossover_idx, mean_prey_traits)
                    mean_predator_traits = self.update_trait_dict(self.predators, predator_crossover_idx, mean_predator_traits)

                    # Compute the normalized average survival times of the prey boids as a fitness proxy
                    normalized_mean_prey_survival_times.append(((np.sum(prey_survival_times) + ((self.num_prey - len(prey_survival_times))  * self.max_time_steps)) / self.num_prey)/ self.max_time_steps  )                      
                    if self.render_sim_verbosity > 0:
                        self.plot_evolution_of_traits(normalized_mean_prey_survival_times, mean_prey_traits, mean_predator_traits)

                    # if one of the classes only has a population of 1 or less, stop the simulation
                    if self.num_prey <= 1 or self.num_predator <= 1:
                        self.plot_evolution_of_traits(normalized_mean_prey_survival_times, mean_prey_traits, mean_predator_traits)
                        print(f"extinction! num_prey={self.num_prey}, num_predator={self.num_predator}")
                        input("Press Enter to quit...")
                        exit = True
                        break
                    elif generation >= self.max_generations:
                        self.plot_evolution_of_traits(normalized_mean_prey_survival_times, mean_prey_traits, mean_predator_traits)
                        print("max generations reached")
                        input("Press Enter to quit...")
                        exit = True
                        break
                    else: 
                        if len(elimination_order) >= self.num_prey:
                            event = 'genocide'
                        elif time_step >= self.max_time_steps:
                            event = 'maximum simulation time reached'

                        # create next generation of boids according
                        self.predators = self.genetic.next_generation(predator_crossover_idx, self.predators, procreation="fixed", crossover_method="mean") #fixed
                        self.prey = self.genetic.next_generation(prey_crossover_idx, self.prey, procreation="fixed", crossover_method="mean") #fixed
                        
                        self.num_predator = self.predators.num_boids
                        self.num_prey = self.prey.num_boids

                        # reset simulation parameters for next generation
                        time_step = 0
                        predator_kill_counts = np.zeros(self.predators.num_boids)
                        elimination_order = []
                        prey_survival_times = []

                    # print generation info
                    print(f"Generation: {generation} / {self.max_generations}   Event: {event}   Number prey: {self.num_prey}   Number predators: {self.num_predator}")

                    # update the generation timer
                    generation += 1

                time_step += 1


if __name__ == "__main__":

    # Define the simulation parameters
    num_prey = 50
    num_predator = 4
    coefficient_of_variation = 0.4
    width = 1200
    height = 1000 
    environment = 'hard_borders' #hard_borders / wrapped_borders
    num_prey_crossover = 10
    num_predator_crossover = 4
    max_time_steps = 1000
    max_generations = 50
    survival_time_scaling_factor = 2 #... better name, the higher the more weight on survival times 
    kill_counts_scaling_factor = 2 # ... better name, the higher the more weight on survival times  
    render_sim_verbosity = 3 # 0: do not render any simulation; 1: Only render evolution of traits (EoT); 2: render EoT and final generation simulation; 3: render EoT, initial and final generation simulation; 4: render EoT and each simulation

    mutation_rate_prey = 0.1
    mutation_rate_predator = 0.5
    mutation_scale = 0.2



    #simulation.render_and_run(num_steps)   
    simulation = Simulation(num_prey, num_predator, coefficient_of_variation, width, height, environment, num_prey_crossover, num_predator_crossover, max_time_steps, max_generations, survival_time_scaling_factor, kill_counts_scaling_factor, render_sim_verbosity)

    simulation.init_genetic(mutation_rate_prey, mutation_rate_predator, mutation_scale)

    simulation.run_forever()