import pygame
import time
import numpy as np
import random

import Prey
import Predator
import Genetic

class Simulation:



    def __init__(self, num_prey, num_predator, scale, width, height, num_prey_crossover, num_predator_crossover, max_time_steps, survival_time_scaling_factor, kill_counts_scaling_factor) -> None:

        self.num_prey = num_prey
        self.num_predator = num_predator
        self.scale = scale
        self.width = width
        self.height = height

        self.num_prey_crossover = num_prey_crossover
        self.num_predator_crossover = num_predator_crossover
        self.max_time_steps = max_time_steps
        self.survival_time_scaling_factor = survival_time_scaling_factor
        self.kill_counts_scaling_factor = kill_counts_scaling_factor

        self.prey = self.init_prey()        
        self.predators = self.init_predators()
        self.canvas = self.init_pygame()

        self.genetic = Genetic.Genetic(self, 0.01)

    # ---- PREY ------
    def init_prey(self):
        # Define model parameters
        num_prey = self.num_prey
        num_predator = self.num_predator
        scale = self.scale
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
        boids = Prey.Prey(num_prey, num_predator, scale, width, height, alignment_distance, cohesion_distance, separation_distance, dodging_distance,
                            alignment_strength, cohesion_strength, separation_strength, dodging_strength, noise_strength,  max_velocity) # vision_distance,
        
        return boids

    def draw_prey(self, positions, velocities):
        shape = pygame.Surface([20,20])

        for pos, vel in zip(positions, velocities):
            #angle = np.angle(vel, deg=True)
            #x, y = pos
            #shape = pygame.Rect(x, y, 5, 10)
            #angled_shape = pygame.transform.rotate(shape, angle)
            #rectangle = angled_shape.get_rect()

            pygame.draw.circle(self.canvas, (255,0,0), pos, 3)
            pygame.draw.circle(self.canvas, (0,255,0), pos + vel, 3)
            #pygame.draw.rect(self.canvas, (0,0,255), rectangle)

    # ---- PREDATORS ------
    def init_predators(self):
        # Define model parameters
        num_predator = self.num_predator
        num_prey = self.num_prey
        scale = self.scale
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25
        hunting_distance = 100
        elimination_distance = 10 #10?
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        hunting_strength = 0.1
        noise_strength = 0.1
        max_velocity = 6    

        # Create Predator object
        boids = Predator.Predators(num_predator, scale, width, height, alignment_distance, cohesion_distance, separation_distance, hunting_distance, elimination_distance,
                                   alignment_strength, cohesion_strength, separation_strength, hunting_strength, noise_strength, max_velocity) #vision_distance, dodging_strength,
        
        return boids

    def draw_predators(self, positions, velocities):
        for pos, vel in zip(positions, velocities):

            pygame.draw.circle(self.canvas, (0,0,255), pos, 5)
            pygame.draw.circle(self.canvas, (0,255,0), pos + vel, 5)

    # ---- CANVAS -----
    def init_pygame(self):
        pygame.init()
        canvas = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Boid simulation")

        return canvas


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
  
    def run_forever(self):

            exit = False

            prey_positions = self.prey.positions
            prey_velocities = self.prey.velocities

            predators_positions = self.predators.positions
            predators_velocities = self.predators.velocities

            elimination_order = []
            prey_survival_times = []
            time_step = 1
            predator_kill_counts = np.zeros(self.num_predator)
 
            while not exit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True

                # reset the canvas
                self.canvas.fill((255,255,255))

                prey_positions, prey_velocities = self.prey.step_pygame(predators_positions, predators_velocities)                            
                

                predators_positions, predators_velocities, eliminated_prey, predator_kills = self.predators.step_pygame(prey_positions, prey_velocities)   # prey_positions, prey_velocities   


                # 'remove/deactivate eliminated prey
                elimination_order.extend(eliminated_prey)
                prey_positions[eliminated_prey] = None

                # Update kill count
                predator_kill_counts += predator_kills
                
                # Draw prey and predators
                simulation.draw_prey(prey_positions, prey_velocities)                     
                simulation.draw_predators(predators_positions, predators_velocities)             
                pygame.display.update()


                for _ in range(len(eliminated_prey)):
                    prey_survival_times.append(time_step)


                # Stop simulation if all prey was hunted down or max steps reached
                if len(elimination_order) >= self.num_prey or time_step >= self.max_time_steps:
                    # Predator crossover selection
                    predator_selection_weights = predator_kill_counts**self.kill_counts_scaling_factor
                    predator_selection_probabilities = predator_selection_weights / np.sum( predator_selection_weights)
                    predator_crossover_idx = list(np.random.choice(range(self.num_predator),size=self.num_predator_crossover,replace=False, p=predator_selection_probabilities))
                    print('predator_crossover_idx')
                    print(predator_crossover_idx)
                    
                    # Prey crossover selection
                    prey_selection_weights = list(np.array(prey_survival_times)**self.survival_time_scaling_factor)
                    prey_selection_probabilities =  prey_selection_weights / np.sum( prey_selection_weights)


                    if len(elimination_order) >= self.num_prey:
                        prey_crossover_idx = list(np.random.choice(elimination_order,size=self.num_prey_crossover,replace=False, p=prey_selection_probabilities))
                        print('prey_crossover_idx')
                        print(prey_crossover_idx)
                        

                    elif time_step >= self.max_time_steps:
                        survivors = list(set(range(num_prey)) - set(elimination_order))
                        num_select_survivors = np.min([self.num_prey_crossover, len(survivors)])

                        # Less survivors than desired number of prey boids for cross over?
                        num_remaining = self.num_prey_crossover - num_select_survivors
                        prey_crossover_idx = list(np.random.choice(survivors,size=num_select_survivors,replace=False))
                        
                        print('survivors')
                        print(survivors)
                        print('prey_crossover_idx')
                        print(prey_crossover_idx)

                        # Select the remaining boids for crossover depending on their survival time
                        if num_remaining:
                            prey_crossover_idx.extend(np.random.choice(elimination_order,size=num_remaining,replace=False, p=prey_selection_probabilities))

                        print('updated prey_crossover_idx')
                        print(prey_crossover_idx)

                    exit = True


                #self.genetic.next_generation([10], self.predators)

                time_step += 1
                 
                time.sleep(0.01)

# ---- GENETIC ALGORITHMS -----
    



if __name__ == "__main__":

    # Define the simulation parameters
    num_prey = 50
    num_predator = 4
    scale = 0.001
    width = 700
    height = 500
    #num_steps = 100  
    num_prey_crossover = 10
    num_predator_crossover = 4
    max_time_steps = 10000
    survival_time_scaling_factor = 2 #... better name, the higher the more weight on survival times 
    kill_counts_scaling_factor = 2 # ... better name, the higher the more weight on survival times  

    simulation = Simulation(num_prey, num_predator, scale, width, height, num_prey_crossover, num_predator_crossover, max_time_steps, survival_time_scaling_factor, kill_counts_scaling_factor)

    #simulation.render_and_run(num_steps)   
    simulation.run_forever()