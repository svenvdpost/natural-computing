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
        self.font = pygame.font.SysFont('arial', 15)

        self.genetic = Genetic.Genetic(self, 0.01)

    # ---- PREY ------
    def init_prey(self):
        # Define model parameters
        num_prey = self.num_prey
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
        boids = Prey.Prey(num_prey, scale, width, height, alignment_distance, cohesion_distance, separation_distance, dodging_distance,
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
        elimination_distance = 10
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

        pygame.font.init()     

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

            generation = 0
 
            while not exit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True

                # reset the canvas
                self.canvas.fill((255,255,255))

                prey_positions, prey_velocities = self.prey.step_pygame(predators_positions, predators_velocities)                            
                

                predators_positions, predators_velocities, eliminated_prey, predator_kills = self.predators.step_pygame(prey_positions, prey_velocities)   # prey_positions, prey_velocities   

                # remove/deactivate eliminated prey
                elimination_order.extend(eliminated_prey)
                prey_positions[eliminated_prey] = None

                # Stor how long the eliminated prey boids survived within the simulation
                for _ in range(len(eliminated_prey)):
                    prey_survival_times.append(time_step)

                # Update kill count
                predator_kill_counts += predator_kills
                
                # Draw prey and predators
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


                # Stop simulation if all prey was hunted down or max steps reached
                if len(elimination_order) >= self.num_prey or time_step >= self.max_time_steps:
                    prey_crossover_idx, predator_crossover_idx = self.genetic.crossover_selection(self, elimination_order, predator_kill_counts, prey_survival_times, time_step)
                    
                    print(f'predator_crossover_idx: {predator_crossover_idx}')
                    print(f'prey_crossover_idx: {prey_crossover_idx}')

                    self.predators = self.genetic.next_generation(predator_crossover_idx, self.predators)
                    self.prey = self.genetic.next_generation(prey_crossover_idx, self.prey)
                    self.num_predator = self.predators.num_boids
                    self.num_prey = self.prey.num_boids

                    #exit = True

                    time_step = 0
                    predator_kill_counts = np.zeros(self.predators.num_boids)
                    elimination_order = []
                    prey_survival_times = []

                    # print generation info
                    print(f"generation: {generation}")
                    print(f"num_prey: {self.num_prey}")
                    print(f"num_predator: {self.num_predator}")

                    # if one of the classes only has a population of 1 or less, stop the simulation
                    if self.num_prey <= 1 or self.num_predator <= 1:
                        exit = True
                        print(f"extinction! num_prey={self.num_prey}, num_predator={self.num_predator}")

                    # update the generation timer
                    generation += 1

                time_step += 1
                 
                time.sleep(0.01)


if __name__ == "__main__":

    # Define the simulation parameters
    num_prey = 50
    num_predator = 4
    scale = 0.1
    width = 700
    height = 500
    #num_steps = 100  
    num_prey_crossover = 10
    num_predator_crossover = 4
    max_time_steps = 100
    survival_time_scaling_factor = 2 #... better name, the higher the more weight on survival times 
    kill_counts_scaling_factor = 2 # ... better name, the higher the more weight on survival times  

    simulation = Simulation(num_prey, num_predator, scale, width, height, num_prey_crossover, num_predator_crossover, max_time_steps, survival_time_scaling_factor, kill_counts_scaling_factor)

    #simulation.render_and_run(num_steps)   
    simulation.run_forever()