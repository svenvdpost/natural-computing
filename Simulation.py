import pygame
import time
import numpy as np

import Prey
import Predator
import Genetic

class Simulation:

   

    def __init__(self, num_prey, num_predator, width, height) -> None:

        self.num_prey = num_prey
        self.num_predator = num_predator
        self.width = width
        self.height = height

        self.prey = self.init_prey()        
        self.predators = self.init_predators()
        self.canvas = self.init_pygame()

        self.genetic = Genetic.Genetic(self, 0.01)

    # ---- PREY ------
    def init_prey(self):
        # Define model parameters
        num_prey = self.num_prey
        num_predator = self.num_predator
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25 #25
        dodging_distance = 100
        #vision_distance = None
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        dodging_strength = 0.1
        noise_strength = 0.1
        max_velocity = 5    

        # Create Boids object
        boids = Prey.Prey(num_prey, num_predator, width, height, alignment_distance, cohesion_distance, separation_distance, dodging_distance,
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
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25
        hunting_distance = 100
        #vision_distance = None
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        hunting_strength = 0.1
        noise_strength = 0.1
        max_velocity = 5    

        # Create Predator object
        boids = Predator.Predators(num_predator, num_prey, width, height, alignment_distance, cohesion_distance, separation_distance, hunting_distance,
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

            while not exit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True

                # reset the canvas
                self.canvas.fill((255,255,255))

                prey_positions, prey_velocities = self.prey.step_pygame(predators_positions, predators_velocities)                            
                simulation.draw_prey(prey_positions, prey_velocities)

                predators_positions, predators_velocities = self.predators.step_pygame(prey_positions, prey_velocities)   # prey_positions, prey_velocities                         
                simulation.draw_predators(predators_positions, predators_velocities)             

                pygame.display.update()

                self.genetic.next_generation([10], self.predators)


                time.sleep(0.05)

# ---- GENETIC ALGORITHMS -----
    



if __name__ == "__main__":

    # Define the simulation parameters
    num_prey = 100
    num_predator = 10
    width = 700
    height = 500
    num_steps = 100    

    simulation = Simulation(num_prey, num_predator, width, height)

    #simulation.render_and_run(num_steps)   
    simulation.run_forever()