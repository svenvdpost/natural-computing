import pygame
import time
import numpy as np

import Prey
import Predator


class Simulation:


    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

        self.boids = self.init_boids()
        #self.predators = self.init_predators()
        self.canvas = self.init_pygame()

    def init_boids(self):
        # Define model parameters
        num_boids = 100
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25 #25
        vision_distance = None
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        noise_strength = 0.1
        max_velocity = 5    

        # Create Boids object
        boids = Prey.Boids(num_boids, width, height, alignment_distance, cohesion_distance, separation_distance,
                           alignment_strength, cohesion_strength, separation_strength, max_velocity)
        
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

    def init_predators(self):
        # Define model parameters
        num_pred = 10
        alignment_distance = 50
        cohesion_distance = 100
        separation_distance = 25
        vision_distance = None
        alignment_strength = 0.1
        cohesion_strength = 0.001
        separation_strength = 0.05
        noise_strength = 0.1
        max_velocity = 5    

        # Create Boids object
        boids = Predator.Predator(num_pred, width, height, alignment_distance, cohesion_distance, separation_distance,
                                  alignment_strength, cohesion_strength, separation_strength, max_velocity)
        
        return boids

    def init_pygame(self):
        pygame.init()
        canvas = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Boid simulation")

        return canvas


    def render_and_run(self, steps):

        positions, velocities = self.boids.run_simulation(steps)
        exit = False

        while not exit:

            for t in positions:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True
                        
                if exit == True:
                    break

                self.canvas.fill((255,255,255))
                for pos in t:
                    pygame.draw.circle(self.canvas, (255,0,0), pos, 3)

                pygame.display.update()

                time.sleep(0.05)
  
    def run_forever(self):

            exit = False

            while not exit:
                positions, velocities = self.boids.step_pygame()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True
                            
                self.canvas.fill((255,255,255))

                simulation.draw_prey(positions, velocities)

                pygame.display.update()

                time.sleep(0.05)


if __name__ == "__main__":

    # Define the simulation parameters
    width = 700
    height = 500
    num_steps = 100    

    simulation = Simulation(width, height)

    #simulation.render_and_run(num_steps)   
    simulation.run_forever()