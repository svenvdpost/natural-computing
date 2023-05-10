import pygame
import time

import Prey
import Predator


class Simulation:


    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

        self.boids = self.init_boids()
        self.predators = self.init_predators()
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
        boids = Prey.Boids(num_boids = num_boids, width = width, height = height, \
            alignment_distance = alignment_distance, cohesion_distance = cohesion_distance, \
            separation_distance = separation_distance, alignment_strength = alignment_strength, \
            cohesion_strength = cohesion_strength, separation_strength = separation_strength, \
            max_velocity = max_velocity)
        
        return boids

    def init_predators(self):
        # Define model parameters
        num_predators = 10
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
        boids = Predator.Predators(num_boids = num_predators, width = width, height = height, \
            alignment_distance = alignment_distance, cohesion_distance = cohesion_distance, \
            separation_distance = separation_distance, alignment_strength = alignment_strength, \
            cohesion_strength = cohesion_strength, separation_strength = separation_strength, \
            max_velocity = max_velocity)
        
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
                positions = self.boids.step_no_save()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit = True
                            
                self.canvas.fill((255,255,255))
                for pos in positions:
                    pygame.draw.circle(self.canvas, (255,0,0), pos, 3)

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