# Natural Selection in Boid Hunter-Prey Simulations

Welcome! This repository contains the source code for running a hunter-prey boid simulation, combined with a genetic algorithm. For the latest version of the simulation check out the `project` branch.

## Running the code
The code can be run by simply running the Simulation.py file. 
In this file there are several parameters given, these parameters can be changed to obtain different simulation results, but for a sample run, you don't need to change any of the parameters.
The simulation has a few dependencies, such as the `pygame` game engine. All these libraries can be installed using `pip`.

A few important parameters are:

`render_sim_verbosity`: the different render options for the simulations, changing this value changes whether the Evolution of Traits plot or the PyGame simulation itself will be rendered
`result_dir`: The name of the directory where the results will be written to, by default this is `Results/`
`record_generations`: Whether the last and first generation of a trial are recorded and save to the `result_dir`



## Screenshot of the simulation
![](BoidBaseSimulation.png?raw=true)


## Results
When running the code, resulting graphs and csv files of the trait evolution will be written to the folder `Results/ `
