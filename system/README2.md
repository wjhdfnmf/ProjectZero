
# System

## Agent: Neural Network

## Domain: Inverted pendulum

## Trainer: Evolutionary Algorithm

---

### Status:
* all work is focused on 'system.hpp'
* migrate 'network_trainer.hpp' to 'system.hpp' as general trainer 'mode'
* domain and agent have worked on their own

### File notes:
* main.cpp : main system run file
* system.hpp : agent and domain system; manages interation
* network_trainer.hpp : trainer - population of agents and Evolutionary Algorithm
* neural_network.hpp : agent - neural network
* pendulum.hpp : domain - inverted pendulum written by Sierra Gonzales
* utilities.hpp : common basic functions
* Makefile : command: make - compiles code (run and test) using g++

### Folder notes:
* _gui: Qt code (WIP)
* _output: output data (runtime, compiler)
* _testing: test functions and WIP files