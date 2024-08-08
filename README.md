#**DRL-Based Highway On-Ramp Merging with Pedestrian and Driver Intention Models**
Overview
This project presents a novel decision-making strategy for autonomous vehicles during highway on-ramp merging scenarios. The strategy is powered by a Deep Reinforcement Learning (DRL) model, specifically the Deep Deterministic Policy Gradient (DDPG) algorithm, which is enhanced by predicting the intentions of both pedestrians and vehicles in the merging area.

Key Features
Pedestrian Intention Model: Predicts the actions of pedestrian clusters near the merging zone, such as crossing, waiting, or walking alongside the road.
Driver Intention Model: Predicts the behavior of main-lane vehicles, including whether they will yield, accelerate, or change lanes.
DRL DDPG Model: Trains the ego vehicle to execute optimal merging strategies by considering the intentions of nearby pedestrians and vehicles.
Safety Control Mechanism: Ensures safe maneuvering by overriding any unsafe decisions made by the DRL model.
Project Components
Pedestrian Intention Model: Utilizes features like position, velocity, and clustering to predict pedestrian behavior.
Driver Intention Model: Analyzes vehicle dynamics and pedestrian intentions to predict driver actions.
DDPG-based DRL Model: Takes in the predicted intentions and the ego vehicle's state to decide the best course of action.
Safety Control Module: Acts as a safeguard, ensuring all decisions adhere to safety standards.
How It Works
The ego vehicle enters the control zone and collects data on nearby vehicles and pedestrians.
Pedestrian and driver intentions are predicted using respective models.
The DRL model processes these predictions and the ego vehicle's current state to determine an optimal merging action.
A safety module checks and potentially adjusts the action to avoid any dangerous situations.
The ego vehicle executes the action, and the DRL model learns from the outcome to improve future performance.
Use Cases
Autonomous highway merging with pedestrian influence.
Enhanced decision-making in complex traffic scenarios.
Research in DRL applications for autonomous driving.
Getting Started
To get started, clone this repository and follow the instructions to set up the SUMO simulation environment, train the DRL model, and run simulations of the merging strategy.
