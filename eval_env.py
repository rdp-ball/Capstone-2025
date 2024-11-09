import os
import sys
import csv
import traci
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import gymnasium

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class SumoEnv(Env):
    def __init__(self, gui=False):
        self.step_length = 0.1
        self.num_observations = 7
        self.rl_id = "rl_car"
        self.crash = False
        self.merged = False
        self.timeout = False
        self.triggered = False

        # Define continuous action space parameters
        self.max_accel = 3.0  # Maximum acceleration/deceleration
        self.max_lane_change = 3.2  # Maximum lateral movement

        # Continuous action space: [acceleration, lane_change]
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-self.max_accel, -self.max_lane_change]),
            high=np.array([self.max_accel, self.max_lane_change]),
            shape=(2,),
            dtype=np.float32
        )

        # Observation space remains the same
        self.observation_space = Box(
            low=-1,
            high=3,
            shape=(self.num_observations,),
            dtype=np.float32
        )

        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create Evaluations directory
        eval_dir = os.path.join(current_dir, "Evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Use absolute path for velocity.csv
        velocity_path = os.path.join(eval_dir, "velocity.csv")
        f = open(velocity_path, "w", newline="")
        self.velocity_writer = csv.writer(f)

        # sumo config with absolute path
        if gui:
            self.sumoBinary = "sumo-gui"
        else:
            self.sumoBinary = "sumo"
        
        config_path = os.path.join(current_dir, "training_sim.sumocfg")
        self.sumoCmd = [
            self.sumoBinary,
            "-c", config_path,
            "--step-length", str(self.step_length),
            "--no-warnings", "true"
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # close sumo if already running
        if 'SUMO_HOME' in os.environ:
            try:
                traci.close()
            except:
                pass

        # start simulation
        traci.start(self.sumoCmd)
        traci.simulationStep()

        # reset variables
        self.crash = False
        self.merged = False
        self.timeout = False
        self.triggered = False

        # get initial state
        self.state = self.get_state(self.rl_id)

        return self.state, {}

    def step(self, action):
        info = {}
        done = False

        # apply the actions and move a step forward
        self.apply_action(action, self.rl_id)
        traci.simulationStep()
        self.velocity_writer.writerow([self.rl_id, traci.vehicle.getSpeed(self.rl_id)])
        
        if ":merge" in traci.vehicle.getRoadID(self.rl_id):
            traci.vehicle.changeSublane(self.rl_id, -traci.vehicle.getLateralLanePosition(self.rl_id))

        # check for collision, merge or timeout here
        if traci.simulation.getCollisions():
            self.crash = True
            info = {"is_success": False}

        if traci.simulation.getTime() > 150:
            self.timeout = True
            info = {"is_success": True}

        if not self.crash and not self.timeout:
            if traci.vehicle.getLaneID(self.rl_id) in ["merging_1", "outgoing_0", "outgoing_1"]:
                self.merged = True
                traci.vehicle.setColor(self.rl_id, (255,255,255))     
                info = {"is_success": True}
        
        # if some conditions are met set done to True
        if self.crash or self.merged or self.timeout:
            done = True
        
        # gets the new state
        self.state = self.get_state(self.rl_id, crash=self.crash)

        # calculates the new reward
        reward = self.get_reward(self.rl_id, crash=self.crash, merged=self.merged)

        terminated = done
        truncated = False

        return self.state, reward, terminated, truncated, info

    def apply_action(self, action, rl_id):
        """
        Apply continuous actions:
        action[0]: acceleration/deceleration (-3 to 3)
        action[1]: lane change (-3.2 to 3.2)
        """
        # Apply acceleration/deceleration
        traci.vehicle.setAcceleration(rl_id, float(action[0]), self.step_length)

        # Apply lane change if on merging section and in position
        if traci.vehicle.getRoadID(rl_id) == "merging" and traci.vehicle.getPosition(rl_id)[0] < 240 and self.triggered == False:
            lateral_movement = float(action[1])
            traci.vehicle.changeSublane(rl_id, lateral_movement)
            if abs(lateral_movement) > 0.1:  # If significant lane change attempted
                self.triggered = True

    def get_state(self, rl_id, crash=False):
        # if crash return zeros
        if crash:
            return np.zeros(self.num_observations)
        
        # get rl vehicle data
        rl_speed = traci.vehicle.getSpeed(rl_id)
        rl_pos = traci.vehicle.getPosition(rl_id)
        rl_lane = traci.vehicle.getLaneID(rl_id)

        # get lead vehicle data
        lead_obs = [-1, -1]
        lead_veh = traci.vehicle.getLeader(rl_id)
        if lead_veh is not None:
            lead_obs = [lead_veh[1], traci.vehicle.getSpeed(lead_veh[0])]

        # get merge vehicle data
        merge_obs = [-1, -1, -1]
        if rl_lane == "merging_0":
            veh_list = traci.vehicle.getIDList()
            for veh in veh_list:
                if veh != rl_id:
                    veh_lane = traci.vehicle.getLaneID(veh)
                    if veh_lane == "incoming_0":
                        veh_pos = traci.vehicle.getPosition(veh)
                        merge_obs = [veh_pos[0]-rl_pos[0], veh_pos[1]-rl_pos[1], traci.vehicle.getSpeed(veh)]

        # combine observations
        obs = np.array([rl_speed] + lead_obs + merge_obs)
        return obs

    def get_reward(self, rl_id, crash=False, merged=False):
        # return large negative reward if crash
        if crash:
            return -100
        
        # return large positive reward if merge complete
        if merged:
            return 100

        # otherwise return speed based reward
        speed = traci.vehicle.getSpeed(rl_id)
        reward = speed/30
        return reward

    def close(self):
        traci.close()