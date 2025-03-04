import sys, random, enum, ast, time, csv
import numpy as np
from matrx import grid_world
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

TIME_CONSTANT_WAIT = 20 # robot will be waiting for human's response for 20 seconds default
TIME_TO_WAIT_FOR_HUMAN_TO_COME = 30 # Robot will be waiting for up to 30 seconds by default
REWARD_BORDER_TIME_TO_WAIT_FOR_HUMAN_TO_COME = 15 # Up to 15 seconds you get a reward, after you get a penalty to competence
CONFIDENCE_HIGH_IMPORTANCE = 0.1
CONFIDENCE_MEDIUM_IMPORTANCE = 0.06
CONFIDENCE_LOW_IMPORTANCE = 0.03
class Phase(enum.Enum):
    INTRO = 1,
    FIND_NEXT_GOAL = 2,
    PICK_UNSEARCHED_ROOM = 3,
    PLAN_PATH_TO_ROOM = 4,
    FOLLOW_PATH_TO_ROOM = 5,
    PLAN_ROOM_SEARCH_PATH = 6,
    FOLLOW_ROOM_SEARCH_PATH = 7,
    PLAN_PATH_TO_VICTIM = 8,
    FOLLOW_PATH_TO_VICTIM = 9,
    TAKE_VICTIM = 10,
    PLAN_PATH_TO_DROPPOINT = 11,
    FOLLOW_PATH_TO_DROPPOINT = 12,
    DROP_VICTIM = 13,
    WAIT_FOR_HUMAN = 14,
    WAIT_AT_ZONE = 15,
    FIX_ORDER_GRAB = 16,
    FIX_ORDER_DROP = 17,
    REMOVE_OBSTACLE_IF_NEEDED = 18,
    ENTER_ROOM = 19

class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._tick = None
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._room_vics = []
        self._searched_rooms = []
        self._reported_rooms = []
        self._found_victims = []
        self._collected_victims = []
        self._found_victim_logs = {}
        self._send_messages = []
        self._current_door = None
        self._team_members = []
        self._carrying_together = False
        self._remove = False
        self._goal_vic = None
        self._goal_loc = None
        self._human_loc = None
        self._distance_human = None
        self._distance_drop = None
        self._agent_loc = None
        self._todo = []
        self._answered = False
        self._to_search = []
        self._carrying = False
        self._waiting = False
        self._waiting_time_start = None
        self._rescue = None
        self._recent_vic = None
        self._received_messages = []
        self._moving = False
        self._processed_messages = []
        self._new_messages = []
        self._waiting_for_human_to_come = False
        self._waiting_for_human_to_come_time_start = None
        self._competence_additions = []
        self._willingness_additions = []
        self._human_asked_for_help_obstacle = False
        self._re_searching = False
        self._if_robot_finds_victim_human_incompetent = False

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                    algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action
        return state

    def decide_on_actions(self, state):
        if self._waiting_for_human_to_come:
            print(self._phase)
            print(time.time() - self._waiting_for_human_to_come_time_start)
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._team_members:
                self._team_members.append(member)
        # Create a list of received messages from the human team member
        for mssg in self.received_messages:
            for member in self._team_members:
                if mssg.from_id == member:
                    self._received_messages.append(mssg.content)

        #if len(self._new_messages) > 0:
            # print("new messages: ", self._new_messages)


        # Initialize and update trust beliefs for team members
        # trustBeliefs = self._loadBelief(self._team_members, self._folder)

        # EVALUATE TRUST BELIEFS baselines
        random_num = random.uniform(0, 2) - 1
        trustBeliefs = {'filip': {'willingness': 1, 'competence': 1, 'confidenceWill': 1, 'confidenceComp': 1}}

        # Process messages from team members
        self._process_messages(state, self._team_members, self._condition, trustBeliefs)
        self._trustBelief(self._team_members, trustBeliefs, self._folder, self._new_messages)
        self._saveBelief(self._folder, trustBeliefs)

        # Check whether human is close in distance
        if state[{'is_human_agent': True}]:
            self._distance_human = 'close'
        if not state[{'is_human_agent': True}]:
            # Define distance between human and agent based on last known area locations
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'far'
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'close'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'far'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'close'

        # Define distance to drop zone based on last known area location
        if self._agent_loc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distance_drop = 'far'
        if self._agent_loc in [3, 4, 7, 10, 13, 14]:
            self._distance_drop = 'close'

        # Check whether victims are currently being carried together by human and agent
        for info in state.values():
            if 'is_human_agent' in info and self._human_name in info['name'] and len(
                    info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id'] or \
                    'is_human_agent' in info and self._human_name in info['name'] and len(
                info['is_carrying']) > 0 and 'mild' in info['is_carrying'][0][
                'obj_id'] and self._rescue == 'together' and not self._moving:

                # If victim is being carried, add to collected victims memory
                if info['is_carrying'][0]['img_name'][8:-4] not in self._collected_victims:
                    self._collected_victims.append(info['is_carrying'][0]['img_name'][8:-4])

                self._carrying_together = True

            if 'is_human_agent' in info and self._human_name in info['name'] and len(info['is_carrying']) == 0:
                self._carrying_together = False

        # If carrying a victim together, let agent be idle (because joint actions are essentially carried out by the human)
        if self._carrying_together == True:

            if self._waiting_for_human_to_come:
                waited_in_seconds = time.time() - self._waiting_for_human_to_come_time_start
                competence_score = (- waited_in_seconds + REWARD_BORDER_TIME_TO_WAIT_FOR_HUMAN_TO_COME) / 100
                if 'critical' in self._goal_vic:
                    self._competence_additions.append((competence_score, CONFIDENCE_HIGH_IMPORTANCE*2))
                else:
                    self._competence_additions.append((competence_score, CONFIDENCE_MEDIUM_IMPORTANCE*2))

                # Continue with what robot was doing
                self._waiting_for_human_to_come = False
                self._waiting_for_human_to_come_time_start = None

                self._send_message('Thank you for helping.', 'RescueBot')

            return None, {}

        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._send_message('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')

        # Ongoing loop until the task is terminated, using different phases for defining the agent's behavior
        while True:
            if Phase.INTRO == self._phase:
                # Send introduction message
                self._send_message('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Wait untill the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._goal_vic = None
                self._goal_loc = None
                self._rescue = None
                self._moving = True
                remaining_zones = []
                # remaining victims
                remaining_vics = []
                remaining = {}
                # Identification of the location of the drop zones
                zones = self._get_drop_zones(state)
                # Identification of which victims still need to be rescued and on which location they should be dropped
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collected_victims:
                        remaining_zones.append(info)
                        remaining_vics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remaining_zones:
                    self._remainingZones = remaining_zones
                    self._remaining = remaining
                # Remain idle if there are no victims left to rescue
                if not remaining_zones:
                    return None, {}


                # Check which victims can be rescued next because human or agent already found them
                for vic in remaining_vics:
                    # Define a previously found victim as target victim because all areas have been searched
                    if vic in self._found_victims and vic in self._todo and len(self._searched_rooms) == 0:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Move to target victim
                        self._rescue = 'together'
                        self._send_message('Moving to ' + self._found_victim_logs[vic][
                            'room'] + ' to pick up ' + self._goal_vic + '. Please come there as well to help me carry '
                                           + self._goal_vic + ' to the drop zone.',
                                           'RescueBot')
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}

                    # Define a previously found victim as target victim
                    if vic in self._found_victims and vic not in self._todo:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Rescue together when victim is critical or when the human is weak and the victim is mildly injured
                        if 'critical' in vic or 'mild' in vic and self._condition == 'weak':
                            self._rescue = 'together'
                        # Rescue alone if the victim is mildly injured and the human not weak
                        if 'mild' in vic and self._condition != 'weak':
                            self._rescue = 'alone'
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}

                    # If there are no target victims found, visit an unsearched area to search for victims
                    if vic not in self._found_victims or vic in self._found_victims and vic in self._todo and len(
                            self._searched_rooms) > 0:
                        self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearched_rooms = [room['room_name'] for room in state.values()
                                    if 'class_inheritance' in room
                                    and 'Door' in room['class_inheritance']
                                    and room['room_name'] not in self._searched_rooms
                                    and room['room_name'] not in self._to_search]

                # If all areas have been searched but the task is not finished, start searching areas again
                if self._remainingZones and len(unsearched_rooms) == 0:
                    # If all areas have been searched, but some are in reported_rooms, start searching those areas again
                    if len(self._reported_rooms) > 0:
                        # Get rooms that have actually been searched
                        self._searched_rooms = self.get_searched_from_reported()
                        self._reported_rooms = []
                    else:
                        self._searched_rooms = []
                    self._to_search = []
                    self._send_messages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._send_message('Going to re-search all areas.', 'RescueBot')
                    self._re_searching = True
                    self._phase = Phase.FIND_NEXT_GOAL
                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._current_door == None:
                        # Find all area entrance locations
                        self._door = \
                        state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, agent_location))[
                            0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._current_door != None:
                        self._door = \
                            state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, self._current_door))[0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, self._current_door))[-1][
                                'doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                # Reset the navigator for a new path planning
                self._navigator.reset_full()

                # Check if there is a goal victim, and it has been found, but its location is not known
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and 'location' not in self._found_victim_logs[self._goal_vic].keys():
                    # Retrieve the victim's room location and related information
                    victim_location = self._found_victim_logs[self._goal_vic]['room']
                    self._door = state.get_room_doors(victim_location)[0]
                    self._doormat = state.get_room(victim_location)[-1]['doormat']

                    # Handle special case for 'area 1'
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)

                    # Set the door location based on the doormat
                    doorLoc = self._doormat

                # If the goal victim's location is known, plan the route to the identified area
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat

                # Add the door location as a waypoint for navigation
                self._navigator.add_waypoints([doorLoc])
                # Follow the route to the next area to search
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                # Check if the previously identified target victim was rescued by the human
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    # Reset current door and switch to finding the next goal
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # If the identified victim was not rescued but the human sent a 'Rescue' message,
                # decrease its willingness

                # Check if the human found the previously identified target victim in a different room
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the human already searched the previously identified area without finding the target victim
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move to the next area to search
                else:
                    # Update the state tracker with the current state
                    self._state_tracker.update(state)

                    # Explain why the agent is moving to the specific area, either:
                    # [-] it contains the current target victim
                    # [-] it is the closest un-searched area
                    if self._goal_vic in self._found_victims \
                            and str(self._door['room_name']) == self._found_victim_logs[self._goal_vic]['room'] \
                            and not self._remove:
                        if self._condition == 'weak':
                            self._send_message('Moving to ' + str(
                                self._door['room_name']) + ' to pick up ' + self._goal_vic + ' together with you.',
                                               'RescueBot')
                        else:
                            self._send_message(
                                'Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goal_vic + '.',
                                'RescueBot')

                    if self._goal_vic not in self._found_victims and not self._remove or not self._goal_vic and not self._remove:
                        self._send_message(
                            'Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.',
                            'RescueBot')

                    # Set the current door based on the current location
                    self._current_door = self._door['location']

                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    # Check for obstacles blocking the path to the area and handle them if needed
                    if action is not None:
                        # Remove obstacles blocking the path to the area
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4),
                                                                                                                (9, 7),
                                                                                                                (9, 19),
                                                                                                                (21,
                                                                                                                 19)]:
                                self._send_message('Reaching ' + str(self._door['room_name'])
                                                   + ' will take a bit longer because I found stones blocking my path.',
                                                   'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    # Identify and remove obstacles if they are blocking the entrance of the area
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

                    # If person reported victim in room but it is blocked by obstacle - person lied so decrease willingness
                    if self._goal_vic in self._found_victims and self._door['room_name'] == self._found_victim_logs[self._goal_vic]['room']:
                        print("Reported victim in room but it is blocked by obstacle - person lied so decrease willingness")
                        self._willingness_additions.append((-0.10, CONFIDENCE_MEDIUM_IMPORTANCE))
                        self._found_victims.remove(self._goal_vic)
                        self._found_victim_logs.pop(self._goal_vic)
                        self._searched_rooms.remove(self._door['room_name'])
                        self._goal_vic = None
                        self._recent_vic = None
                        print(self._todo)

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                # Start timer for waiting for response from human
                if not self._waiting:
                    self._waiting_time_start = time.time()

                objects = []
                agent_location = state[self.agent_id]['location']
                # Identify which obstacle is blocking the entrance
                for info in state.values():

                    # Object found is a big rock, RescueBot needs to request help from the human agent
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance

                        if self._answered == False and not self._remove and not self._waiting:

                            # If re-searching area, then it means human lied about searching the area (since it is blocked by obstacle)
                            if self._re_searching:
                                self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))
                                print("re-searching area, then it means human lied about searching the area (since it is blocked by obstacle) so decrease willingness")

                            self._send_message('Found rock blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 5 seconds \n afstand - distance between us: ' + self._distance_human,
                                               'RescueBot')
                            self._waiting = True

                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:

                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL

                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            print("Remove here, inc will")
                            willInc = self._incWillWithResponseTime(time.time() - self._waiting_time_start)
                            self._willingness_additions.append((willInc, CONFIDENCE_LOW_IMPORTANCE))

                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle until human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove rock.',
                                    'RescueBot')
                                self._waiting_for_human_to_come = True
                                self._waiting_for_human_to_come_time_start = time.time()
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove rock blocking ' + str(self._door['room_name']) + '!',
                                                   'RescueBot')
                                self._waiting_for_human_to_come = True
                                self._waiting_for_human_to_come_time_start = time.time()
                                return None, {}

                        # If RescueBot has been waiting for more than 10 seconds with no answer or the willingness
                        # value is below a certain threshold, RescueBot continues searching
                        waiting_response_time_exceeded = self._waiting and not self._answered and time.time() - self._waiting_time_start > TIME_CONSTANT_WAIT

                        if waiting_response_time_exceeded or not self.decide_trust_message(trustBeliefs):

                            if (waiting_response_time_exceeded):
                                print("waiting_response_time_exceeded")
                                self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))

                            self._answered = True
                            self._waiting = False
                            if (waiting_response_time_exceeded):
                                self._send_message('No response, I am going to continue searching ' + str(self._door['room_name']) + '.',
                                                   'RescueBot')
                            else:
                                self._send_message('I am going to continue searching ' + str(self._door['room_name']) + '.',
                                                   'RescueBot')

                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                            return None, {}

                        # If robot has been waiting for human to come but it waited for too long
                        elif self._waiting_for_human_to_come and time.time() - self._waiting_for_human_to_come_time_start > TIME_TO_WAIT_FOR_HUMAN_TO_COME:
                            waited_in_seconds = time.time() - self._waiting_for_human_to_come_time_start
                            self._competence_additions.append((- (waited_in_seconds / 100), CONFIDENCE_LOW_IMPORTANCE*2))

                            # Continue with what robot was doing
                            self._waiting_for_human_to_come = False
                            self._waiting_for_human_to_come_time_start = None

                            self._waiting = False
                            self._send_message(
                                "You didn't arrive. I am going to continue searching.",
                                'RescueBot')
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                            return None, {}

                        # Remain idle until the human communicates what to do with the identified obstacle
                        else:
                            return None, {}

                    # Object found is a tree, RescueBot needs to remove it alone
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:

                            # If re-searching area, then it means human lied about searching the area (since it is blocked by obstacle)
                            if self._re_searching:
                                self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))
                                print("re-searching area, then it means human lied about searching the area (since it is blocked by obstacle) so decrease willingness")

                            self._send_message('Found tree blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + '\n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 10 seconds', 'RescueBot')
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the tree if the human tells the agent to do so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            print("remove tree here, inc will")
                            willInc = self._incWillWithResponseTime(time.time() - self._waiting_time_start)
                            self._willingness_additions.append((willInc, CONFIDENCE_LOW_IMPORTANCE))

                            if not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing tree blocking ' + str(self._door['room_name']) + '.',
                                                   'RescueBot')
                            if self._remove:
                                self._send_message('Removing tree blocking ' + str(
                                    self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                        # If RescueBot has been waiting for more than 10 seconds with no answer or the willingness
                        # value is below a certain threshold, RescueBot removes the tree on its own

                        waiting_response_time_exceeded = self._waiting and not self._answered and time.time() - self._waiting_time_start > TIME_CONSTANT_WAIT

                        if waiting_response_time_exceeded or not self.decide_trust_message(trustBeliefs):

                            if(waiting_response_time_exceeded):
                                print("waiting_response_time_exceeded")
                                self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))

                            self._answered = True
                            self._waiting = False

                            if(waiting_response_time_exceeded):
                                self._send_message('No response, removing tree blocking ' + str(self._door['room_name']) + '.',
                                                'RescueBot')
                            else:
                                self._send_message('Removing tree blocking ' + str(self._door['room_name']) + '.',
                                                'RescueBot')

                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                        # Remain idle untill the human communicates what to do with the identified obstacle
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)

                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:

                            # If re-searching area, then it means human lied about searching the area (since it is blocked by obstacle)
                            if self._re_searching:
                                self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))
                                print("re-searching area, then it means human lied about searching the area (since it is blocked by obstacle) so decrease willingness")

                            self._send_message('Found stones blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area', '') + ' \
                                \n clock - removal time together: 3 seconds \n afstand - distance between us: ' + self._distance_human + '\n clock - removal time alone: 20 seconds',
                                               'RescueBot')
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL

                        # Remove the obstacle alone if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove alone' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                               'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                        # Remove the obstacle together if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove together' or self._remove:

                            print("remove stone together here, inc will")
                            willInc = self._incWillWithResponseTime(time.time() - self._waiting_time_start)
                            self._willingness_additions.append((willInc, CONFIDENCE_LOW_IMPORTANCE))

                            if not self._remove:
                                self._answered = True

                            # Tell the human to come over and be idle until human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(
                                        self._door['room_name']) + ' to remove stones together.',
                                    'RescueBot')
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Lets remove stones blocking ' + str(self._door['room_name']) + '!',
                                    'RescueBot')
                                return None, {}

                        # If RescueBot has been waiting for more than 10 seconds with no answer or the willingness
                        # value is below a certain threshold, RescueBot removes the stones on its own
                        waiting_response_time_exceeded = self._waiting and not self._answered and time.time() - self._waiting_time_start > TIME_CONSTANT_WAIT

                        if waiting_response_time_exceeded or not self.decide_trust_message(trustBeliefs):

                            if (waiting_response_time_exceeded):
                                print("waiting_response_time_exceeded")
                                self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))

                            self._answered = True
                            self._waiting = False

                            if(waiting_response_time_exceeded):
                                self._send_message('No response, removing stones blocking ' + str(self._door['room_name'])
                                                   + '.', 'RescueBot')
                            else:
                                self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                                   'RescueBot')

                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                        # If robot has been waiting for human to come but it waited for too long
                        elif self._waiting_for_human_to_come and time.time() - self._waiting_for_human_to_come_time_start > TIME_TO_WAIT_FOR_HUMAN_TO_COME:
                            waited_in_seconds = time.time() - self._waiting_for_human_to_come_time_start
                            self._competence_additions.append((- (waited_in_seconds / 100), CONFIDENCE_LOW_IMPORTANCE*2))

                            # Continue with what robot was doing
                            self._waiting_for_human_to_come = False
                            self._waiting_for_human_to_come_time_start = None

                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                               'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                        # Remain idle until the human communicates what to do with the identified obstacle
                        else:
                            return None, {}

                    if "current_action" in info and info["current_action"] == "RemoveObjectTogether" and self._waiting_for_human_to_come:
                        waited_in_seconds = time.time() - self._waiting_for_human_to_come_time_start
                        competence_score = (- waited_in_seconds + REWARD_BORDER_TIME_TO_WAIT_FOR_HUMAN_TO_COME) / 100
                        self._competence_additions.append((competence_score, CONFIDENCE_LOW_IMPORTANCE*2))

                        # Continue with what robot was doing
                        self._waiting_for_human_to_come = False
                        self._waiting_for_human_to_come_time_start = None

                        self._send_message('Thank you for helping.', 'RescueBot')

                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM

                    if(self._human_asked_for_help_obstacle):
                        self._human_asked_for_help_obstacle = False
                        # Human lied about needing help with obstacle, decrease willingness
                        self._willingness_additions.append((-0.10, CONFIDENCE_LOW_IMPORTANCE))
                        print("Human lied about needing help with obstacle, decrease willingness")

                    # If the room has no obstacle and is re-searching and it finds a victim then we know that human is not competent at searching
                    if self._re_searching:
                        self._if_robot_finds_victim_human_incompetent = True

            if Phase.ENTER_ROOM == self._phase:
                self._answered = False

                # Check if the target victim has been rescued by the human, and switch to finding the next goal
                if self._goal_vic in self._collected_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the target victim is found in a different area, and start moving there
                if self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if area already searched without finding the target victim, and plan to search another area
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Enter the area and plan to search it
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, plan to search the room
                    if action is not None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                # Extract the numeric location from the room name and set it as the agent's location
                self._agent_loc = int(self._door['room_name'].split()[-1])

                # Store the locations of all area tiles in the current room
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._door['room_name']]
                self._roomtiles = room_tiles

                # Make the plan for searching the area
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(room_tiles))

                # Initialize variables for storing room victims and switch to following the room search path
                self._room_vics = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                # Search the area
                if not self._waiting:
                    self._waiting_time_start = time.time()

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    # Identify victims present in the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if vic not in self._room_vics:
                                self._room_vics.append(vic)

                            # Identify the exact location of the victim that was found by the human earlier
                            if vic in self._found_victims and 'location' not in self._found_victim_logs[vic].keys():
                                self._recent_vic = vic
                                # Add the exact victim location to the corresponding dictionary
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                if vic == self._goal_vic:
                                    # Communicate which victim was found
                                    self._send_message('Found ' + vic + ' in ' + self._door[
                                        'room_name'] + ' because you told me ' + vic + ' was located here.',
                                                       'RescueBot')
                                    # Add the area to the list with searched areas
                                    if self._door['room_name'] not in self._searched_rooms:
                                        self._searched_rooms.append(self._door['room_name'])
                                    # Do not continue searching the rest of the area but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL


                            # Identify injured victim in the area
                            if 'healthy' not in vic and vic not in self._found_victims:
                                self._recent_vic = vic
                                # Add the victim and the location to the corresponding dictionary
                                self._found_victims.append(vic)
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}

                                # If victim was found in previously searched room that didn't have blocking then person is not compitent at searching
                                if self._if_robot_finds_victim_human_incompetent:
                                    self._if_robot_finds_victim_human_incompetent = False
                                    self._competence_additions.append((-0.2, CONFIDENCE_LOW_IMPORTANCE*2))
                                    print("If victim was found in previously searched room that didn't have blocking then person is not compitent at searching")

                                # Communicate which victim the agent found and ask the human whether to rescue the victim now or at a later stage
                                if 'mild' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                        Important features to consider are: \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area ', '') + '\n \
                                        clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distance_human,
                                                       'RescueBot')
                                    self._waiting = True

                                if 'critical' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                        Important features to consider are: \n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area',
                                                                      '') + ' \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n \
                                        afstand - distance between us: ' + self._distance_human, 'RescueBot')
                                    self._waiting = True
                                    # Execute move actions to explore the area
                    return action, {}

                # Communicate that the agent did not find the target victim in the area while the human previously communicated the victim was located here
                if self._goal_vic in self._found_victims and self._goal_vic not in self._room_vics and \
                        self._found_victim_logs[self._goal_vic]['room'] == self._door['room_name']:
                    self._send_message(self._goal_vic + ' not present in ' + str(self._door[
                                                                                     'room_name']) + ' because I searched the whole area without finding ' + self._goal_vic + '.',
                                       'RescueBot')
                    # Remove the victim location from memory
                    self._found_victim_logs.pop(self._goal_vic, None)
                    self._found_victims.remove(self._goal_vic)
                    self._room_vics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searched_rooms:
                    self._searched_rooms.append(self._door['room_name'])

                # Make a plan to rescue a found critically injured victim if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue' and 'critical' in self._recent_vic:
                    print("Rescue critical")
                    willInc = self._incWillWithResponseTime(time.time() - self._waiting_time_start)
                    self._willingness_additions.append((willInc, CONFIDENCE_HIGH_IMPORTANCE))

                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False

                    # Tell the human to come over and help carry the critically injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                            self._recent_vic) + ' together.', 'RescueBot')
                        self._waiting_for_human_to_come = True
                        self._waiting_for_human_to_come_time_start = time.time()
                    # Tell the human to carry the critically injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(
                            self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                            self._recent_vic) + '.', 'RescueBot')
                        self._waiting_for_human_to_come = True
                        self._waiting_for_human_to_come_time_start = time.time()
                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM

                # Make a plan to rescue a found mildly injured victim together if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue together' and 'mild' in self._recent_vic:

                    print("Rescue together here, mild")
                    willInc = self._incWillWithResponseTime(time.time() - self._waiting_time_start)
                    self._willingness_additions.append((willInc, CONFIDENCE_MEDIUM_IMPORTANCE))
                    self._rescue = 'together'

                    self._answered = True
                    self._waiting = False

                    if self.calc_comp(trustBeliefs) >= -0.4:
                        self._rescue = 'together'
                        # Tell the human to come over and help carry the mildly injured victim
                        if not state[{'is_human_agent': True}]:
                            self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                                self._recent_vic) + ' together.', 'RescueBot')
                            self._waiting_for_human_to_come = True
                            self._waiting_for_human_to_come_time_start = time.time()
                        # Tell the human to carry the mildly injured victim together
                        if state[{'is_human_agent': True}]:
                            self._send_message('Lets carry ' + str(
                                self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                                self._recent_vic) + '.', 'RescueBot')
                            self._waiting_for_human_to_come = True
                            self._waiting_for_human_to_come_time_start = time.time()
                    # RescueBot will rescue the victim alone if the competence is too low
                    else:
                        self._rescue = 'alone'
                        self._send_message(
                            'I will pick up ' + str(self._recent_vic) + ' in ' + self._door['room_name'] + ' alone.',
                            'RescueBot')

                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM

                # Make a plan to rescue the mildly injured victim alone if the human decides so,
                # and communicate this to the human
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue alone' and 'mild' in self._recent_vic:
                    self._send_message(
                        'Picking up ' + str(self._recent_vic) + ' in ' + self._door['room_name'] + '.',
                        'RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._goal_vic = self._recent_vic
                    self._goal_loc = self._remaining[str(self._goal_vic)]
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM

                # Continue searching other areas if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._answered = True
                    self._waiting = False
                    self._todo.append(self._recent_vic)
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Remain idle untill the human communicates to the agent what to do with the found victim
                if self.received_messages_content and self._waiting and self.received_messages_content[
                    -1] != 'Rescue' and self.received_messages_content[-1] != 'Continue':

                    # RescueBot has been waiting for more than 10 seconds with no answer or the willingness
                    # value is below a certain threshold

                    waiting_response_time_exceeded = self._waiting and (not self._answered) and time.time() - self._waiting_time_start > TIME_CONSTANT_WAIT

                    if not self.decide_trust_message(trustBeliefs) or waiting_response_time_exceeded:
                        if waiting_response_time_exceeded:
                            print("waiting exceeded for victim response")
                            if 'mild' in self._recent_vic:
                                self._willingness_additions.append((-0.10, CONFIDENCE_MEDIUM_IMPORTANCE))
                            else:
                                self._willingness_additions.append((-0.10, CONFIDENCE_HIGH_IMPORTANCE))

                        self._answered = True
                        self._waiting = False

                        # If the victim is mildly injured, RescueBot picks them up on its own
                        if 'mild' in self._recent_vic:
                            self._goal_vic = self._recent_vic
                            if waiting_response_time_exceeded:
                                self._send_message(
                                    'No response, Picking up ' + str(self._goal_vic) + ' in ' + self._door['room_name'] + '.',
                                    'RescueBot')
                            else:
                                self._send_message(
                                    'Picking up ' + str(self._goal_vic) + ' in ' + self._door[
                                        'room_name'] + '.',
                                    'RescueBot')

                            self._rescue = 'alone'
                            self._answered = True
                            self._waiting = False
                            self._goal_vic = self._recent_vic
                            self._goal_loc = self._remaining[str(self._goal_vic)]
                            self._recent_vic = None
                            self._phase = Phase.PLAN_PATH_TO_VICTIM

                        # If the victim is critically injured, RescueBot continues searching
                        else:
                            self._send_message(
                                'No answer, I am going to continue searching ' + str(self._door['room_name']) + '.',
                                'RescueBot')
                            self._todo.append(self._recent_vic)
                            self._recent_vic = None
                            self._phase = Phase.FIND_NEXT_GOAL

                    return None, {}

                # Find the next area to search when the agent is not waiting for an answer from the human or occupied with rescuing a victim
                if not self._waiting and not self._rescue:
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                # Plan the path to a found victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._found_victim_logs[self._goal_vic]['location']])
                # Follow the path to the found victim
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # Start searching for other victims if the human already rescued the target victim
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move towards the location of the found victim
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, switch to taking the victim
                    if action is not None:
                        return action, {}
                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                # Store all area tiles in a list
                room_tiles = [info['location'] for info in state.values()
                              if 'class_inheritance' in info
                              and 'AreaTile' in info['class_inheritance']
                              and 'room_name' in info
                              and info['room_name'] == self._found_victim_logs[self._goal_vic]['room']]
                self._roomtiles = room_tiles
                objects = []

                for info in state.values():

                    # If robot has been waiting for human to come, but it waited for too long
                    if self._rescue == 'together' and self._waiting_for_human_to_come and time.time() - self._waiting_for_human_to_come_time_start > TIME_TO_WAIT_FOR_HUMAN_TO_COME:
                        waited_in_seconds = time.time() - self._waiting_for_human_to_come_time_start

                        # Continue with what robot was doing
                        self._waiting_for_human_to_come = False
                        self._waiting_for_human_to_come_time_start = None
                        self._waiting = False

                        # If mild victim just carry alone
                        if 'mild' in self._goal_vic:
                            self._send_message("You didn't arrive, I am saving the mildly injured victim alone.",
                                               'RescueBot')
                            self._rescue = 'alone'
                            self._competence_additions.append((- (waited_in_seconds / 100), CONFIDENCE_MEDIUM_IMPORTANCE*2))

                            # self._answered = True
                            self._waiting = False
                            self._goal_loc = self._remaining[str(self._goal_vic)]
                            self._recent_vic = None
                            self._phase = Phase.PLAN_PATH_TO_VICTIM

                        elif 'critical' in self._goal_vic:
                            self._send_message("Since you didn't come I will continue searching", 'RescueBot')
                            self._competence_additions.append((- (waited_in_seconds / 100), CONFIDENCE_HIGH_IMPORTANCE*2))
                            # self._answered = True
                            self._waiting = False
                            self._todo.append(self._goal_vic)
                            self._phase = Phase.FIND_NEXT_GOAL

                        return None, {}

                    # TODO: if human has not arrived to the location, decrease willingness value?
                    # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in \
                            info['obj_id'] and info['location'] in self._roomtiles or \
                            'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info[
                        'location'] in self._roomtiles and self._rescue == 'together' or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles:

                        # info contains victim that needs to be carried by both the robot and the human
                        objects.append(info)

                        # Remain idle when the human has not arrived at the location
                        if not self._human_name in info['name']:
                            self._waiting = True
                            self._moving = False
                            return None, {}

                # TODO: when victim has not been picked up in a long time, do we assume the task hasn't/won't happen?
                # I think this condition is executed when the victim is dropped
                # If RescueBot has been waiting for too long or the willingness is too low

                # Add the victim to the list of rescued victims when it has been picked up
                if len(objects) == 0 and 'critical' in self._goal_vic or len(
                        objects) == 0 and 'mild' in self._goal_vic and self._rescue == 'together':
                    self._waiting = False
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying_together = True
                    # Determine the next victim to rescue or search
                    self._phase = Phase.FIND_NEXT_GOAL

                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._found_victim_logs[self._goal_vic]['obj_id'],
                                                  'human_name': self._human_name}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goal_loc])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                # Communicate that the agent is transporting a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Transporting ' + self._goal_vic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                # Follow the path to the drop zone
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                # Drop the victim at the drop zone
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Delivered ' + self._goal_vic + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._current_door = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__, {'human_name': self._human_name}

    def _get_drop_zones(self, state):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones

    def _process_messages(self, state, teamMembers, condition, trustBeliefs):
        '''
        process incoming messages received from the team members
        '''

        receivedMessages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)

        # Check the content of the received messages
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                # If a received message involves team members searching areas, add these areas to the memory of areas that have been explored
                if msg.startswith("Search:") and msg not in self._processed_messages:
                    self._processed_messages.append(msg)
                    area = 'area ' + msg.split()[-1]
                    if area not in self._searched_rooms:

                        # check willingness is bigger than 0.4 to trust the human
                        if self.calc_will(trustBeliefs) >= 0.4:
                            self._searched_rooms.append(area)
                        elif (0.4 > self.calc_will(trustBeliefs) > -0.2) or self.calc_comp(trustBeliefs) < -0.4:
                            # between -0.4 and 0.4, trust but if task not completed, search those rooms first
                            # competence low then search those rooms first
                            self._searched_rooms.append(area)
                            self._reported_rooms.append(area)
                        else:
                            # dont trust
                            continue

                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:") and msg not in self._processed_messages:
                    self._processed_messages.append(msg)
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    if self.decide_trust_message(trustBeliefs):
                        # Add the area to the memory of searched areas
                        if loc not in self._searched_rooms:
                            self._searched_rooms.append(loc)
                        # Add the victim and its location to memory
                        if foundVic not in self._found_victims:
                            self._found_victims.append(foundVic)
                            self._found_victim_logs[foundVic] = {'room': loc}
                        if foundVic in self._found_victims and self._found_victim_logs[foundVic]['room'] != loc:
                            self._found_victim_logs[foundVic] = {'room': loc}
                        # Decide to help the human carry a found victim when the human's condition is 'weak'
                        if condition == 'weak':
                            self._rescue = 'together'
                        # Add the found victim to the to do list when the human's condition is not 'weak'
                        if 'mild' in foundVic and condition != 'weak':
                            self._todo.append(foundVic)
                # If a received message involves team members rescuing victims, add these victims and their locations to memory
                if msg.startswith('Collect:') and msg not in self._processed_messages:
                    self._processed_messages.append(msg)

                    toTrust = self.calc_will(trustBeliefs) > 0.8
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if toTrust and loc not in self._searched_rooms and self.decide_trust_message(trustBeliefs):
                        self._searched_rooms.append(loc)
                    # Add the victim and location to the memory of found victims
                    if toTrust and collectVic not in self._found_victims and self.decide_trust_message(trustBeliefs):
                        self._found_victims.append(collectVic)
                        self._found_victim_logs[collectVic] = {'room': loc}
                    if toTrust and (collectVic in self._found_victims and self._found_victim_logs[collectVic]['room'] != loc and
                            self.decide_trust_message(trustBeliefs)):
                        self._found_victim_logs[collectVic] = {'room': loc}
                    # Add the victim to the memory of rescued victims when the human's condition is not weak and human is trustworthy
                    if condition != 'weak' and collectVic not in self._collected_victims and toTrust:
                        self._collected_victims.append(collectVic)
                    # Decide to help the human carry the victim together when the human's condition is weak or when the human is not competent
                    if condition == 'weak' or self.calc_comp(trustBeliefs) < 0:
                        self._rescue = 'together'
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over
                if msg.startswith('Remove:') and msg not in self._processed_messages:
                    self._processed_messages.append(msg)
                    # Come over immediately when the agent is not carrying a victim
                    if self.decide_trust_message(trustBeliefs):
                        if not self._carrying:

                            #  Remember that human asked for help, removing an obstacle (to evaluate if obstacle was there)
                            self._human_asked_for_help_obstacle = True

                            # Identify at which location the human needs help
                            area = 'area ' + msg.split()[-1]
                            self._door = state.get_room_doors(area)[0]
                            self._doormat = state.get_room(area)[-1]['doormat']
                            if area in self._searched_rooms:
                                self._searched_rooms.remove(area)
                            # Clear received messages (bug fix)
                            self.received_messages = []
                            self.received_messages_content = []
                            self._moving = True
                            self._remove = True
                            if self._waiting and self._recent_vic:
                                self._todo.append(self._recent_vic)
                            self._waiting = False
                            # Let the human know that the agent is coming over to help
                            self._send_message(
                                'Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.',
                                'RescueBot')
                            # Plan the path to the relevant area
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                        # Come over to help after dropping a victim that is currently being carried by the agent
                        else:
                            area = 'area ' + msg.split()[-1]
                            self._send_message('Will come to ' + area + ' after dropping ' + self._goal_vic + '.',
                                               'RescueBot')
            # Store the current location of the human in memory
            if mssgs and len(mssgs) >= 0 and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                                   '14']:
                self._human_loc = int(mssgs[-1].split()[-1])



    def _loadBelief(self, members, folder):
        '''
        Loads trust belief values if agent already collaborated with human before, otherwise trust belief values are initialized using default values.
        '''
        # Create a dictionary with trust values for all team members
        trustBeliefs = {}
        # Set a default starting trust value
        default = 0.5
        trustfile_header = []
        trustfile_contents = []
        # Check if agent already collaborated with this human before, if yes: load the corresponding trust values, if no: initialize using default trust values
        with open(folder + '/beliefs/allTrustBeliefs.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if trustfile_header == []:
                    trustfile_header = row
                    continue
                # Retrieve trust values
                if row and row[0] == self._human_name:
                    name = row[0]
                    competence = float(row[1])
                    willingness = float(row[2])
                    confidenceComp = np.clip(float(row[3]) - 0.1, 0, 1)
                    confidenceWill = np.clip(float(row[4]) - 0.1, 0, 1)
                    trustBeliefs[name] = {'competence': competence, 'willingness': willingness, 'confidenceComp': confidenceComp, 'confidenceWill': confidenceWill}
                # Initialize default trust values
                if row and row[0] != self._human_name:
                    competence = default
                    willingness = default
                    confidenceComp = 0
                    confidenceWill = 0
                    trustBeliefs[self._human_name] = {'competence': competence, 'willingness': willingness, 'confidenceComp': confidenceComp, 'confidenceWill': confidenceWill}

        with open(folder + '/beliefs/currentTrustBelief.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                # Retrieve trust values
                if row and row[0] in self._team_members:
                    name = row[0]
                    competence = float(row[1])
                    willingness = float(row[2])
                    confidenceComp = float(row[3])
                    confidenceWill = float(row[4])
                    trustBeliefs[name] = {'competence': competence, 'willingness': willingness, 'confidenceComp': confidenceComp, 'confidenceWill': confidenceWill}

        return trustBeliefs

    def _trustBelief(self, members, trustBeliefs, folder, receivedMessages):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # Update the trust value based on for example the received messages
        for message in receivedMessages:
            # Increase agent trust in a team member that rescued a victim
            if 'Collect' in message and message not in self._processed_messages:
                self._processed_messages.append(message)

        for addition_to_consider in self._competence_additions:
            trustBeliefs[self._human_name]['competence'] += addition_to_consider[0]
            trustBeliefs[self._human_name]['confidenceComp'] += addition_to_consider[1]
            trustBeliefs[self._human_name]['competence'] = np.clip(trustBeliefs[self._human_name]['competence'], -1, 1)
            trustBeliefs[self._human_name]['confidenceComp'] = np.clip(trustBeliefs[self._human_name]['confidenceComp'], 0, 1)
        self._competence_additions = [] # reset

        for addition_to_consider in self._willingness_additions:
            trustBeliefs[self._human_name]['willingness'] += addition_to_consider[0]
            trustBeliefs[self._human_name]['confidenceWill'] += addition_to_consider[1]
            trustBeliefs[self._human_name]['willingness'] = np.clip(trustBeliefs[self._human_name]['willingness'], -1, 1)
            trustBeliefs[self._human_name]['confidenceWill'] = np.clip(trustBeliefs[self._human_name]['confidenceWill'], 0, 1)
        self._willingness_additions = [] # reset

        # Save current trust belief values so we can later use and retrieve them to add to a csv file with all the logged trust belief values
        with open(folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'competence', 'willingness', 'confidenceComp', 'confidenceWill'])
            csv_writer.writerow([self._human_name, trustBeliefs[self._human_name]['competence'],
                                 trustBeliefs[self._human_name]['willingness'], trustBeliefs[self._human_name]['confidenceComp'], trustBeliefs[self._human_name]['confidenceWill']])

        return trustBeliefs

    def _saveBelief(self, folder, trustBeliefs):
        trustfile_header = []
        old_trust_beliefs = {}
        need_rewrite = False  # Flag to determine if rewriting is necessary

        # Read the trust belief values from the csv file
        with open(folder + '/beliefs/allTrustBeliefs.csv', mode='r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if not trustfile_header:
                    trustfile_header = row
                    continue
                if row and row[0] != '':
                    old_trust_beliefs[row[0]] = {
                        'competence': float(row[1]),
                        'willingness': float(row[2]),
                        'confidenceComp': float(row[3]),
                        'confidenceWill': float(row[4])
                    }

        # Determine if an update is required or a new entry needs to be added
        if self._human_name in old_trust_beliefs:
            # Check for any change in trust values
            for key in ['competence', 'willingness', 'confidenceComp', 'confidenceWill']:
                if old_trust_beliefs[self._human_name][key] != trustBeliefs[self._human_name][key]:
                    need_rewrite = True
                    break
            # Update the values in old_trust_beliefs if changed
            if need_rewrite:
                old_trust_beliefs[self._human_name].update(trustBeliefs[self._human_name])
        else:
            # If the human is not already known, mark for rewrite to include this new entry
            need_rewrite = True
            old_trust_beliefs[self._human_name] = trustBeliefs[self._human_name]

        # Rewrite the entire file if an update or new addition is required
        if need_rewrite:
            with open(folder + '/beliefs/allTrustBeliefs.csv', mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';', quotechar="'", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(trustfile_header)  # Write the header first
                for name, values in old_trust_beliefs.items():
                    row = [
                        name,
                        values['competence'],
                        values['willingness'],
                        values['confidenceComp'],
                        values['confidenceWill']
                    ]
                    writer.writerow(row)

    def _send_message(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._send_messages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        '''
        calculate which area is closest to the agent's location
        '''
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        '''
        efficiently transverse areas instead of moving over every single area tile
        '''
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs

    # adds confidence to willingness calculation
    def calc_will(self, trustBeliefs):
        return trustBeliefs[self._human_name]['willingness'] * trustBeliefs[self._human_name]['confidenceWill']

    # adds cofidence to competence calculation
    def calc_comp(self, trustBeliefs):
        return trustBeliefs[self._human_name]['competence'] * trustBeliefs[self._human_name]['confidenceComp']

    # takes list of reported rooms and turns it into list of searched rooms
    def get_searched_from_reported(self):
        '''
        Reported rooms are rooms that were reported searched by an agent that we do not fully trust.
        When bot thinks all the rooms were searched, it will restart the search, first looking at the reported rooms.

        To do that we have to find the difference between set of all rooms and set of reported rooms.
        '''
        all_rooms = set(['area 1', 'area 2', 'area 3', 'area 4', 'area 5', 'area 6', 'area 7', 'area 8',
                         'area 9', 'area 10', 'area 11', 'area 12', 'area 13', 'area 14'])
        reported = set(self._reported_rooms)
        searched = all_rooms - reported
        return list(searched)

    # Function to decide whether to ignore a message based on trust values
    def decide_trust_message(self, trustBeliefs):
        '''
        if willingness > 0 then we will trust, if willinges < -0.8 we will not trust
        if willingness is between -0.8 and 0, for optimistic approach and benefit we will
        probabilistically decide to trust. Bigger the willingness bigger the chance we will trust.
        '''
        random_num = random.uniform(0, 1)
        willingness = self.calc_will(trustBeliefs)

        if willingness < -0.8:
            return False
        elif willingness > 0:
            return True
        else:
            return random_num > (willingness * (-1))

    def _incWillWithResponseTime(self, timeTaken):
        print("Inc by: ", (10 - timeTaken) / 50)
        return (10 - timeTaken) / 50
