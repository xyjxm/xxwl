import time
import math
import struct
import argparse
import signal
import sys
import os
import threading
import runtime_paths

BASE_DIR = runtime_paths.configure()

from quanser.communications import Stream

from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_light import QLabsTrafficLight
from qvl.person import QLabsPerson
from qvl.animal import QLabsAnimal
from qvl.generic_sensor import QLabsGenericSensor

# getting scenartio number from command line
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--scenario", type=int, default=3)
args = arg_parser.parse_args()
scenario_num = args.scenario


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


TRAFFIC_ACTOR_MODE = os.environ.get("TRAFFIC_ACTOR_MODE", "triggered").strip().lower()
TRAFFIC_LIGHT_MODE = os.environ.get("TRAFFIC_LIGHT_MODE", "triggered").strip().lower()

if env_flag("TRAFFIC_DETERMINISTIC", False):
    TRAFFIC_ACTOR_MODE = os.environ.get("TRAFFIC_ACTOR_MODE", "scripted").strip().lower()
    TRAFFIC_LIGHT_MODE = os.environ.get("TRAFFIC_LIGHT_MODE", "fixed_green").strip().lower()

if TRAFFIC_ACTOR_MODE not in ("triggered", "scripted", "static_start", "static_endpoints"):
    TRAFFIC_ACTOR_MODE = "triggered"

if TRAFFIC_LIGHT_MODE not in ("triggered", "fixed_green"):
    TRAFFIC_LIGHT_MODE = "triggered"

SCRIPTED_PERSON1_DELAY = env_float("TRAFFIC_PERSON1_DELAY", 0.0)
SCRIPTED_PERSON2_DELAY = env_float("TRAFFIC_PERSON2_DELAY", 0.0)
SCRIPTED_COW_DELAY = env_float("TRAFFIC_COW_DELAY", 0.0)

READY_FILE = os.path.join(BASE_DIR, "traffic_ready.flag")
STATUS_FILE = os.path.join(BASE_DIR, "traffic_status.txt")


def _remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def write_status(message, ready=False):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with open(STATUS_FILE, "a", encoding="utf-8") as file:
        file.write(line + "\n")
    if ready:
        with open(READY_FILE, "w", encoding="utf-8") as file:
            file.write(line + "\n")


for stale_file in (READY_FILE, STATUS_FILE):
    _remove_if_exists(stale_file)
write_status(f"Traffic script started; scenario={scenario_num}")
write_status(
    "Traffic config: "
    f"actors={TRAFFIC_ACTOR_MODE}, lights={TRAFFIC_LIGHT_MODE}, "
    f"delays=({SCRIPTED_PERSON1_DELAY:.2f},"
    f"{SCRIPTED_PERSON2_DELAY:.2f},{SCRIPTED_COW_DELAY:.2f})"
)

# handler for SIGTERM
def terminate_handler(signal, frame):
    """
    Signal handler for termination
    终止信号处理函数

    Args:
        signal (int): Signal number / 信号号
        frame (frame): Current frame / 当前帧
    """
    qlabs.destroy_all_spawned_actors()
    qlabs.close()



# creates a server connection with Quanser Interactive Labs and manages the communications
qlabs = QuanserInteractiveLabs()

print("Traffic Control is connecting to QLabs...")
# trying to connect to QLabs and open the instance we have created - program will end if this fails
try:
    qlabs.open("localhost")
    print("Traffic Control connected to QLabs")
    write_status("Traffic Control connected to QLabs")
except Exception as exc:
    write_status(f"ERROR: unable to connect to QLabs: {exc}")
    sys.exit(1)

# qcar map offset
x_offset = 0.13
y_offset = 1.67

#   set sensors
TrafficTrigger0 = QLabsGenericSensor(qlabs)
TrafficTrigger1 = QLabsGenericSensor(qlabs)
PeopleTrigger0 = QLabsGenericSensor(qlabs)
PeopleTrigger1 = QLabsGenericSensor(qlabs)
CowTrigger0 = QLabsGenericSensor(qlabs)
Checkpoint0 = QLabsGenericSensor(qlabs)
Checkpoint1 = QLabsGenericSensor(qlabs)
QCarTestSensor = QLabsGenericSensor(qlabs)

TrafficTrigger0.spawn_degrees([2.109, -0.95, 0.1], [0, 0, -45], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
TrafficTrigger1.spawn_degrees([-2.196, 2.8, 0.1], [0, 0, 0], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
PeopleTrigger0.spawn_degrees([-2.194, 3.9, 0.1], [0, 0, 0], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
PeopleTrigger1.spawn_degrees([0.241, -1.07, 0.1], [0, 0, 90], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
CowTrigger0.spawn_degrees([0.631, 3.808, 0.1], [0, 0, 90], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
Checkpoint0.spawn_degrees([2.335, 0.989, 0.1], [0, 0, 0], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
Checkpoint1.spawn_degrees([-2.149, 0.98, 0.1], [0, 0, 0], scale=[1, 1, 1], configuration=0, waitForConfirmation=True)
QCarTestSensor.spawn_id_and_parent_with_relative_transform(actorNumber=150, location=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1], configuration=0, parentClassID=160, parentActorNumber=0, parentComponent=0, waitForConfirmation=True)

TrafficTrigger0.set_beam_size(startDistance=0, endDistance=-0.8, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
TrafficTrigger1.set_beam_size(startDistance=0, endDistance=0.8, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
PeopleTrigger0.set_beam_size(startDistance=0, endDistance=1.0, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
PeopleTrigger1.set_beam_size(startDistance=0, endDistance=0.5, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
CowTrigger0.set_beam_size(startDistance=0, endDistance=0.8, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
Checkpoint0.set_beam_size(startDistance=0, endDistance=-0.8, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
Checkpoint1.set_beam_size(startDistance=0, endDistance=0.8, heightOrRadius=0.01, width=0.01, waitForConfirmation=True)
QCarTestSensor.set_beam_size(startDistance=0, endDistance=0.8, heightOrRadius=0.1, waitForConfirmation=True)

# TrafficTrigger0.show_sensor()
# TrafficTrigger1.show_sensor()
# PeopleTrigger0.show_sensor()   
# PeopleTrigger1.show_sensor()
# CowTrigger0.show_sensor()
# Checkpoint0.show_sensor()
# Checkpoint1.show_sensor()
# QCarTestSensor.show_sensor()

def test_beam_hit_safe(sensor):
    result = sensor.test_beam_hit()
    if len(result) == 5:
        return result
    if len(result) == 4:
        status, hit, actor_class, actor_number = result
        return status, hit, actor_class, actor_number, 0.0
    return False, False, 0, 0, 0.0

# traffic light
TrafficLight0 = QLabsTrafficLight(qlabs)
TrafficLight0.spawn_degrees([2.3 + x_offset, y_offset, 0], [0, 0, 0], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
if scenario_num > 1:
    TrafficLight1 = QLabsTrafficLight(qlabs)
    TrafficLight1.spawn_degrees([-2.3 + x_offset, -1 + y_offset, 0], [0, 0, 180], scale=[.1, .1, .1], configuration=0, waitForConfirmation=True)
    TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)

# Add person and cow
# Define three different paths
paths = {
    1: ([-1.451, 3.172, 0.006], [-2.2, 3.1722, 0.006]),      # Original path
    2: ([1.1, -0.56, 0.006], [1.1, -1.3, 0.006]),    # Second path
    3: ([-0.159, 3.9, 0.006], [-0.159, 4.6, 0.006])       # Third path
}

Endpoint1, Startpoint1 = paths[1]
Startpoint2, Endpoint2 = paths[2]
Endpoint3, Startpoint3 = paths[3]

def format_location(location):
    return ",".join(f"{value:.3f}" for value in location)


def spawn_and_confirm(label, actor, location, rotation, scale, configuration):
    last_status = -1
    last_actor_number = -1
    last_transform_ok = False
    last_location = [0.0, 0.0, 0.0]

    for attempt in range(1, 9):
        status, actor_number = actor.spawn_degrees(
            location=location,
            rotation=rotation,
            scale=scale,
            configuration=configuration,
            waitForConfirmation=True,
        )
        transform_ok, actual_location, _, _ = actor.get_world_transform()
        write_status(
            f"{label}: attempt={attempt} spawn_status={status} actor={actor_number} "
            f"transform_ok={transform_ok} location={format_location(actual_location)}"
        )
        if status == 0 and transform_ok:
            return

        last_status = status
        last_actor_number = actor_number
        last_transform_ok = transform_ok
        last_location = actual_location
        time.sleep(0.8)

    write_status(
        f"{label}: final spawn_status={last_status} actor={last_actor_number} "
        f"transform_ok={last_transform_ok} location={format_location(last_location)}"
    )
    raise RuntimeError(f"{label} failed to load")


actor_move_done = {
    "person1": False,
    "person2": False,
    "cow1": False,
}
scripted_start_time = None


def move_actor_once(label, actor, location, speed):
    if actor_move_done.get(label, False):
        return
    actor.move_to(location=location, speed=speed, waitForConfirmation=True)
    actor_move_done[label] = True
    write_status(f"{label}: move_to {format_location(location)} speed={speed:.2f}")


try:
    if scenario_num == 3:
        time.sleep(2.0)
        person1 = QLabsPerson(qlabs)
        person2 = QLabsPerson(qlabs)
        cow1 = QLabsAnimal(qlabs)
        person1_spawn = Endpoint1 if TRAFFIC_ACTOR_MODE == "static_endpoints" else Startpoint1
        person2_spawn = Endpoint2 if TRAFFIC_ACTOR_MODE == "static_endpoints" else Startpoint2
        cow1_spawn = Endpoint3 if TRAFFIC_ACTOR_MODE == "static_endpoints" else Startpoint3
        spawn_and_confirm("person1", person1, person1_spawn, [0, 0, 0], [0.1, 0.1, 0.1], 10)
        spawn_and_confirm("person2", person2, person2_spawn, [0, 0, 0], [0.1, 0.1, 0.1], 10)
        spawn_and_confirm("cow1", cow1, cow1_spawn, [0, 0, -90], [0.1, 0.1, 0.1], QLabsAnimal.COW)
        if TRAFFIC_ACTOR_MODE == "static_endpoints":
            actor_move_done["person1"] = True
            actor_move_done["person2"] = True
            actor_move_done["cow1"] = True
            write_status("static_endpoints: actors spawned at endpoint locations")
        scripted_start_time = time.time()
        write_status("READY: required actors loaded; people=2 cow=1", ready=True)
    else:
        write_status(f"READY: scenario={scenario_num}; people=0 cow=0", ready=True)
except Exception as exc:
    write_status(f"ERROR: traffic actor setup failed: {exc}")
    qlabs.destroy_all_spawned_actors()
    qlabs.close()
    raise

i = 0
checkscore0 = 0
checkscore1 = 0
timer0 = None  # 第一个交通灯的计时器
timer1 = None  # 第二个交通灯的计时器
timer0_elapsed = None
timer1_elapsed = None

try:
    while (True):
        status0, Traffic0_hit, actorClass0, actorNumber0, distance0 = test_beam_hit_safe(TrafficTrigger0)
        status1, Traffic1_hit, actorClass1, actorNumber1, distance1 = test_beam_hit_safe(TrafficTrigger1)
        status2, People0_hit, actorClass2, actorNumber2, distance2 = test_beam_hit_safe(PeopleTrigger0)
        status3, People1_hit, actorClass3, actorNumber3, distance3 = test_beam_hit_safe(PeopleTrigger1)
        status4, Cow0_hit, actorClass4, actorNumber4, distance4 = test_beam_hit_safe(CowTrigger0)
        status5, Checkpoint0_hit, actorClass5, actorNumber5, distance5 = test_beam_hit_safe(Checkpoint0)
        status6, Checkpoint1_hit, actorClass6, actorNumber6, distance6 = test_beam_hit_safe(Checkpoint1)

        if TRAFFIC_LIGHT_MODE == "fixed_green":
            Traffic0_hit = False
            Traffic1_hit = False
            Checkpoint0_hit = False
            Checkpoint1_hit = False
            TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)
            if scenario_num > 1:
                TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)

        if TRAFFIC_ACTOR_MODE != "triggered":
            People0_hit = False
            People1_hit = False
            Cow0_hit = False


        if checkscore0 != 0 or checkscore1 != 0:
            sys.stdout.write(f"\r罚时+{checkscore0}s,+{checkscore1}s")
        
        # 更新计时器0
        if timer0 is not None:
            timer0_elapsed = time.time() - timer0
            if timer0_elapsed > 5:
                timer0 = None
                timer0_elapsed = None
        # 处理第一个检查点
            elif timer0_elapsed < 4 and Checkpoint0_hit and actorClass5 == 160:
                checkscore0 = 10
        # 更新计时器1
        if timer1 is not None:
            timer1_elapsed = time.time() - timer1
            if timer1_elapsed > 5:
                timer1 = None
                timer1_elapsed = None

        # 处理第一个交通灯
        if timer0 is None and Traffic0_hit and actorClass0 == 160:
                timer0 = time.time()
                TrafficLight0.set_state(QLabsTrafficLight.STATE_RED)
        if timer0_elapsed is not None and timer0_elapsed > 4:
            TrafficLight0.set_state(QLabsTrafficLight.STATE_GREEN)

        if scenario_num == 2:
            if timer1 is None and Traffic1_hit and actorClass1 == 160:
                timer1 = time.time()
                TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)

            if timer1_elapsed is not None:
                if (timer1_elapsed < 4 and
                    Checkpoint1_hit and 
                    actorClass6 == 160):
                    checkscore1 = 10
                elif timer1_elapsed > 4:
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)

        elif scenario_num == 3:
            if timer1 is None and Traffic1_hit and actorClass1 == 160:
                if timer1 is None:
                    timer1 = time.time()
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_RED)
                  
            if timer1_elapsed is not None: 
                if (timer1_elapsed < 4 and
                    Checkpoint1_hit and 
                    actorClass6 == 160):
                    checkscore1 = 10
                elif timer1_elapsed > 4:
                    TrafficLight1.set_state(QLabsTrafficLight.STATE_GREEN)

                
            if People0_hit and actorClass2 == 160:
                person1.move_to(location=Endpoint1, speed=0.25, waitForConfirmation=True)
            if People1_hit and actorClass3 == 160:
                person2.move_to(location=Endpoint2, speed=0.5, waitForConfirmation=True)
            if Cow0_hit and actorClass4 == 160:
                cow1.move_to(location=Endpoint3, speed=0.2, waitForConfirmation=True)

            if TRAFFIC_ACTOR_MODE == "scripted" and scripted_start_time is not None:
                scripted_elapsed = time.time() - scripted_start_time
                if scripted_elapsed >= SCRIPTED_PERSON1_DELAY:
                    move_actor_once("person1", person1, Endpoint1, 0.25)
                if scripted_elapsed >= SCRIPTED_PERSON2_DELAY:
                    move_actor_once("person2", person2, Endpoint2, 0.5)
                if scripted_elapsed >= SCRIPTED_COW_DELAY:
                    move_actor_once("cow1", cow1, Endpoint3, 0.2)

            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, terminate_handler)
                signal.signal(signal.SIGINT, terminate_handler)

except KeyboardInterrupt:
    qlabs.destroy_all_spawned_actors()
    qlabs.close()

qlabs.destroy_all_spawned_actors()
qlabs.close()
print("Done!")
