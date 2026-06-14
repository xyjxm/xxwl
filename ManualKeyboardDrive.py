import argparse
import ctypes
import time

import runtime_paths

runtime_paths.configure()

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from Setup_Competition import setup


KEYS = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "q": 0x51,
    "space": 0x20,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}


def key_down(name):
    return bool(ctypes.windll.user32.GetAsyncKeyState(KEYS[name]) & 0x8000)


def stop_car(car, brake=True):
    try:
        car.set_velocity_and_request_state(
            forward=0,
            turn=0,
            headlights=False,
            leftTurnSignal=False,
            rightTurnSignal=False,
            brakeSignal=brake,
            reverseSignal=False,
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Manual keyboard control for QLabs QCar.")
    parser.add_argument("--scenario", type=int, default=3, help="Scenario to spawn when not using --attach.")
    parser.add_argument("--attach", action="store_true", help="Attach to existing QCar actor 0 instead of resetting the scene.")
    parser.add_argument("--speed", type=float, default=0.35, help="Forward speed while W/Up is held.")
    parser.add_argument("--reverse-speed", type=float, default=0.16, help="Reverse speed while S/Down is held.")
    parser.add_argument("--turn", type=float, default=0.34, help="Steering command while A/D or arrows are held.")
    parser.add_argument("--period", type=float, default=0.04, help="Control update period in seconds.")
    args = parser.parse_args()

    qlabs = None
    if args.attach:
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        car = QLabsQCar(qlabs)
        car.actorNumber = 0
        car.possess()
    else:
        car = setup(scenario_num=args.scenario)
        qlabs = car._qlabs

    print()
    print("Manual keyboard control is active.")
    print("  W / Up    : forward")
    print("  S / Down  : reverse")
    print("  A / Left  : turn left")
    print("  D / Right : turn right")
    print("  Space     : brake")
    print("  Q         : quit")
    print()
    print("Focus this console window while driving.")

    try:
        while True:
            if key_down("q"):
                break

            brake = key_down("space")
            forward = 0.0
            turn = 0.0

            if not brake:
                if key_down("w") or key_down("up"):
                    forward += args.speed
                if key_down("s") or key_down("down"):
                    forward -= args.reverse_speed
                if key_down("a") or key_down("left"):
                    turn += args.turn
                if key_down("d") or key_down("right"):
                    turn -= args.turn

            status, position, orientation, _, _ = car.set_velocity_and_request_state(
                forward=forward,
                turn=turn,
                headlights=False,
                leftTurnSignal=False,
                rightTurnSignal=False,
                brakeSignal=brake,
                reverseSignal=forward < 0,
            )

            if status:
                print(
                    f"\rpos=({position[0]: .3f}, {position[1]: .3f}) "
                    f"yaw={orientation[2]: .2f} forward={forward: .2f} turn={turn: .2f}   ",
                    end="",
                    flush=True,
                )

            time.sleep(args.period)
    except KeyboardInterrupt:
        pass
    finally:
        stop_car(car, brake=True)
        if qlabs is not None:
            qlabs.close()
        print("\nManual keyboard control stopped.")


if __name__ == "__main__":
    main()
