from controller import Robot

def run_simulation():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    left_motor = robot.getDevice('left_wheel_motor')
    right_motor = robot.getDevice('right_wheel_motor')

    if left_motor and right_motor:
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

    ds_front = robot.getDevice('ds_front')
    if ds_front:
        ds_front.enable(timestep)

    camera = robot.getDevice('camera')
    if camera:
        camera.enable(timestep)

    print("Mower Controller Initialized. Starting Loop...")

    while robot.step(timestep) != -1:
        dist_val = ds_front.getValue() if ds_front else 2000.0
        
        # Simple obstacle avoidance logic
        if dist_val < 500.0:
            left_motor.setVelocity(-2.0)
            right_motor.setVelocity(2.0)
        else:
            left_motor.setVelocity(3.0)
            right_motor.setVelocity(3.0)

if __name__ == "__main__":
    run_simulation()