#region VEXcode Generated Robot Configuration
from vex import *
import urandom

# Brain should be defined by default
brain=Brain()

# Robot configuration code
brain_inertial = Inertial()
IntakeGroup_motor_a = Motor(Ports.PORT5, False)
IntakeGroup_motor_b = Motor(Ports.PORT11, True)
IntakeGroup = MotorGroup(IntakeGroup_motor_a, IntakeGroup_motor_b)
ArmGroup_motor_a = Motor(Ports.PORT6, True)
ArmGroup_motor_b = Motor(Ports.PORT12, False)
ArmGroup = MotorGroup(ArmGroup_motor_a, ArmGroup_motor_b)
TopTouchLED = Touchled(Ports.PORT10)
controller = Controller()
left_drive_smart = Motor(Ports.PORT1, 2.0, False)
right_drive_smart = Motor(Ports.PORT7, 2.0, True)

drivetrain = SmartDrive(left_drive_smart, right_drive_smart, brain_inertial, 200)
motor_4 = Motor(Ports.PORT4, False)



# Make random actually random
def setRandomSeedUsingAccel():
    wait(100, MSEC)
    xaxis = brain_inertial.acceleration(XAXIS) * 1000
    yaxis = brain_inertial.acceleration(YAXIS) * 1000
    zaxis = brain_inertial.acceleration(ZAXIS) * 1000
    urandom.seed(int(xaxis + yaxis + zaxis))
    
# Set random seed 
setRandomSeedUsingAccel()

def calibrate_drivetrain():
    # Calibrate the Drivetrain Inertial
    sleep(200, MSEC)
    brain.screen.print("Calibrating")
    brain.screen.next_row()
    brain.screen.print("Inertial")
    brain_inertial.calibrate()
    while brain_inertial.is_calibrating():
        sleep(25, MSEC)
    brain.screen.clear_screen()
    brain.screen.set_cursor(1, 1)



# define variables used for controlling motors based on controller inputs
controller_left_shoulder_control_motors_stopped = True
controller_right_shoulder_control_motors_stopped = True
e_buttons_control_motors_stopped = True
drivetrain_needs_to_be_stopped_controller = False

# define a task that will handle monitoring inputs from controller
def rc_auto_loop_function_controller():
    global drivetrain_needs_to_be_stopped_controller, controller_left_shoulder_control_motors_stopped, controller_right_shoulder_control_motors_stopped, e_buttons_control_motors_stopped, remote_control_code_enabled
    # process the controller input every 20 milliseconds
    # update the motors based on the input values
    while True:
        if remote_control_code_enabled:
            
            # calculate the drivetrain motor velocities from the controller joystick axies
            # left = axisA + axisB
            # right = axisA - axisB
            drivetrain_left_side_speed = controller.axisA.position() + controller.axisB.position()
            drivetrain_right_side_speed = controller.axisA.position() - controller.axisB.position()
            
            # check if the values are inside of the deadband range
            if abs(drivetrain_left_side_speed) < 5 and abs(drivetrain_right_side_speed) < 5:
                # check if the motors have already been stopped
                if drivetrain_needs_to_be_stopped_controller:
                    # stop the drive motors
                    left_drive_smart.stop()
                    right_drive_smart.stop()
                    # tell the code that the motors have been stopped
                    drivetrain_needs_to_be_stopped_controller = False
            else:
                # reset the toggle so that the deadband code knows to stop the motors next
                # time the input is in the deadband range
                drivetrain_needs_to_be_stopped_controller = True
            
            # only tell the left drive motor to spin if the values are not in the deadband range
            if drivetrain_needs_to_be_stopped_controller:
                left_drive_smart.set_velocity(drivetrain_left_side_speed, PERCENT)
                left_drive_smart.spin(FORWARD)
            # only tell the right drive motor to spin if the values are not in the deadband range
            if drivetrain_needs_to_be_stopped_controller:
                right_drive_smart.set_velocity(drivetrain_right_side_speed, PERCENT)
                right_drive_smart.spin(FORWARD)
            # check the buttonLUp/buttonLDown status
            # to control IntakeGroup
            if controller.buttonLUp.pressing():
                IntakeGroup.spin(REVERSE)
                controller_left_shoulder_control_motors_stopped = False
            elif controller.buttonLDown.pressing():
                IntakeGroup.spin(FORWARD)
                controller_left_shoulder_control_motors_stopped = False
            elif not controller_left_shoulder_control_motors_stopped:
                IntakeGroup.stop()
                # set the toggle so that we don't constantly tell the motor to stop when
                # the buttons are released
                controller_left_shoulder_control_motors_stopped = True
            # check the buttonRUp/buttonRDown status
            # to control ArmGroup
            if controller.buttonRUp.pressing():
                ArmGroup.spin(FORWARD)
                controller_right_shoulder_control_motors_stopped = False
            elif controller.buttonRDown.pressing():
                ArmGroup.spin(REVERSE)
                controller_right_shoulder_control_motors_stopped = False
            elif not controller_right_shoulder_control_motors_stopped:
                ArmGroup.stop()
                # set the toggle so that we don't constantly tell the motor to stop when
                # the buttons are released
                controller_right_shoulder_control_motors_stopped = True
            # check the buttonEUp/buttonEDown status
            # to control motor_4
            if controller.buttonEUp.pressing():
                motor_4.spin(FORWARD)
                e_buttons_control_motors_stopped = False
            elif controller.buttonEDown.pressing():
                motor_4.spin(REVERSE)
                e_buttons_control_motors_stopped = False
            elif not e_buttons_control_motors_stopped:
                motor_4.stop()
                # set the toggle so that we don't constantly tell the motor to stop when
                # the buttons are released
                e_buttons_control_motors_stopped = True
        # wait before repeating the process
        wait(20, MSEC)

# define variable for remote controller enable/disable
remote_control_code_enabled = True

rc_auto_loop_thread_controller = Thread(rc_auto_loop_function_controller)

#endregion VEXcode Generated Robot Configuration

import time

modes=[
Color.BLUE,
Color.BLUE_GREEN,
Color.BLUE_VIOLET,
Color.GREEN,
Color.ORANGE,
Color.PURPLE,
Color.RED,
Color.RED_ORANGE,
Color.RED_VIOLET,
Color.VIOLET,
Color.WHITE,
Color.YELLOW,
Color.YELLOW_GREEN,
Color.YELLOW_ORANGE]
stop_=True
click=0
def callback():
    global stop_
    stop_=True
    while stop_:
        for i in modes:
            if stop_:
                # brain.screen.set_fill_color(i)
                TopTouchLED.set_color(i)
                time.sleep(0.2)
                # brain.screen.draw_rectangle(0, 0, 159, 107)
    TopTouchLED.set_color(Color.BLACK)
time_now=time.time()

def callback2():
    global stop_
    if stop_:
        stop_=False
    else:
        brain.play_sound(SoundType.ALARM2)
def flip():
    global click
    click+=1
    # brain.play_sound(SoundType.WRONG_WAY)
    if click == 3:
        click=0
        ArmGroup.spin_to_position(1149, DEGREES)
        # callback()
        ArmGroup.spin_to_position(0, DEGREES)
        TopTouchLED.set_color(Color.GREEN)
        time.sleep(0.2)
        TopTouchLED.set_color(Color.BLACK)
        return
    TopTouchLED.set_color(Color.WHITE)
    time.sleep(0.2)
    TopTouchLED.set_color(Color.BLACK)

calibrate_drivetrain()
IntakeGroup.set_stopping(HOLD)
ArmGroup.set_stopping(HOLD)
motor_4.set_stopping(HOLD)
drivetrain.set_stopping(HOLD)
IntakeGroup.set_velocity(100,PERCENT)
# drivetrain.set_velocity(100,PERCENT)
motor_4.set_velocity(25,PERCENT)
TopTouchLED.pressed(callback)
controller.buttonFUp.pressed(callback)
controller.buttonFDown.pressed(callback2)
controller.buttonR3.pressed(flip)
brain.screen.print("Haha I made you")
brain.screen.next_row()
brain.screen.print("waste 3.826428")
brain.screen.next_row()
brain.screen.print("seconds of your")
brain.screen.next_row()
brain.screen.print("life reading")
brain.screen.next_row()
brain.screen.print("this text")