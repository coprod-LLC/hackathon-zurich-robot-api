from __future__ import annotations
from dataclasses import dataclass
import cv2
# import cv2.typing # Ensure this is uncommented
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped, Twist # Import TwistStamped and Twist
from cv_bridge import CvBridge, CvBridgeError
import threading
import time
import math # Import math for sqrt
import copy # For deepcopy

@dataclass
class Point2d:
    x: int = 0
    y: int = 0


class Omnibot:
    def __init__(self, node_name="omnibot_client_node"):
        """
        Initializes the Omnibot, sets up a ROS2 node, and subscribes to the camera topic.
        """
        if not rclpy.ok():
            rclpy.init()
            self._ros_initialized_by_me = True
        else:
            self._ros_initialized_by_me = False

        self.node = rclpy.create_node(node_name)
        self.bridge = CvBridge()
        self.current_frame: cv2.typing.MatLike | None = None # Added type hint
        self.frame_lock = threading.Lock()

        self.current_thermal_frame: cv2.typing.MatLike | None = None # Added type hint
        self.thermal_frame_lock = threading.Lock()

        self.current_command_twist = Twist() # Stores the target twist
        self.command_twist_lock = threading.Lock()
        self.cmd_vel_publisher = self.node.create_publisher(TwistStamped, '/omnibot/cmd_vel', 10)
        self.cmd_vel_rate = 50.0  # Hz
        self.shutdown_event = threading.Event()
        self.movement_revert_timer: threading.Timer | None = None

        self.subscription = self.node.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed', # Standard topic for compressed images
            self._image_callback,
            10) # QoS profile depth
        
        self.thermal_subscription = self.node.create_subscription(
            CompressedImage,
            '/thermal_image/compressed',
            self._thermal_image_callback,
            10) # QoS profile depth

        self.cmd_vel_publisher_thread = threading.Thread(target=self._publish_cmd_vel_loop, daemon=True)
        self.executor_thread = threading.Thread(target=self._spin_node, daemon=True)
        self.executor_thread.start()
        self.cmd_vel_publisher_thread.start() # Start the new cmd_vel publishing thread
        self.node.get_logger().info(f'{node_name} started, cmd_vel publisher running at {self.cmd_vel_rate}Hz, image subs active.')

    def _spin_node(self):
        """Spins the node in a separate thread."""
        try:
            rclpy.spin(self.node)
        except rclpy.exceptions.RCLError:
            # This can happen if rclpy.shutdown() is called elsewhere
            self.node.get_logger().info("RCLPY spin interrupted, likely due to shutdown.")
        finally:
            # Ensure the node is destroyed if spinning stops unexpectedly
            if rclpy.ok() and self.node and not self.node.executor.is_shutdown:
                 self.node.destroy_node()


    def _image_callback(self, msg: CompressedImage):
        """
        Callback for the image subscription. Converts and stores the received frame.
        """
        # self.node.get_logger().debug('Image received') # For debugging
        try:
            # Convert compressed ROS image message to OpenCV image (BGR format)
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.current_frame = cv_image
        except CvBridgeError as e:
            self.node.get_logger().error(f'CV Bridge Error: {e}')
        except Exception as e:
            self.node.get_logger().error(f'Error processing image: {e}')

    def _thermal_image_callback(self, msg: CompressedImage):
        """
        Callback for the thermal image subscription. Converts and stores the received frame.
        """
        # self.node.get_logger().debug('Thermal image received') # For debugging
        try:
            # Convert compressed ROS image message to OpenCV image
            # Using 'bgr8' as desired_encoding. If thermal is mono, cv_bridge might convert or return grayscale.
            cv_thermal_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.thermal_frame_lock:
                self.current_thermal_frame = cv_thermal_image
        except CvBridgeError as e:
            self.node.get_logger().error(f'CV Bridge Error (Thermal): {e}')
        except Exception as e:
            self.node.get_logger().error(f'Error processing thermal image: {e}')

    def _publish_cmd_vel_loop(self):
        """Continuously publishes the current_command_twist at a fixed rate."""
        rate = self.node.create_rate(self.cmd_vel_rate)
        self.node.get_logger().info("Starting cmd_vel publishing loop...")
        while rclpy.ok() and not self.shutdown_event.is_set():
            cmd_to_publish = TwistStamped()
            cmd_to_publish.header.stamp = self.node.get_clock().now().to_msg()
            # cmd_to_publish.header.frame_id = "base_link" # Optional
            
            with self.command_twist_lock:
                # Make a copy to ensure thread safety during publish, 
                # especially if the Twist object itself could be complex or have nested structures.
                # For standard Twist, direct assignment might be okay, but copy is safer.
                cmd_to_publish.twist = copy.deepcopy(self.current_command_twist)
            
            self.cmd_vel_publisher.publish(cmd_to_publish)
            try:
                rate.sleep()
            except rclpy.exceptions.RCLError:
                 # This can happen if rclpy.shutdown() is called while sleep is active
                self.node.get_logger().info("Rate sleep interrupted, likely due to shutdown.")
                break
        self.node.get_logger().info("cmd_vel publishing loop stopped.")

    def _revert_twist_to_zero(self):
        """Callback for the timer to set current_command_twist to zero."""
        with self.command_twist_lock:
            self.current_command_twist = Twist() # Reset to zero
            self.node.get_logger().info("Movement duration elapsed. Target twist reverted to zero.")
        if self.movement_revert_timer:
             # It's good practice to clear the reference after the timer has fired and done its job.
            self.movement_revert_timer = None 

    def get_frame_rgb(self) -> cv2.typing.MatLike | None:
        """
        Returns the current camera frame in BGR format (OpenCV default).
        Returns None if no frame has been received yet.
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy() # Return a copy
            return None

    def get_frame_thermal(self) -> cv2.typing.MatLike | None:
        """
        Returns the current thermal camera frame.
        Returns None if no thermal frame has been received yet.
        """
        with self.thermal_frame_lock:
            if self.current_thermal_frame is not None:
                return self.current_thermal_frame.copy() # Return a copy
            return None

    def advance(self, axis: Point2d, distance: float, rotation_angle_deg: float = 0.0, *, wait_for_completion: bool = False) -> bool:
        base_desired_linear_speed = 0.1  # m/s
        base_desired_angular_speed_dps = 30.0 # degrees/s
        speed_correction_factor = 1.1
        angular_speed_correction_factor = 1.2
        min_duration_seconds = 0.5

        if distance < 0:
            self.node.get_logger().error("Distance cannot be negative.")
            return False
        
        ax_x = float(axis.x)
        ax_y = float(axis.y)
        linear_axis_magnitude = math.sqrt(ax_x**2 + ax_y**2)

        # Handle explicit stop or no-op command
        if distance == 0 and rotation_angle_deg == 0:
            with self.command_twist_lock:
                if self.movement_revert_timer and self.movement_revert_timer.is_alive():
                    self.movement_revert_timer.cancel()
                    self.node.get_logger().info("Cancelled ongoing movement timer due to zero movement command.")
                self.current_command_twist = Twist() # Set to zero
                self.movement_revert_timer = None
            self.node.get_logger().info("Zero movement command received. Target twist set to zero.")
            return True

        vel_x, vel_y, angular_vel_rps = 0.0, 0.0, 0.0
        duration_seconds = 0.0
        log_parts = []

        # --- Determine duration and velocities ---
        if distance > 0:
            if linear_axis_magnitude == 0:
                self.node.get_logger().error("Linear movement distance > 0 but axis magnitude is zero.")
                return False
            
            duration_seconds = distance / base_desired_linear_speed + min_duration_seconds
            log_parts.append(f"linear_dist={distance:.2f}m")
            
            commanded_linear_speed = base_desired_linear_speed * speed_correction_factor
            norm_x = ax_x / linear_axis_magnitude
            norm_y = ax_y / linear_axis_magnitude
            vel_x = commanded_linear_speed * norm_x
            vel_y = commanded_linear_speed * norm_y

            if rotation_angle_deg != 0:
                # Concurrent rotation: angular speed derived to complete angle in linear movement's duration
                effective_angular_speed_dps = rotation_angle_deg / duration_seconds
                commanded_angular_speed_dps = effective_angular_speed_dps * angular_speed_correction_factor
                angular_vel_rps = math.radians(commanded_angular_speed_dps)
                log_parts.append(f"rot_angle={rotation_angle_deg:.1f}deg (concurrent)")
        
        elif rotation_angle_deg != 0: # Pure rotation (distance is 0)
            duration_seconds = abs(rotation_angle_deg) / base_desired_angular_speed_dps + min_duration_seconds
            log_parts.append(f"rot_angle={rotation_angle_deg:.1f}deg (pure)")
            
            # Angular speed maintains direction of rotation_angle_deg
            effective_angular_speed_dps = base_desired_angular_speed_dps * math.copysign(1.0, rotation_angle_deg)
            commanded_angular_speed_dps = effective_angular_speed_dps * angular_speed_correction_factor
            angular_vel_rps = math.radians(commanded_angular_speed_dps)
            # vel_x, vel_y remain 0.0 as initialized

        if duration_seconds <= 0:
            # This case should ideally be caught by the (distance == 0 and rotation_angle_deg == 0) check.
            # If reached, it implies no effective movement. Log and ensure twist is zero.
            self.node.get_logger().warning(f"Calculated duration is {duration_seconds:.2f}s. Ensuring stop.")
            with self.command_twist_lock:
                if self.movement_revert_timer and self.movement_revert_timer.is_alive():
                    self.movement_revert_timer.cancel()
                self.current_command_twist = Twist()
                self.movement_revert_timer = None
            return True # Effectively a stop

        movement_twist = Twist()
        movement_twist.linear.x = vel_x
        movement_twist.linear.y = vel_y
        movement_twist.angular.z = angular_vel_rps

        log_message = f"Advance called: {', '.join(log_parts)}, calculated_duration={duration_seconds:.2f}s"
        self.node.get_logger().info(log_message)

        with self.command_twist_lock:
            if self.movement_revert_timer and self.movement_revert_timer.is_alive():
                self.movement_revert_timer.cancel() 
                self.node.get_logger().info("Cancelled previous movement timer.")
            
            self.current_command_twist = movement_twist
            self.node.get_logger().info(f"Target twist set to L.x={vel_x:.2f}, L.y={vel_y:.2f}, A.z={angular_vel_rps:.2f}rad/s")

            self.movement_revert_timer = threading.Timer(duration_seconds, self._revert_twist_to_zero)
            self.movement_revert_timer.daemon = True 
            self.movement_revert_timer.start()

        if wait_for_completion:
            self.node.get_logger().info(f"Waiting for movement completion ({duration_seconds:.2f}s)...")
            if self.movement_revert_timer: 
                self.movement_revert_timer.join() 
            self.node.get_logger().info("Movement completion wait finished.")
        
        return True

    def shutdown(self):
        """
        Shuts down the ROS2 node and cleans up resources.
        """
        self.node.get_logger().info("Shutting down Omnibot node...")
        self.shutdown_event.set() # Signal cmd_vel loop to stop

        if self.movement_revert_timer and self.movement_revert_timer.is_alive():
            self.node.get_logger().info("Cancelling active movement timer during shutdown...")
            self.movement_revert_timer.cancel()
            self.movement_revert_timer.join() # Wait for cancel to be processed / thread to end if mid-callback
            self.movement_revert_timer = None

        with self.command_twist_lock:
            self.current_command_twist = Twist() # Ensure target is zero

        # Publish one final zero command immediately before stopping publisher thread
        final_stop_cmd = TwistStamped()
        final_stop_cmd.header.stamp = self.node.get_clock().now().to_msg()
        final_stop_cmd.twist = Twist() # Zero twist
        if self.cmd_vel_publisher.get_subscription_count() > 0: # Check if anyone is listening
             self.node.get_logger().info("Publishing final zero TwistStamped before shutdown.")
             self.cmd_vel_publisher.publish(final_stop_cmd)
             time.sleep(0.1) # Give a very short moment for publish to go through

        if self.cmd_vel_publisher_thread.is_alive():
            self.node.get_logger().info("Waiting for cmd_vel publisher thread to join...")
            self.cmd_vel_publisher_thread.join(timeout=2.0)
            if self.cmd_vel_publisher_thread.is_alive():
                 self.node.get_logger().warning("cmd_vel publisher thread did not join in time.")

        if self.node and rclpy.ok(): # Check if node still exists and rclpy is initialized
            try:
                self.node.destroy_node()
            except Exception as e:
                print(f"Error destroying node: {e}") # Use print if logger is invalid
        
        if self._ros_initialized_by_me and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"Error shutting down rclpy: {e}")
        print("Omnibot node shutdown process complete.")

    def __del__(self):
        """
        Destructor to attempt cleanup if shutdown isn't explicitly called.
        """
        self.shutdown()

if __name__ == '__main__':
    # Example usage:
    print("Creating Omnibot instance...")
    omnibot_instance = Omnibot()
    print("Omnibot instance created. Waiting for thermal frame...")

    # rgb_frame = None # Removed
    thermal_frame = None
    max_wait_time_seconds = 10  # Maximum time to wait for a frame
    check_interval_seconds = 0.1 # How often to check for a frame
    waited_time = 0
    # display_rgb = False # Removed
    display_thermal = False

    try:
        # print("Waiting for thermal frame...")
        # # Loop until thermal frame is received or timeout
        # while thermal_frame is None and waited_time < max_wait_time_seconds:
        #     thermal_frame = omnibot_instance.get_frame_thermal()
        #     if thermal_frame is not None:
        #         print(f"Thermal Frame received: Shape {thermal_frame.shape}")
        #         cv2.imshow("Omnibot Thermal Frame", thermal_frame)
        #         display_thermal = True
        #         print("Displaying thermal frame. Press 'q' in the OpenCV window or Ctrl+C in terminal to continue/close.")
                
        #         key_pressed_in_window = False
        #         while True: 
        #             key = cv2.waitKey(100) 
        #             if key != -1:
        #                 print("Key pressed in OpenCV window.")
        #                 if key == ord('q'): # Quit if 'q' is pressed
        #                     key_pressed_in_window = True
        #                     break
                    
        #             if display_thermal and cv2.getWindowProperty("Omnibot Thermal Frame", cv2.WND_PROP_VISIBLE) < 1:
        #                 print("Thermal OpenCV window was closed.")
        #                 key_pressed_in_window = True 
        #                 break
        #             if key_pressed_in_window: break # Exit if q pressed or window closed
        #         break 
            
        #     if not display_thermal: 
        #         time.sleep(check_interval_seconds)
        #         waited_time += check_interval_seconds
        
        # if thermal_frame is None:
        #     print(f"No Thermal frame received within {max_wait_time_seconds} seconds.")

        # Example of using the advance method
        print("\n--- Testing advance method ---")
        # Ensure node is active before trying to use advance
        # A short sleep might be needed for the node and publisher to fully initialize after Omnibot()
        # time.sleep(0.5) 
        if omnibot_instance.node.executor._is_shutdown:
            print("ROS Node is shutdown, cannot test advance.")
        else:
            # print("\n1. Advancing forward (axis x=1, y=0) by 0.5m, non-blocking...")
            # success = omnibot_instance.advance(Point2d(x=1, y=0), distance=0.5, wait_for_completion=False)
            # print(f"Advance non-blocking linear initiated: {success}")
            # if success:
            #     print("Waiting for 2 seconds to observe non-blocking movement (total duration should be ~5s)...")
            #     time.sleep(2) # Partial wait

            # print("\n2. Pure rotation: 90 degrees, blocking...")
            # success = omnibot_instance.advance(Point2d(x=0, y=0), distance=0, rotation_angle_deg=90.0, wait_for_completion=True)
            # print(f"Advance blocking pure rotation (90deg) completed: {success}")
            # time.sleep(1) # Pause

            # print("\n3. Combined: Forward 0.3m and Rotate -45 degrees, blocking...")
            # success = omnibot_instance.advance(Point2d(x=1, y=0), distance=0.3, rotation_angle_deg=-45.0, wait_for_completion=True)
            # print(f"Advance blocking combined (0.3m, -45deg) completed: {success}")
            # time.sleep(1) # Pause

            # print("\n4. Explicit stop command (distance 0, rotation 0)...")
            # success = omnibot_instance.advance(Point2d(x=0,y=0), distance=0, rotation_angle_deg=0, wait_for_completion=True)
            # print(f"Advance explicit stop: {success}")
            # time.sleep(1)

            print("\n5. Advancing backward (axis x=-1, y=0) by 0.2m, blocking...")
            success = omnibot_instance.advance(Point2d(x=0, y=-1), distance=0.84, wait_for_completion=True)
            print(f"Advance blocking backward (0.2m) completed: {success}")

            # # Original tests (can be re-enabled if needed)
            # print("\nAdvancing diagonally (axis x=1, y=1) by 0.1m, blocking...")
            # success = omnibot_instance.advance(Point2d(x=1, y=1), distance=0.1, wait_for_completion=True)
            # print(f"Advance blocking completed: {success}")
            # print("Movement should be finished.")

            # print("\nAttempting to advance 0 distance...")
            # success = omnibot_instance.advance(Point2d(x=1, y=0), distance=0.0)
            # print(f"Advance with 0 distance: {success}")

            # print("\nAttempting to advance with 0 magnitude axis and non-zero distance...")
            # success = omnibot_instance.advance(Point2d(x=0, y=0), distance=0.1)
            # print(f"Advance with 0-axis, non-zero distance: {success}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
            print("Shutting down Omnibot instance...")
            omnibot_instance.shutdown()
            cv2.destroyAllWindows() # Ensure OpenCV windows are closed
            print("Omnibot example finished.")