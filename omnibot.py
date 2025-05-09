from __future__ import annotations
from dataclasses import dataclass
import cv2
# import cv2.typing # Ensure this is uncommented
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu # Import Imu message
from geometry_msgs.msg import TwistStamped, Twist, Quaternion # Import Quaternion for type hint
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

        self.current_orientation_quat: Quaternion | None = None
        self.orientation_lock = threading.Lock()
        self.initial_yaw_deg: float | None = None # To store the initial yaw offset
        self.initial_orientation_lock = threading.Lock() # Lock for setting initial_yaw_deg

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

        self.imu_subscription = self.node.create_subscription(
            Imu,
            '/imu/data',
            self._imu_callback,
            10) # QoS profile for IMU data

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

    def _imu_callback(self, msg: Imu):
        """Callback for the IMU data. Stores the orientation quaternion and sets initial yaw offset."""
        with self.orientation_lock:
            self.current_orientation_quat = msg.orientation
        
        # Set initial orientation offset on first message
        with self.initial_orientation_lock:
            if self.initial_yaw_deg is None:
                # Calculate yaw from this first message
                q = msg.orientation
                t0 = +2.0 * (q.w * q.z + q.x * q.y)
                t1 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                initial_yaw_rad = math.atan2(t0, t1)
                self.initial_yaw_deg = math.degrees(initial_yaw_rad)
                self.node.get_logger().info(f"Initial orientation captured. Yaw offset set to: {self.initial_yaw_deg:.2f} degrees.")

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

    def get_orientation(self) -> float | None:
        """
        Computes the robot's current orientation (yaw) in degrees from IMU data.
        Returns None if no IMU data has been received yet.
        """
        q: Quaternion | None = None
        with self.orientation_lock:
            if self.current_orientation_quat is not None:
                # Create a copy to work with, to release lock faster
                q = Quaternion(
                    x=self.current_orientation_quat.x,
                    y=self.current_orientation_quat.y,
                    z=self.current_orientation_quat.z,
                    w=self.current_orientation_quat.w
                )
        
        if q is None:
            # self.node.get_logger().debug("No orientation data available yet.")
            return None

        # Calculate current absolute yaw
        t0 = +2.0 * (q.w * q.z + q.x * q.y)
        t1 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        current_absolute_yaw_rad = math.atan2(t0, t1)
        current_absolute_yaw_deg = math.degrees(current_absolute_yaw_rad)

        # Apply offset if initial orientation has been set
        with self.initial_orientation_lock: # Protect read of initial_yaw_deg in case it's being set
            if self.initial_yaw_deg is not None:
                relative_yaw_deg = current_absolute_yaw_deg - self.initial_yaw_deg
                # Normalize to [-180, 180)
                relative_yaw_deg = (relative_yaw_deg + 180.0) % 360.0 - 180.0
                return relative_yaw_deg
            else:
                # Initial orientation not yet captured, return None or absolute based on desired behavior
                # Returning None until zeroed is consistent with the idea of "get_orientation" after zeroing
                self.node.get_logger().debug("Initial orientation not yet captured. Cannot provide relative orientation.")
                return None 

    def move(self, axis: Point2d, distance: float, rotation_angle_deg: float = 0.0, *, wait_for_completion: bool = False) -> bool:
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

    def advance(self, axis: Point2d, distance: float, *, wait_for_completion: bool = False) -> bool:
        return self.move(axis=axis, distance=distance, wait_for_completion=wait_for_completion)

    def rotate(self, rotation_angle_deg: float, *, wait_for_completion: bool = False) -> bool:
        return self.move(axis=Point2d(x=0, y=0), distance=0, rotation_angle_deg=rotation_angle_deg, wait_for_completion=wait_for_completion)

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
        if omnibot_instance.node.executor._is_shutdown:
            print("ROS Node is shutdown, cannot test advance.")
        else:
            # print("Testing advance method...")
            # omnibot_instance.(axis=Point2d(x=1, y=0), distance=0.5, wait_for_completion=True)
            # print("Advance method tested successfully.")
            print("Testing rotate method...")
            omnibot_instance.rotate(rotation_angle_deg=45.0, wait_for_completion=True)
            print("Rotate method tested successfully.")

        # Test IMU Orientation - should now be relative after first few readings
        print("\n--- Testing get_orientation (zeroed at startup) ---")
        print("Waiting a moment for initial IMU data to establish zero orientation...")
        time.sleep(1.0) # Give a little time for the first IMU message to arrive

        for i in range(15): # Check for orientation 15 times over 3 seconds
            orientation = omnibot_instance.get_orientation()
            if orientation is not None:
                print(f"Current zeroed orientation (yaw): {orientation:.2f} degrees")
            else:
                print("Waiting for IMU data / initial orientation to be set...")
            time.sleep(0.2)
        
        # Existing __main__ tests for movement, adjusted to use move/rotate methods
        # print("\n--- Testing movement methods ---")
        # if hasattr(omnibot_instance.node.executor, '_is_shutdown') and omnibot_instance.node.executor._is_shutdown:
        #     print("ROS Node is shutdown, cannot test movement.")
        # elif omnibot_instance.node.executor is None:
        #     print("ROS Node executor not available, cannot test movement.")
        # else:
        #     print("\n1. Moving forward by 0.3m, blocking...")
        #     success = omnibot_instance.move(Point2d(x=1, y=0), distance=0.3, wait_for_completion=True)
        #     print(f"Move forward completed: {success}")
        #     time.sleep(1)

        #     print("\n2. Rotating +90 degrees, blocking...")
        #     success = omnibot_instance.rotate(rotation_angle_deg=90.0, wait_for_completion=True)
        #     print(f"Rotate +90deg completed: {success}")
        #     time.sleep(1)

        #     print("\n3. Moving forward 0.2m and Rotating -45 degrees, blocking...")
        #     success = omnibot_instance.move(Point2d(x=1, y=0), distance=0.2, rotation_angle_deg=-45.0, wait_for_completion=True)
        #     print(f"Move combined completed: {success}")
        #     time.sleep(1)

        #     print("\n4. Explicit stop command...")
        #     success = omnibot_instance.move(Point2d(x=0,y=0), distance=0, rotation_angle_deg=0, wait_for_completion=True)
        #     print(f"Explicit stop: {success}")
        #     time.sleep(1)

            # # The thermal image display part is commented out as per previous state of __main__ in user's file
            # print("\n--- Testing thermal image retrieval (original example) ---")
            # ... (thermal image code was here)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
            print("Shutting down Omnibot instance...")
            omnibot_instance.shutdown()
            cv2.destroyAllWindows() # Ensure OpenCV windows are closed
            print("Omnibot example finished.")