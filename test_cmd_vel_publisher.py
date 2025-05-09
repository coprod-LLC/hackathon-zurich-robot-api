import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Twist
import time

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('simple_cmd_vel_tester')
    publisher = node.create_publisher(TwistStamped, '/omnibot/cmd_vel', 10)

    # Create a TwistStamped message for moving forward
    move_cmd = TwistStamped()
    move_cmd.header.stamp = node.get_clock().now().to_msg()
    # move_cmd.header.frame_id = "base_link"  # Optional: Set if your robot requires it
    move_cmd.twist.linear.x = 0.2  # m/s
    move_cmd.twist.linear.y = 0.0
    move_cmd.twist.linear.z = 0.0
    move_cmd.twist.angular.x = 0.0
    move_cmd.twist.angular.y = 0.0
    move_cmd.twist.angular.z = 0.0

    node.get_logger().info(f'Publishing move command: linear.x = {move_cmd.twist.linear.x:.2f} m/s')
    publisher.publish(move_cmd)

    # Wait for 5 seconds
    node.get_logger().info('Waiting for 5 seconds...')
    time.sleep(5)

    # Create a TwistStamped message for stopping
    stop_cmd = TwistStamped()
    stop_cmd.header.stamp = node.get_clock().now().to_msg()
    # stop_cmd.header.frame_id = "base_link"  # Optional
    # .twist will default to all zeros, which is what we want for stopping
    # stop_cmd.twist = Twist() # Explicitly set to zero twist

    node.get_logger().info('Publishing stop command.')
    publisher.publish(stop_cmd)
    
    # Give a moment for the message to be sent before shutting down
    time.sleep(0.1)

    node.get_logger().info('Shutting down node.')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 