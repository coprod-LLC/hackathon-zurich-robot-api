import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import os
from datetime import datetime
import cv2 # Import OpenCV
from cv_bridge import CvBridge, CvBridgeError # Import CvBridge

class CameraSubscriberNode(Node):

    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            10) # QoS depth
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Camera subscriber node started and listening to /camera/image_raw/compressed...')

        # Initialize CvBridge
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('Received a compressed image message!')
        try:
            # Convert ROS CompressedImage message to OpenCV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the image
            cv2.imshow("Camera Feed", cv_image)
            cv2.waitKey(1) # Necessary to update the window and process events

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriberNode()
    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        camera_subscriber.destroy_node()
        cv2.destroyAllWindows() # Close OpenCV windows
        rclpy.shutdown()

if __name__ == '__main__':
    main() 