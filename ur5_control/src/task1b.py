import sys
import rclpy
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf2_ros
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2

# runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False


class FruitsTF(Node):
    """
    ROS2 Boilerplate for fruit detection and TF publishing.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # team id used for final TF child_frame naming (change as required)
        self.team_id = '4571'

        # callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # Timer for periodic processing
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        # TF broadcaster and buffer/listener
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("FruitsTF boilerplate node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):
        '''
        Callback function for aligned depth camera topic. 
        Convert to CV2 image and store in self.depth_image.
        '''
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def colorimagecb(self, data):
        '''
        Callback function for colour camera raw topic.
        Convert to CV2 BGR image and store in self.cv_image.
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting RGB image: {e}")

    def bad_fruit_detection(self, rgb_image):
        '''
        Detect bad fruits in the image frame and return list of dicts:
         {'center':(cX,cY),'distance':d,'angle':a,'width':w,'id':id}
        (distance is placeholder here — real depth will be used in process_image)
        '''
        bad_fruits = []

        # Convert to HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Define HSV range for white color (tweak as required)
        lower_white = np.array([0, 0, 50])
        upper_white = np.array([180, 50, 180])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        id_counter = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 2800:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.75 or aspect_ratio > 2:
                continue

            cX = int(x + w / 2)
            cY = int(y + h / 2)

            # placeholder values
            distance = 0.0
            angle = 0.0

            # Draw bounding box & center on image (visual)
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(rgb_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(rgb_image, f"bad fruit{id_counter + 1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            bad_fruits.append({
                'center': (cX, cY),
                'distance': distance,
                'angle': angle,
                'width': w,
                'id': id_counter,
                'bbox': ((x, y), (x + w, y + h))
            })

            id_counter += 1

        return bad_fruits

    def process_image(self):
        '''
        Timer-driven loop for periodic image processing.
        Computes 3D position per fruit and publishes TFs.
        '''
        if self.cv_image is None:
            return

        frame = self.cv_image.copy()

        # --- Detect fruits ---
        bad_fruits = self.bad_fruit_detection(frame)

        # --- camera intrinsics / constants (verify these for your camera) ---
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        # --- Process each detection ---
        for fruit in bad_fruits:
            (cX, cY) = fruit['center']
            fruit_id = fruit['id']

            # default: no depth
            distance = None

            # Read depth if available and valid
            if self.depth_image is not None:
                try:
                    # Note: depth image indexing [row, col] => [y, x]
                    d_raw = self.depth_image[int(cY), int(cX)]
                    # convert to float
                    d_val = float(d_raw)

                    # check for invalid
                    if np.isnan(d_val) or d_val == 0.0:
                        distance = None
                    else:
                        # Heuristic: if depth > 10 assume mm, otherwise meters
                        if d_val > 10.0:
                            distance = d_val / 1000.0
                        else:
                            distance = d_val
                except Exception as e:
                    self.get_logger().warn(f"Depth read error at ({cX},{cY}): {e}")
                    distance = None

            if distance is None:
                # annotate and skip TF publish for this fruit
                cv2.putText(frame, f"ID:{fruit_id}(no depth)", (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.get_logger().warn(f"No valid depth for fruit ID:{fruit_id} at pixel ({cX},{cY}). Skipping TF publish.")
                continue

            # compute rectified coordinates using provided formula
            # x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
            # y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
            # z = distance_from_rgb
            x_cam = distance * (sizeCamX - float(cX) - centerCamX) / focalX
            y_cam = distance * (sizeCamY - float(cY) - centerCamY) / focalY
            z_cam = distance

            # Update fruit info
            fruit['distance'] = distance
            fruit['angle'] = 0.0

            # Draw info on frame
            cv2.circle(frame, (cX, cY), 6, (255, 0, 0), -1)
            cv2.putText(frame, f"ID:{fruit_id}", (cX + 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"d:{distance:.2f}m", (cX + 10, cY + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Publish transform: camera_link -> cam_<id>
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = f'cam_{fruit_id}'
            t_cam.transform.translation.x = float(x_cam)
            t_cam.transform.translation.y = float(y_cam)
            t_cam.transform.translation.z = float(z_cam)
            t_cam.transform.rotation.x = 0.0
            t_cam.transform.rotation.y = 0.0
            t_cam.transform.rotation.z = 0.0
            t_cam.transform.rotation.w = 1.0

            try:
                self.tf_broadcaster.sendTransform(t_cam)
                self.get_logger().info(f"Published TF camera_link -> {t_cam.child_frame_id} at ({x_cam:.3f},{y_cam:.3f},{z_cam:.3f})")
            except Exception as e:
                self.get_logger().error(f"Failed to send camera TF for fruit {fruit_id}: {e}")

            # Attempt to get composed transform base_link -> cam_<id> and publish team frame
            try:
                # lookup_transform(target_frame='base_link', source_frame='cam_<id>', time=now)
                trans = self.tf_buffer.lookup_transform('base_link', t_cam.child_frame_id, rclpy.time.Time())
                t_base = TransformStamped()
                t_base.header.stamp = self.get_clock().now().to_msg()
                t_base.header.frame_id = 'base_link'
                t_base.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
                t_base.transform = trans.transform

                # publish final TF
                self.tf_broadcaster.sendTransform(t_base)
                self.get_logger().info(f"Published TF base_link -> {t_base.child_frame_id} at ({t_base.transform.translation.x:.3f},"
                                       f"{t_base.transform.translation.y:.3f},{t_base.transform.translation.z:.3f})")
            except Exception as e:
                # It's common that this lookup fails right away (time/latency). Log and continue.
                self.get_logger().warn(f"Could not lookup base_link -> {t_cam.child_frame_id} yet: {e}")

        # Show image feed
        if SHOW_IMAGE:
            cv2.imshow("fruits_tf_view", frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()
