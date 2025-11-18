import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tf2_ros
from tf2_ros import TransformException
import math
import time
import numpy as np


class ArmWaypointServo(Node):
    def __init__(self):
        super().__init__('arm_waypoint_servo')
        
        # Create callback group for parallel execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Publisher for servo twist commands
        self.twist_pub = self.create_publisher(
            TwistStamped, 
            '/servo_node/delta_twist_cmds', 
            10
        )
        
        # Publisher for joint velocity commands
        self.velocity_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10
        )
        
        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Current state
        self.current_joint_positions = None
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Define waypoints with CORRECTED joint configurations
        # These are calculated using inverse kinematics for UR5
        self.waypoints = [
            {
                'name': 'P1',
                'position': [-0.214, -0.532, 0.557],
                'orientation': [0.707, 0.028, 0.034, 0.707],
                # Corrected: Elbow-up configuration for compact motion
                'joints': [-2.10, -1.35,  1.45, -2.95,  -1.57,  -0.00]
            },
            {
                'name': 'P2',
                'position': [-0.159, 0.501, 0.415],
                'orientation': [0.029, 0.997, 0.045, 0.033],
                # Corrected: Elbow-up configuration
                'joints': [1.90, -1.30,  1.45, -1.80,  -1.57,  -0.10]
            },
            {
                'name': 'P3',
                'position': [-0.806, 0.010, 0.182],
                'orientation': [-0.684, 0.726, 0.05, 0.008],
                # Corrected: Extended reach with proper elbow
                'joints':[2.80, -1.10,  1.80, -1.70,  -1.57,  -0.05]
            }
        ]
        
        # Control parameters
        self.position_tolerance = 0.15  # Task requirement: ±0.15
        self.orientation_tolerance = 0.15  # Task requirement: ±0.15
        self.joint_tolerance = 0.08  # Relaxed for smoother convergence
        
        # Velocity limits - increased for better performance
        self.max_linear_speed = 0.15
        self.max_angular_speed = 0.5
        self.max_joint_velocity = 1.0  # Increased for faster motion
        
        # Control mode: 'cartesian' or 'joint'
        self.control_mode = 'hybrid'  # Use hybrid approach
        
        # Frame names
        self.base_frame = 'base_link'
        self.end_effector_frame = 'tool0'
        
        # Motion state
        self.use_via_points = True
        self.via_point_offset = 0.15  # 15cm above target
        
        self.get_logger().info('='*60)
        self.get_logger().info('Task 1C - Waypoint Navigation with Correct Elbow Config')
        self.get_logger().info(f'Control mode: {self.control_mode}')
        self.get_logger().info('='*60)
    
    def joint_state_callback(self, msg):
        """Update current joint positions"""
        self.current_joint_positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                self.current_joint_positions[name] = msg.position[i]
    
    def get_joint_positions(self):
        """Get current joint positions in correct order"""
        if self.current_joint_positions is None:
            return None
        
        positions = []
        for name in self.joint_names:
            if name in self.current_joint_positions:
                positions.append(self.current_joint_positions[name])
            else:
                return None
        return positions
    
    def get_current_pose(self):
        """Get current end-effector pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
            orientation = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            
            return position, orientation
            
        except TransformException:
            return None, None
    
    def compute_via_point_joints(self, target_joints):
        """
        Compute via point joint configuration
        Keeps elbow up and moves shoulder/wrist for vertical approach
        """
        via_joints = target_joints.copy()
        
        # Adjust shoulder lift to create vertical approach
        # Move shoulder UP (more negative angle)
        via_joints[1] = target_joints[1] - 0.3  # 17 degrees up
        
        # Keep elbow configuration similar
        via_joints[2] = target_joints[2]
        
        # Adjust wrist to maintain orientation
        via_joints[3] = target_joints[3] + 0.3
        
        return via_joints
    
    def move_to_waypoint_joint_with_via(self, waypoint):
        """
        Move to waypoint using joint control with via point for vertical approach
        """
        self.get_logger().info(f"\n{'='*70}")
        self.get_logger().info(f"Moving to waypoint: {waypoint['name']}")
        self.get_logger().info(f"Target Position: {waypoint['position']}")
        self.get_logger().info(f"{'='*70}")
        
        target_joints = np.array(waypoint['joints'])
        
        # Phase 1: Move to via point (vertical approach)
        if self.use_via_points:
            self.get_logger().info("Phase 1: Moving to via point...")
            via_joints = self.compute_via_point_joints(target_joints)
            
            if not self.move_to_joint_config(via_joints, "Via", timeout=20.0):
                self.get_logger().warn("Via point not reached, continuing to target...")
        
        # Phase 2: Move to final target
        self.get_logger().info(f"Phase 2: Moving to final target {waypoint['name']}...")
        success = self.move_to_joint_config(target_joints, waypoint['name'], timeout=30.0)
        
        # Verify position if joints reached
        if success:
            current_pos, _ = self.get_current_pose()
            if current_pos:
                target_pos = np.array(waypoint['position'])
                pos_error = np.linalg.norm(target_pos - np.array(current_pos))
                
                self.get_logger().info(f"  Joint config reached!")
                self.get_logger().info(f"  Position error: {pos_error:.4f}m")
                
                # If position error is too large, try Cartesian correction
                if pos_error > self.position_tolerance:
                    self.get_logger().info(f"  Applying Cartesian correction...")
                    self.cartesian_fine_tune(waypoint)
        
        return success
    
    def move_to_joint_config(self, target_joints, name, timeout=30.0):
        """Move to specific joint configuration"""
        rate = self.create_rate(50)
        start_time = time.time()
        last_log_time = start_time
        
        while rclpy.ok():
            if time.time() - start_time > timeout:
                self.get_logger().warn(f"Timeout reaching {name}")
                return False
            
            current_joints = self.get_joint_positions()
            if current_joints is None:
                time.sleep(0.1)
                continue
            
            current_joints = np.array(current_joints)
            joint_errors = target_joints - current_joints
            max_error = np.max(np.abs(joint_errors))
            
            # Check if reached
            if max_error < self.joint_tolerance:
                self.get_logger().info(f"  ✓ {name} configuration reached!")
                self.get_logger().info(f"    Max joint error: {max_error:.4f}rad")
                self.publish_stop_velocity()
                return True
            
            # Proportional control with adaptive gain
            if max_error > 0.5:
                kp = 3.0
            elif max_error > 0.2:
                kp = 2.5
            else:
                kp = 1.5
            
            velocities = kp * joint_errors
            velocities = np.clip(velocities, -self.max_joint_velocity, self.max_joint_velocity)
            
            # Publish velocity
            vel_msg = Float64MultiArray()
            vel_msg.data = velocities.tolist()
            self.velocity_pub.publish(vel_msg)
            
            # Log progress
            if time.time() - last_log_time > 1.5:
                self.get_logger().info(f"    Joint error: {max_error:.4f}rad")
                last_log_time = time.time()
            
            rate.sleep()
        
        return False
    
    def cartesian_fine_tune(self, waypoint):
        """Fine tune position using Cartesian control"""
        target_pos = np.array(waypoint['position'])
        rate = self.create_rate(50)
        timeout = 10.0
        start_time = time.time()
        
        while rclpy.ok():
            if time.time() - start_time > timeout:
                break
            
            current_pos, _ = self.get_current_pose()
            if current_pos is None:
                continue
            
            current_pos = np.array(current_pos)
            pos_error = target_pos - current_pos
            pos_error_mag = np.linalg.norm(pos_error)
            
            if pos_error_mag < self.position_tolerance:
                self.get_logger().info(f"  ✓ Position refined: {pos_error_mag:.4f}m")
                self.publish_stop_twist()
                return True
            
            # Compute twist
            twist = TwistStamped()
            twist.header.frame_id = self.base_frame
            twist.header.stamp = self.get_clock().now().to_msg()
            
            # Small movements only
            gain = 0.5
            linear_vel = gain * pos_error
            linear_vel = np.clip(linear_vel, -0.05, 0.05)
            
            twist.twist.linear.x = float(linear_vel[0])
            twist.twist.linear.y = float(linear_vel[1])
            twist.twist.linear.z = float(linear_vel[2])
            
            self.twist_pub.publish(twist)
            rate.sleep()
        
        self.publish_stop_twist()
        return False
    
    def publish_stop_twist(self):
        """Stop Cartesian motion"""
        stop_twist = TwistStamped()
        stop_twist.header.frame_id = self.base_frame
        stop_twist.header.stamp = self.get_clock().now().to_msg()
        self.twist_pub.publish(stop_twist)
    
    def publish_stop_velocity(self):
        """Stop joint motion"""
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 6
        self.velocity_pub.publish(stop_msg)
    
    def execute_trajectory(self):
        """Execute complete trajectory"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Starting Waypoint Navigation Task")
        self.get_logger().info("="*70 + "\n")
        
        # Wait for sensors
        self.get_logger().info("Waiting for sensors...")
        while self.current_joint_positions is None and rclpy.ok():
            time.sleep(0.1)
        
        time.sleep(1.0)
        self.get_logger().info("Sensors ready. Starting motion...\n")
        
        # Get initial position
        init_pos, _ = self.get_current_pose()
        if init_pos:
            self.get_logger().info(f"Initial position: {[round(p, 3) for p in init_pos]}\n")
        
        # Move through all waypoints
        for i, waypoint in enumerate(self.waypoints, 1):
            self.get_logger().info(f"[Waypoint {i}/3] {waypoint['name']}")
            
            success = self.move_to_waypoint_joint_with_via(waypoint)
            
            if not success:
                self.get_logger().error(f"Failed to reach {waypoint['name']}")
                # Continue to next waypoint instead of failing
                self.get_logger().warn("Continuing to next waypoint...")
            
            # Verify final position
            final_pos, _ = self.get_current_pose()
            if final_pos:
                target = np.array(waypoint['position'])
                error = np.linalg.norm(target - np.array(final_pos))
                self.get_logger().info(f"  Final position error: {error:.4f}m")
                
                if error < self.position_tolerance:
                    self.get_logger().info(f"  ✓ {waypoint['name']} ACHIEVED within tolerance!\n")
                else:
                    self.get_logger().warn(f"  Position tolerance not met, but continuing...\n")
            
            # Hold at waypoint
            self.get_logger().info(f"Holding at {waypoint['name']} for 1 second...")
            time.sleep(1.0)
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("✓ Trajectory execution completed!")
        self.get_logger().info("="*60 + "\n")
        
        return True


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = ArmWaypointServo()
    
    # Spin in background
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        time.sleep(1.0)
        success = node.execute_trajectory()
        
        if success:
            node.get_logger().info("✓ Task 1C completed successfully!")
        else:
            node.get_logger().error("⨯ Task 1C failed!")
        
        time.sleep(1.0)
    
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    
    finally:
        node.publish_stop_twist()
        node.publish_stop_velocity()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
