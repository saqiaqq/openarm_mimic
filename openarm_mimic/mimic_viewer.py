#!/usr/bin/python3
"""
Mimic Viewer Node
=================
笔记本端节点：订阅板端 mimic_vision 发来的带骨架叠加的压缩 RGB 帧，
在 cv2 窗口里显示；鼠标左键点击或按 's' 键切换 /mimic/enable，
按 'q' 退出。

Subscribes:
  - /mimic/vision/annotated/compressed  (sensor_msgs/CompressedImage)

Publishes:
  - /mimic/enable                        (std_msgs/Bool)

参数:
  - annotated_topic (string): 订阅的压缩图话题名
  - enable_topic    (string): 启停布尔话题名
  - window_name     (string): cv2 窗口标题
"""

import sys
sys.path = [p for p in sys.path if 'miniconda' not in p and 'anaconda' not in p]

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class MimicViewerNode(Node):
    def __init__(self):
        super().__init__('mimic_viewer')

        self.declare_parameter('annotated_topic', '/mimic/vision/annotated/compressed')
        self.declare_parameter('enable_topic', '/mimic/enable')
        self.declare_parameter('window_name', 'Mimic Viewer')

        self.annotated_topic = self.get_parameter('annotated_topic').value
        self.enable_topic = self.get_parameter('enable_topic').value
        self.window_name = self.get_parameter('window_name').value

        self.sub_img = self.create_subscription(
            CompressedImage, self.annotated_topic, self._img_cb, 10)
        self.pub_enable = self.create_publisher(Bool, self.enable_topic, 10)

        self._latest = None
        self._active_local = False  # UI 层的意图状态

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        # 30Hz 刷新，不阻塞 rclpy.spin()
        self.timer = self.create_timer(0.033, self._tick)

        self.get_logger().info(
            f"Mimic Viewer started. Subscribing {self.annotated_topic} | "
            f"Publishing {self.enable_topic}"
        )
        self.get_logger().info("LMB / 's' = toggle enable, 'q' = quit")

    def _img_cb(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                self._latest = img
        except Exception as e:
            self.get_logger().error(f"Decode error: {e}")

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._toggle_enable('mouse')

    def _toggle_enable(self, source: str):
        self._active_local = not self._active_local
        msg = Bool()
        msg.data = self._active_local
        self.pub_enable.publish(msg)
        self.get_logger().info(
            f"[{source}] /mimic/enable -> {self._active_local}")

    def _tick(self):
        if self._latest is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank,
                        f"Waiting for {self.annotated_topic} ...",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(self.window_name, blank)
        else:
            disp = self._latest.copy()
            banner = f"LOCAL intent: {'ON' if self._active_local else 'OFF'}  (LMB/s toggle, q quit)"
            cv2.putText(disp, banner, (20, disp.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(self.window_name, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('s'):
            self._toggle_enable('keyboard')


def main(args=None):
    rclpy.init(args=args)
    node = MimicViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
