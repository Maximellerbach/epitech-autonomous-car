"""
03-simulator
"""

import glob
import logging
import os
import time
import json
import base64

import cv2
import numpy as np

from gym_donkeycar.core.client import SDClient

logging.basicConfig(level=logging.INFO)


class Client(SDClient):
    def __init__(self, data_path, host="127.0.0.1", port=9091):
        super().__init__(host, port, poll_socket_sleep_time=0.001)

        self.data_path = data_path
        assert self.data_path is not None

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.running = True

        self.handlers = {
            "telemetry": self.on_telemetry,
        }

    def on_msg_recv(self, json_packet):
        """
        Dispatches the message to the appropriate handler.
        """
        msg_type = json_packet["msg_type"]

        if msg_type in self.handlers:
            self.handlers[msg_type](json_packet)
        else:
            logging.warning("Unknown message type: %s", msg_type)

    def on_telemetry(self, json_packet):
        """
        Receives telemetry data from the simulator.
        """
        encimg = json_packet["image"]
        image = np.frombuffer(base64.b64decode(encimg), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is not None:
            # do some stuff with the image

            cv2.imshow("image", image)
            cv2.waitKey(1)


    def update(self, steering, throttle):
        """
        Sends control commands to the simulator.
        """
        msg = {
            "msg_type": "control",
            "throttle": throttle.__str__(),
            "steering": steering.__str__(),
            "brake": "0.0",
        }

        self.send_now(json.dumps(msg))

    def main_loop(self):
        """
        Main loop of the client.
        """
        while self.running:
            self.update(-1.0, 0.1)
            time.sleep(0.1)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data')
    logging.info(data_path)

    client = Client(data_path)
    time.sleep(2)
    client.main_loop()
