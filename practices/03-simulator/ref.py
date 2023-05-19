"""
03-simulator
author: @maximellerbach
"""

import base64
import json
import logging
import os
import time

import cv2
import keyboard
import numpy as np
from sdclient import SDClient

logger = logging.getLogger(__name__)


class ManualClient(SDClient):
    def __init__(self, data_path, host="127.0.0.1", port=9091):
        super().__init__(host, port, poll_socket_sleep_time=0.001)

        self.data_path = data_path
        assert self.data_path is not None

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.running = True
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.telemetry = {}

        self.last_received = time.time()
        self.last_processed = self.last_received

        self.handlers = {
            "telemetry": self.on_telemetry,
        }
        
        self.steering = 0.0
        self.throttle = 0.0

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
        decode the image and store it in the image attribute.
        You can then remove the image from the json_packet.
        store the telemetry data (json_packet without the image) in the telemetry attribute.
        """

        encimg = json_packet["image"]
        image = np.frombuffer(base64.b64decode(encimg), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        self.image = image
        del json_packet["image"]
        self.telemetry = json_packet

        self.last_received = time.time()

    def await_telemetry(self):
        """
        Waits for a telemetry packet to arrive.
        """
        while self.last_received == self.last_processed:
            time.sleep(0.001)

        self.last_processed = self.last_received

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

    def get_manual_controls(self):
        """
        Gets manual controls from the keyboard.
        The changes in steering and throttle should be continuous.
        When no key is pressed, the steering and throttle should change slowly back to 0.   
        """

        steps = 0.2
        decay = 0.5

        max_steering = 0.75
        max_throttle = 0.7

        if keyboard.is_pressed('a'):
            self.steering -= steps
        elif keyboard.is_pressed('d'):
            self.steering += steps
        else:
            self.steering = self.steering * decay
        
        if keyboard.is_pressed('w'):
            self.throttle += steps
        elif keyboard.is_pressed('s'):
            self.throttle -= steps
        else:
            self.throttle = self.throttle * decay

        self.steering = np.clip(self.steering, -max_steering, max_steering)
        self.throttle = np.clip(self.throttle, -max_throttle, max_throttle)

        return self.steering, self.throttle

    def main_loop(self):
        """
        Main loop of the client.

        Waits for telemetry packets and processes them.
        This is also where you can send control commands.

        The goal here is to be able to manually gather data by driving the car
        This can be done by using the keyboard or a gamepad.
        Easier is to use a gamepad, but it's harder to implement.

        You can imagine driving with WASD, and recording only when pressing the spacebar.
        """
        while self.running:
            self.await_telemetry()
            steering, throttle = self.get_manual_controls()

            # print(steering, throttle)

            # recording
            if keyboard.is_pressed('space'):
                labels = self.telemetry.copy()
                labels["steering"] = steering
                labels["throttle"] = throttle

                # save image
                recording_time = time.time()
                image_path = os.path.join(
                    self.data_path, f"{recording_time}.png")
                image = cv2.resize(self.image, (160, 120))
                cv2.imwrite(image_path, image)

                # save labels
                labels_path = os.path.join(
                    self.data_path, f"{recording_time}.json")
                with open(labels_path, 'w') as f:
                    json.dump(labels, f)

            # manual controls
            self.update(steering, throttle)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data')
    logging.info(data_path)

    client = ManualClient(data_path)
    time.sleep(2)
    client.main_loop()
