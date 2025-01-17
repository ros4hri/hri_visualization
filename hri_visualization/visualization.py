#!/usr/bin/env python3

# Copyright 2024 PAL Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from hri_msgs.msg import Skeleton2D
from sensor_msgs.msg import Image, CompressedImage
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri import HRIListener

import cv2
from cv_bridge import CvBridge

import numpy as np

from PIL import ImageFont, ImageDraw
from PIL import Image as PILImage

from threading import Lock
import hashlib

from ament_index_python.packages import get_package_share_directory
from pathlib import Path

# Drawing parameters definition
PASTEL_YELLOW = (174, 239, 238)
BLACK = (0, 0, 0)
TEXT_BLACK = (0, 0, 0, 255)
BOX_THICKNESS = 3
THICKNESS_CORNERS = 2
LABEL_DISTANCE = 50
LABEL_WIDTH = 80
LABEL_HEIGHT = 30
SPACE_PER_CHARACTER = 14
LABEL_LINE_THICKNESS = 1
JOINT_RADIUS = 10
FILLED = -1

package_path = Path(get_package_share_directory("hri_visualization"))

FONTS_FOLDER = package_path / "fonts"
FONT = str(FONTS_FOLDER / "Montserrat/Montserrat-SemiBold.ttf")
LARGEST_LETTER = "M"
VIS_CORNER_GAP_RATIO = 0.1
VIS_CORNER_WIDTH_RATIO = 0.1
VIS_CORNER_HEIGHT_RATIO = 0.1
EMOJI_SIZE_MAGIC_NUMBER = 0.0625

# 2D skeleton joints
# related to face keypoints won't be drawn
joints_to_draw = [
    Skeleton2D.NECK,
    Skeleton2D.RIGHT_SHOULDER,
    Skeleton2D.RIGHT_ELBOW,
    Skeleton2D.RIGHT_WRIST,
    Skeleton2D.LEFT_SHOULDER,
    Skeleton2D.LEFT_ELBOW,
    Skeleton2D.LEFT_WRIST,
    Skeleton2D.RIGHT_HIP,
    Skeleton2D.RIGHT_KNEE,
    Skeleton2D.RIGHT_ANKLE,
    Skeleton2D.LEFT_HIP,
    Skeleton2D.LEFT_KNEE,
    Skeleton2D.LEFT_ANKLE,
]

iconic_pokemons = ['Pikachu',
                   'Charizard',
                   'Mewtwo',
                   'Blastoise',
                   'Snorlax',
                   'Bulbasaur',
                   'Gyarados',
                   'Dragonite',
                   'Eevee',
                   'Lapras']

adjectives = ['Vegan',
              'Fiery',
              'Powerful',
              'Bright',
              'Sleepy',
              'Brave',
              'Fierce',
              'Clever',
              'Adaptable',
              'Gentle']


class HRIVisualizer(Node):
    """
    Object handling the display of body features over an image stream.

    A class managing publishing an
    image stream where information
    regarding the detected people
    is displayed.
    """

    class PersonDescriptor:
        """Internal class used to describe person-related aspects."""

        id_to_display = None
        label_width = None
        font = None
        max_distance_corner = None

    def __init__(self):
        super().__init__('hri_visualization')
        self.declare_parameter('funny_names', False)
        self.declare_parameter('compressed_input', True)
        self.declare_parameter('compressed_output', True)
        self.funny_names = self.get_parameter('funny_names').value
        self.compressed_input = self.get_parameter('compressed_input').value
        self.compressed_output = self.get_parameter('compressed_output').value
        self.font = self.calibrate_font_size(5)

        self.hri_listener = HRIListener('hri_listener')

        resolved_topic_name = self.resolve_topic_name("/image")
        if self.compressed_input:
            compressed_image_topic = resolved_topic_name+"/compressed"
            self.img_sub = self.create_subscription(
                CompressedImage,
                compressed_image_topic,
                self.compressed_img_cb,
                qos_profile=qos_profile_sensor_data)
        else:
            self.img_sub = self.create_subscription(
                Image, "/image",
                self.img_cb,
                qos_profile=qos_profile_sensor_data)
        self.hri_overlay_topic = resolved_topic_name + "/hri_overlay"

        if self.compressed_output:
            self.img_pub = self.create_publisher(
                CompressedImage,
                self.hri_overlay_topic + "/compressed",
                qos_profile=qos_profile_sensor_data)
        else:
            self.img_pub = self.create_publisher(
                Image,
                self.hri_overlay_topic,
                qos_profile=qos_profile_sensor_data)

        diag_period = self.declare_parameter("diagnostic_period", 1.0).value
        self.diag_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 1)
        self.diag_timer = self.create_timer(diag_period, self.do_diagnostics)

        self.proc_time_ms = 0

        self.bridge = CvBridge()

        self.faces = {}
        self.bodies = {}
        self.persons = {}

        self.expressions = {}

        self.persons_lock = Lock()

    def skeleton_cb(self, skeleton_msg, args):
        """Store information regarding a body's skeleton coordinates."""
        id = args
        self.bodies[id][1] = skeleton_msg.skeleton

    def generate_funny_ids(self, person_id):
        """
        Generate a funny id.

        A funny id is a combination
        of an adjective and a name from two
        predefined lists.
        """
        person_id_reverse = person_id[::-1]
        hash_value_adjective = hashlib.md5(
            person_id.encode('utf-8')).hexdigest()
        hash_value_name = hashlib.md5(
            person_id_reverse.encode('utf-8')).hexdigest()
        random_adjective_index = int(
            hash_value_adjective, 16) % len(adjectives)
        random_name_index = int(hash_value_name, 16) % len(iconic_pokemons)
        return adjectives[random_adjective_index] \
            + " " \
            + iconic_pokemons[random_name_index]

    def get_expression_image(self, expression):
        """Return the expression-related emoji."""
        if (expression not in self.expressions) or (self.expressions[expression] is None):
            filename = f"{expression.lower()}.png"
            emoji = self.load_image(filename)
            if emoji is not None:
                emoji_size = (int(emoji.shape[1]*EMOJI_SIZE_MAGIC_NUMBER),
                              int(emoji.shape[1]*EMOJI_SIZE_MAGIC_NUMBER))
                emoji = cv2.resize(emoji, emoji_size)
                emoji[:, :, :3] = emoji[:, :, :3] + PASTEL_YELLOW
            self.expressions[expression] = emoji

        return self.expressions[expression]

    def load_image(self, filename):
        image_path = Path(package_path) / 'images' / \
            filename
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"Image file '{filename}' not found.")
            return image
        except Exception as e:
            self.get_logger().error(f"Error loading image: {e}")
            return None

    def compressed_img_cb(self, msg):
        """
        Manage the incoming compressed images.

        Function performing the same operations as
        img_cb, but with compressed images.
        """
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        img_msg.header = msg.header
        self.img_cb(img_msg)

    def img_cb(self, msg):
        """
        Manage the incoming images.

        Function handling the incoming images,
        drawing the bounding boxes for the
        detected faces and the skeleton joints
        for the detected bodies.
        """
        if not self.persons_lock.locked():
            with self.persons_lock:
                start_proc_time = self.get_clock().now()

                # updating the tracked persons
                tracked_persons = self.hri_listener.tracked_persons
                for person in tracked_persons:
                    if person not in self.persons:
                        self.persons[person] = self.PersonDescriptor()
                        if self.funny_names:
                            id_displayed = self.generate_funny_ids(person)
                        else:
                            id_displayed = person
                        self.persons[person].id_to_display = id_displayed
                        self.persons[person].label_width = SPACE_PER_CHARACTER * \
                            len(id_displayed)
                        self.persons[person].font = self.calibrate_font_size(
                            len(id_displayed))

                for person in list(self.persons.keys()):
                    if person not in tracked_persons:
                        del self.persons[person]

                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                (height, width, _) = img.shape
                for person in list(self.persons):
                    face = tracked_persons[person].face
                    if face and (roi := face.roi):
                        # Label sizing calibration
                        label_width = self.persons[person].label_width
                        font = self.persons[person].font
                        face_x = int(roi[0] * width)
                        face_y = int(roi[1] * height)
                        face_width = int(roi[2] * width)
                        face_height = int(roi[3] * height)

                        starting_point = (
                            face_x,
                            face_y
                        )
                        ending_point = (
                            face_x + face_width,
                            face_y + face_height
                        )
                        img = cv2.rectangle(
                            img,
                            starting_point,
                            ending_point,
                            PASTEL_YELLOW,
                            BOX_THICKNESS,
                            lineType=cv2.LINE_AA,
                        )

                        # sizing the height/width of the localisation corners
                        visual_roi_x = face_x + BOX_THICKNESS
                        visual_roi_y = face_y + BOX_THICKNESS
                        visual_roi_width = face_width - (2*BOX_THICKNESS)
                        visual_roi_height = face_height - (2*BOX_THICKNESS)

                        ptsTopLeft = np.array(
                            [
                                [
                                    visual_roi_x
                                    + VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + VIS_CORNER_GAP_RATIO*visual_roi_height
                                    + VIS_CORNER_HEIGHT_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + VIS_CORNER_GAP_RATIO*visual_roi_width
                                    + VIS_CORNER_WIDTH_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                            ],
                            dtype=np.int32,
                        )
                        ptsBottomLeft = np.array(
                            [
                                [
                                    visual_roi_x
                                    + VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + visual_roi_height
                                    - VIS_CORNER_GAP_RATIO*visual_roi_height
                                    - VIS_CORNER_HEIGHT_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + visual_roi_height
                                    - VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + VIS_CORNER_GAP_RATIO*visual_roi_width
                                    + VIS_CORNER_WIDTH_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + visual_roi_height
                                    - VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                            ],
                            dtype=np.int32,
                        )
                        ptsTopRight = np.array(
                            [
                                [
                                    visual_roi_x
                                    + visual_roi_width
                                    - VIS_CORNER_GAP_RATIO*visual_roi_width
                                    - VIS_CORNER_WIDTH_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + visual_roi_width
                                    - VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + visual_roi_width
                                    - VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + VIS_CORNER_GAP_RATIO*visual_roi_height
                                    + VIS_CORNER_HEIGHT_RATIO*visual_roi_height
                                ],
                            ],
                            dtype=np.int32,
                        )
                        ptsBottomRight = np.array(
                            [
                                [
                                    visual_roi_x
                                    + visual_roi_width
                                    - VIS_CORNER_GAP_RATIO*visual_roi_width
                                    - VIS_CORNER_WIDTH_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + visual_roi_height
                                    - VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + visual_roi_width
                                    - VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + visual_roi_height
                                    - VIS_CORNER_GAP_RATIO*visual_roi_height
                                ],
                                [
                                    visual_roi_x
                                    + visual_roi_width
                                    - VIS_CORNER_GAP_RATIO*visual_roi_width,
                                    visual_roi_y
                                    + visual_roi_height
                                    - VIS_CORNER_GAP_RATIO*visual_roi_height
                                    - VIS_CORNER_HEIGHT_RATIO*visual_roi_height
                                ],
                            ],
                            dtype=np.int32,
                        )

                        img = cv2.polylines(
                            img,
                            [ptsTopLeft, ptsBottomLeft,
                                ptsBottomRight, ptsTopRight],
                            isClosed=False,
                            color=PASTEL_YELLOW,
                            thickness=THICKNESS_CORNERS,
                            lineType=cv2.LINE_AA,
                        )

                        # Now inserting the label.
                        # Step 1: checking if a label point already exists.
                        # If this is not the case, selecting the further point
                        # the corners of the image.
                        if not self.persons[person].max_distance_corner:
                            further_corner = self.find_max_distance_corner(
                                face_x + face_width / 2,
                                face_y + face_height / 2,
                                width,
                                height,
                            )
                            self.persons[person].max_distance_corner = [
                                further_corner[0] * width,
                                further_corner[1] * height,
                            ]

                        # Step 2: if the corner has already been computed,
                        # it will remain the same as long as the label would
                        # be outside of the image.
                        roi_corner = np.array(
                            [
                                face_x +
                                (self.persons[person].max_distance_corner[0] /
                                 width * face_width),
                                face_y +
                                (self.persons[person].max_distance_corner[1] /
                                 height * face_height),
                            ]
                        )
                        label_to_corner = np.array(
                            self.persons[person].max_distance_corner) - roi_corner
                        label_to_corner_distance = np.linalg.norm(
                            [roi_corner[0]
                             - self.persons[person].max_distance_corner[0],
                             roi_corner[1]
                             - self.persons[person].max_distance_corner[1]],
                            2)
                        label_corner = (
                            roi_corner
                            + (label_to_corner / label_to_corner_distance)
                            * LABEL_DISTANCE
                        )
                        if (
                            label_to_corner_distance
                            < (LABEL_DISTANCE + np.sqrt(label_width ** 2 + LABEL_HEIGHT ** 2))
                            or (label_corner[0] - label_width < 0)
                            or (label_corner[0] + label_width > width)
                            or (label_corner[1] - LABEL_HEIGHT < 0)
                            or (label_corner[1] + LABEL_HEIGHT > height)
                        ):
                            further_corner = self.find_max_distance_corner(
                                face_x + face_width / 2,
                                face_y + face_height / 2,
                                width,
                                height,
                            )
                            self.persons[person].max_distance_corner = [
                                further_corner[0] * width,
                                further_corner[1] * height,
                            ]
                            roi_corner = np.array(
                                [
                                    face_x +
                                    (self.persons[person].max_distance_corner[0] /
                                     width * face_width),
                                    face_y +
                                    (self.persons[person].max_distance_corner[1] /
                                     height * face_height),
                                ]
                            )
                            label_to_corner = np.array(
                                self.persons[person].max_distance_corner) - roi_corner
                            label_corner = (
                                roi_corner
                                + (label_to_corner /
                                   np.linalg.norm(label_to_corner, 2))
                                * LABEL_DISTANCE
                            )

                        # Step 3: at this point the face
                        # should have assigned a far-enough
                        # corner point. It's time to print the label
                        roi_corner = np.array(roi_corner, dtype=int)
                        roi_corner = tuple(roi_corner)
                        label_corner = np.array(label_corner, dtype=int)
                        label_corner = tuple(label_corner)
                        img = cv2.line(
                            img,
                            roi_corner,
                            label_corner,
                            color=PASTEL_YELLOW,
                            thickness=LABEL_LINE_THICKNESS,
                            lineType=cv2.LINE_AA,
                        )
                        roi_corner_opposite = self.encode_opposite_corner(
                            self.persons[person].max_distance_corner[0],
                            self.persons[person].max_distance_corner[1]
                        )
                        if roi_corner_opposite[0] == 0:
                            label_top_left_x = label_corner[0]
                            label_bottom_right_x = label_corner[0] + \
                                label_width
                        else:
                            label_top_left_x = label_corner[0] - label_width
                            label_bottom_right_x = label_corner[0]
                        if roi_corner_opposite[1] == 0:
                            label_top_left_y = label_corner[1]
                            label_bottom_right_y = label_corner[1] + \
                                LABEL_HEIGHT
                        else:
                            label_top_left_y = label_corner[1] - LABEL_HEIGHT
                            label_bottom_right_y = label_corner[1]

                        if label_top_left_x < 0:
                            label_top_left_x = 0
                            label_bottom_right_x = label_width
                        elif label_bottom_right_x > width:
                            label_bottom_right_x = width
                            label_top_left_x = width - label_width

                        if label_top_left_y < 0:
                            label_top_left_y = 0
                            label_bottom_right_y = LABEL_HEIGHT
                        elif label_bottom_right_y > height:
                            label_bottom_right_y = height
                            label_top_left_y = height - LABEL_HEIGHT

                        img = cv2.rectangle(
                            img,
                            (label_top_left_x, label_top_left_y),
                            (label_bottom_right_x, label_bottom_right_y),
                            PASTEL_YELLOW,
                            -1,
                            lineType=cv2.LINE_AA,
                        )

                        text_width, text_height = font.getsize(
                            self.persons[person].id_to_display)

                        text_top_left = [
                            label_top_left_x + (label_width - text_width) // 2,
                            label_top_left_y +
                            (LABEL_HEIGHT - text_height) // 2,
                        ]
                        text_top_left = np.array(text_top_left, dtype=int)
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = PILImage.fromarray(rgb_img)

                        draw = ImageDraw.Draw(pil_img)
                        if self.persons[person].id_to_display:
                            draw.text(text_top_left,
                                      self.persons[person].id_to_display,
                                      font=self.font,
                                      fill=TEXT_BLACK)
                        else:
                            draw.text(text_top_left,
                                      id,
                                      font=self.font,
                                      fill=TEXT_BLACK)
                        img = cv2.cvtColor(
                            np.array(pil_img), cv2.COLOR_RGB2BGR)
                        # Print expression if any by emoji
                        if face and (expression := face.expression):
                            expression = str(expression).split('.')[-1].title()
                            emoji = self.get_expression_image(expression)

                            if emoji is not None:
                                emoji_bgr = emoji[:, :, :3]
                                emoji_mask = emoji[:, :, 3]

                                emoji_x = face_x + face_width + 2
                                emoji_y = face_y - emoji.shape[0] - 2

                                if emoji_y < 0:
                                    emoji_y = face_y + face_height + 2

                                if emoji_x + emoji.shape[1] > img.shape[1]:
                                    emoji_x = face_x - emoji.shape[1] - 2

                                emoji_x = max(0, emoji_x)
                                emoji_y = min(emoji_y, img.shape[0] - emoji.shape[0])

                                img = cv2.circle(img, (int(emoji_x + (emoji.shape[1]/2)),
                                                       int(emoji_y + (emoji.shape[0]/2))),
                                                 int(min(emoji.shape[0]/2, emoji.shape[1]/2)) - 1,
                                                 BLACK, FILLED)

                                roi = img[emoji_y:emoji_y + emoji.shape[1],
                                          emoji_x:emoji_x + emoji.shape[0]]

                                alpha_mask = emoji_mask / 255.0

                                roi = alpha_mask[:, :, None] * emoji_bgr \
                                    + (1 - alpha_mask)[:, :, None] * roi

                                img[emoji_y:emoji_y + emoji.shape[1],
                                    emoji_x:emoji_x + emoji.shape[0]] = roi

                    body = tracked_persons[person].body
                    if body and (skeleton := body.skeleton):
                        skeleton = list(skeleton.values())
                        upper_chain = [
                            skeleton[Skeleton2D.RIGHT_WRIST],
                            skeleton[Skeleton2D.RIGHT_ELBOW],
                            skeleton[Skeleton2D.RIGHT_SHOULDER],
                            skeleton[Skeleton2D.NECK],
                            skeleton[Skeleton2D.LEFT_SHOULDER],
                            skeleton[Skeleton2D.LEFT_ELBOW],
                            skeleton[Skeleton2D.LEFT_WRIST],
                        ]

                        body = [
                            skeleton[Skeleton2D.LEFT_SHOULDER],
                            skeleton[Skeleton2D.LEFT_HIP],
                            skeleton[Skeleton2D.RIGHT_HIP],
                            skeleton[Skeleton2D.RIGHT_SHOULDER],
                        ]

                        left_leg = [
                            skeleton[Skeleton2D.LEFT_HIP],
                            skeleton[Skeleton2D.LEFT_KNEE],
                            skeleton[Skeleton2D.LEFT_ANKLE],
                        ]

                        right_leg = [
                            skeleton[Skeleton2D.RIGHT_HIP],
                            skeleton[Skeleton2D.RIGHT_KNEE],
                            skeleton[Skeleton2D.RIGHT_ANKLE],
                        ]

                        skeleton_lines_segments = [
                            upper_chain, body, left_leg, right_leg]

                        for joint in joints_to_draw:
                            joint_x = int(skeleton[joint][0] * width)
                            joint_y = int(skeleton[joint][1] * height)
                            img = cv2.circle(
                                img, (joint_x, joint_y), JOINT_RADIUS, PASTEL_YELLOW, FILLED
                            )

                        for idx, segment in enumerate(skeleton_lines_segments):
                            segment = [
                                (int(joint[0] * width), int(joint[1] * height))
                                for joint in segment
                            ]
                            segment = np.array(segment, dtype=np.int32)
                            skeleton_lines_segments[idx] = segment

                        img = cv2.polylines(
                            img,
                            skeleton_lines_segments,
                            isClosed=False,
                            color=PASTEL_YELLOW,
                            thickness=THICKNESS_CORNERS,
                            lineType=cv2.LINE_AA,
                        )

                if self.compressed_output:
                    img_msg = CompressedImage()
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.format = "jpeg"
                    img_msg.data = np.array(
                        cv2.imencode(".jpg", img)[1]).tobytes()
                else:
                    img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")

                # Convert to milliseconds
                self.proc_time_ms = (
                    self.get_clock().now() - start_proc_time).nanoseconds / 1e6
                self.img_pub.publish(img_msg)

    def do_diagnostics(self):
        """Publish diagnostic messages."""
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()

        msg = DiagnosticStatus(
            name="Social perception: Visualization", hardware_id="none")
        msg.level = DiagnosticStatus.OK
        msg.values = [
            KeyValue(key="Package name", value='hri_visualization'),
            # Update with actual rendering time if needed
            KeyValue(key="Rendering time", value="Not Calculated")
        ]

        arr.status = [msg]
        self.diag_pub.publish(arr)

    def find_max_distance_corner(self, x, y, width, height):
        """
        Return the maximum distance corner of the image.

        The result is encoded as:
                - [0, 0] = top left corner;
                - [1, 0] = top right corner;
                - [0, 1] = bottom left corner;
                - [1, 1] = bottom right corner.
        """
        return [x < width / 2, y < height / 2]

    def encode_opposite_corner(self, corner_x, corner_y):
        """Return the opposite corner."""
        return [int(not bool(corner_x)), int(not bool(corner_y))]

    def calibrate_font_size(self, number_of_letters):
        """Calibrate font to fit the person label."""
        self.fontsize = 1
        label = LARGEST_LETTER * number_of_letters
        label_width = SPACE_PER_CHARACTER * number_of_letters

        # Ensure FONT is a string
        font = ImageFont.truetype(FONT, self.fontsize)
        text_width, text_height = font.getsize(label)

        while text_width < (label_width - 10) and text_height < (LABEL_HEIGHT - 6):
            self.fontsize += 1
            font = ImageFont.truetype(FONT, self.fontsize)
            text_size, text_height = font.getsize(label)

        self.fontsize -= 2
        return ImageFont.truetype(FONT, self.fontsize)


def main(args=None):
    rclpy.init(args=args)

    visualizer = HRIVisualizer()

    rclpy.spin(visualizer)

    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
