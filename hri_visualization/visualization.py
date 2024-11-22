#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from hri_msgs.msg import IdsList, Skeleton2D
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
TEXT_BLACK = (0, 0, 0, 255)
BOX_THICKNESS = 3
THICKNESS_CORNERS = 2
LABEL_DISTANCE = 50
LABEL_WIDTH = 80
LABEL_HEIGHT = 30
SPACE_PER_CHARACTER = 14
LABEL_LINE_THICKNESS = 1
JOINT_RADIUS = 15
JOINT_THICKNESS = -1

package_path = Path(get_package_share_directory("hri_visualization"))

FONTS_FOLDER = package_path / "fonts"
FONT = str(FONTS_FOLDER / "Montserrat/Montserrat-SemiBold.ttf")
LARGEST_LETTER = "M"
VIS_CORNER_GAP_RATIO = 0.1
VIS_CORNER_WIDTH_RATIO = 0.1
VIS_CORNER_HEIGHT_RATIO = 0.1

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
    """ A class managing publishing an
        image stream where information
        regarding the detected people
        is displayed
    """
    class PersonDescriptor:
        """ Internal class used to describe
            person-related aspects
        """
        id_to_display = None
        label_width = None
        font = None
        max_distance_corner = None

    def __init__(self):
        """ Constructor """
        super().__init__('hri_visualization')
        self.declare_parameter('funny_names', False)
        self.declare_parameter('compressed_input', True)
        self.declare_parameter('compressed_output', True)
        self.funny_names = self.get_parameter('funny_names').value
        self.compressed_input = self.get_parameter('compressed_input').value
        self.compressed_output = self.get_parameter('compressed_output').value
        self.font = self.calibrate_font_size(5)

        self.hri_listener = HRIListener('hri_listener')

        self.body_sub = self.create_subscription(
            IdsList, "/humans/bodies/tracked", self.body_cb, 1)
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
                Image, "/image", self.img_cb, qos_profile=qos_profile_sensor_data)
        self.hri_overlay_topic = resolved_topic_name + "/hri_overlay"

        if self.compressed_output:
            self.img_pub = self.create_publisher(
                CompressedImage, self.hri_overlay_topic + "/compressed", qos_profile=qos_profile_sensor_data)
        else:
            self.img_pub = self.create_publisher(
                Image, self.hri_overlay_topic, qos_profile=qos_profile_sensor_data)

        diag_period = self.declare_parameter("diagnostic_period", 1.0).value
        self.diag_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 1)
        self.diag_timer = self.create_timer(diag_period, self.do_diagnostics)

        self.proc_time_ms = 0

        self.bridge = CvBridge()

        self.faces = {}
        self.bodies = {}
        self.persons = {}

        self.persons_lock = Lock()

    def body_cb(self, msg):
        """ Callback storing information regarding the detected bodies """
        for id in msg.ids:
            if id not in self.bodies:
                skeleton_topic = f"/humans/bodies/{id}/skeleton2d"
                self.bodies[id] = [
                    self.create_subscription(
                        Skeleton2D, skeleton_topic, self.skeleton_cb, 1),
                    None,
                ]

        for id in list(self.bodies):
            if id not in msg.ids:
                # Unregister the subscription in ROS2
                self.bodies[id][0].destroy()
                del self.bodies[id]

    def skeleton_cb(self, skeleton_msg, args):
        """ Callback storing information regarding
            a body's skeleton coordinates
        """
        id = args
        self.bodies[id][1] = skeleton_msg.skeleton

    def generate_funny_ids(self, person_id):
        """ Extracting a funny id, which is a combination
            of an adjective and a name from two 
            predefined lists
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
        """Assuming expression is of type Happy, Sad, Neutral,
        find their respective emoji directories like happy.png
        """
        filename = f"{expression.lower()}.png"
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
        """ Callback managing the incoming images.
            It performs the same operations as
            img_cb, but with compressed images
        """
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        img_msg.header = msg.header
        self.img_cb(img_msg)

    def img_cb(self, msg):
        """ Callback managing the incoming images.
            It draws the bounding boxes for the
            detected faces and the skeletons
            for the detected bodies
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
                    if not person in tracked_persons:
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
                        label_to_corner_distance = np.linalg.norm([roi_corner[0]
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
                            self.persons[person].max_distance_corner[0], self.persons[person].max_distance_corner[1]
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
                            emoji_image = self.get_expression_image(expression)

                            if emoji_image is not None:
                                emoji_size = (40, 40)
                                emoji_image = cv2.resize(
                                    emoji_image, emoji_size)
                                emoji_bgr = emoji_image[:,
                                                        :, :3] + PASTEL_YELLOW
                                emoji_mask = emoji_image[:, :, 3]

                                emoji_x = face_x + face_width + 2
                                emoji_y = face_y - emoji_size[1] - 2

                                if emoji_y < 0:
                                    emoji_y = face_y + face_height + 2

                                elif emoji_x + emoji_size[0] > img.shape[1]:

                                    emoji_x = img.shape[1] - emoji_size[0] - 2
                                elif emoji_y + emoji_size[1] > img.shape[0]:
                                    emoji_y = img.shape[0] - emoji_size[1] - 2

                                if (emoji_y + emoji_size[1] > img.shape[0]) or (emoji_x + emoji_size[0] > img.shape[1]):
                                    emoji_x = face_x + face_width - \
                                        emoji_size[0]
                                    emoji_y = face_y + face_height

                                emoji_x = max(0, emoji_x)
                                emoji_y = max(0, emoji_y)

                                roi = img[emoji_y:emoji_y + emoji_size[1],
                                          emoji_x:emoji_x + emoji_size[0]]
                                alpha_mask = emoji_mask / 255.0

                                for c in range(0, 3):
                                    roi[:, :, c] = (
                                        alpha_mask * emoji_bgr[:, :, c] + (1 - alpha_mask) * roi[:, :, c])

                                img[emoji_y:emoji_y + emoji_size[1],
                                    emoji_x:emoji_x + emoji_size[0]] = roi

                for id in list(self.bodies):
                    skeleton = self.bodies[id][1]
                    if skeleton:
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
                            joint_x = int(skeleton[joint].x * width)
                            joint_y = int(skeleton[joint].y * height)
                            img = cv2.circle(
                                img, (joint_x, joint_y), JOINT_RADIUS, PASTEL_YELLOW, JOINT_THICKNESS
                            )

                        for idx, segment in enumerate(skeleton_lines_segments):
                            segment = [
                                (int(joint.x * width), int(joint.y * height))
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
        """ Periodic function to publish diagnostic messages """
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
        """ Function returning the maximum distance
            corner of the image. The result is encoded as:
                - [0, 0] = top left corner
                - [1, 0] = top right corner
                - [0, 1] = bottom left corner
                - [1, 1] = bottom right corner
        """
        return [x < width / 2, y < height / 2]

    def encode_opposite_corner(self, corner_x, corner_y):
        """ Following the same type of enconding
            described in the find_max_distance_corner,
            this function returns the opposite corner
        """
        return [int(not bool(corner_x)), int(not bool(corner_y))]

    def calibrate_font_size(self, number_of_letters):
        """ Initial calibration of the font size
            for the ids label. It uses PIL tools,
            not opencv tools
        """
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
    # Initialize the ROS2 system
    rclpy.init(args=args)

    # Create the HRIVisualizer node
    visualizer = HRIVisualizer()

    # Spin the visualizer node
    rclpy.spin(visualizer)

    # Clean up
    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
