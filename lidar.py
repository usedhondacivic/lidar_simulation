from typing import Any
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
from numpy._typing import NDArray

import util

M_TO_MM = 1e3

# measurements from datasheet
# https://wiki.youyeetoo.com/en/Lidar/LD20
LIDAR_DIAMETER = 59.82
LASER_SENSOR_SPACING = LIDAR_DIAMETER / 2.0
LENS_FOCAL_LENGTH = LIDAR_DIAMETER / 2.0

class Lidar:
    def __init__(self, plt, axs):
        # Line segment representing the lens / sensor
        self.lens_seg = Seg(
            np.array([LASER_SENSOR_SPACING / 2.0 - 10.0, LIDAR_DIAMETER / 2.0 - 2.5]),
            np.array([LASER_SENSOR_SPACING / 2.0 + 10.0, LIDAR_DIAMETER / 2.0 - 2.5]),
        )

        self.sensor_seg = Seg(
            np.array([LASER_SENSOR_SPACING / 2.0 - 10.0, 2.5]),
            np.array([LASER_SENSOR_SPACING / 2.0 + 10.0, 2.5]),
        )

        self.axs = axs
        self.plt = plt

    def draw(self):
        head = mpatches.Circle((0.0, 0.0), LIDAR_DIAMETER / 2.0, fill=False)
        diode = mpatches.Rectangle(
            (-LASER_SENSOR_SPACING / 2.0, 0.0),
            10.0,
            LIDAR_DIAMETER / 2.0,
            edgecolor=(1.0, 0.0, 0.0),
            facecolor=(1.0, 1.0, 1.0),
        )
        diode = mpatches.Rectangle(
            (-LASER_SENSOR_SPACING / 2.0 - 5.0, 0.0),
            10.0,
            LIDAR_DIAMETER / 2.0,
            edgecolor=(1.0, 0.0, 0.0),
            facecolor=(1.0, 1.0, 1.0),
            label="Laser",
        )

        sensor_holder = mpatches.Rectangle(
            (LASER_SENSOR_SPACING / 2.0 - 10.0, 1.0),
            20.0,
            LIDAR_DIAMETER / 2.0 - 2.0,
            facecolor=(1.0, 1.0, 1.0),
        )

        lens = mpatches.Rectangle(
            (LASER_SENSOR_SPACING / 2.0 - 10.0, LIDAR_DIAMETER / 2.0 - 5.0),
            20.0,
            5.0,
            facecolor=(0.75, 1.0, 1.0),
            label="Lens",
        )

        sensor = mpatches.Rectangle(
            (LASER_SENSOR_SPACING / 2.0 - 10.0, 0.0),
            20.0,
            5.0,
            facecolor=(0.75, 0.75, 0.75),
            label="Sensor",
        )

        self.axs.add_artist(head)
        self.axs.add_artist(diode)
        self.axs.add_artist(sensor_holder)
        self.axs.add_artist(lens)
        self.axs.add_artist(sensor)

    def initial_raycast(self, walls, draw):
        """
        Find the first wall the laser hits
        """

        # Where the laser starts, traveling in the +y direction
        laser_origin = np.array([-LASER_SENSOR_SPACING / 2.0, 10.0])

        # Find all intersections of the world / laser
        intersection_points = []
        for wall in walls:
            hit = util.ray_line_seg_intersection(
                laser_origin, np.array([0.0, 1.0]), wall.start, wall.end
            )
            match hit:
                case None:
                    continue
                case point:
                    intersection_points.append(point)

        # We only care about the closes intersection, the rest are occluded
        hit_point = intersection_points[
            np.argmin(np.linalg.norm(np.array(intersection_points), axis=1))
        ]

        if draw:
            self.plt.plot(
                [laser_origin[0], hit_point[0]], [laser_origin[1], hit_point[1]], "ro-"
            )

        return hit_point

    def find_backwards_focal_point(self, hit_point, draw):
        """
        Use two ray casts to determine the focal point behind the lens
        The first comes in through the forward focal point of the lens, and comes out parallel to the principle axis
        The second travels straight through the center point of the lens, leaving unrefracted
        See https://www.physicsclassroom.com/Class/refrn/U14L5da.cfm#rules
        """

        # Ray through the forward focal point will emerge from the lens parallel to the principle axis
        forward_focal_point = np.array(
            [LASER_SENSOR_SPACING / 2.0, LIDAR_DIAMETER / 2.0 - 2.5 + LENS_FOCAL_LENGTH]
        )

        through_forward_focal_point_lens_intersect = util.ray_line_intersection(
            hit_point,
            forward_focal_point - hit_point,
            self.lens_seg.start,
            self.lens_seg.end,
        )

        if through_forward_focal_point_lens_intersect is None:
            print("Error with ray line intersection calculation")
            return

        # Ray through the lens center point will continue traveling straight
        lens_center_point = np.array(
            [LASER_SENSOR_SPACING / 2.0, LIDAR_DIAMETER / 2.0 - 2.5]
        )

        backward_focal_point = util.ray_ray_intersection(
            hit_point,
            lens_center_point - hit_point,
            through_forward_focal_point_lens_intersect,
            np.array([0.0, -1.0]),
        )

        if backward_focal_point is None:
            print("Error with ray ray intersection calculation")
            return

        if draw:
            self.plt.plot(
                [through_forward_focal_point_lens_intersect[0], hit_point[0]],
                [through_forward_focal_point_lens_intersect[1], hit_point[1]],
                "k",
            )

            self.plt.scatter(
                [forward_focal_point[0]], [forward_focal_point[1]], label="Focal point"
            )

            self.plt.scatter(
                [lens_center_point[0]], [lens_center_point[1]], label="Center of lens"
            )

            self.plt.scatter(
                [backward_focal_point[0]],
                [backward_focal_point[1]],
                label="Image focal point",
            )

            self.plt.plot(
                [backward_focal_point[0], hit_point[0]],
                [backward_focal_point[1], hit_point[1]],
                "k",
                label="Raycasts used for finding image focal",
            )

            self.plt.plot(
                [
                    backward_focal_point[0],
                    through_forward_focal_point_lens_intersect[0],
                ],
                [
                    backward_focal_point[1],
                    through_forward_focal_point_lens_intersect[1],
                ],
                "k",
            )

        return backward_focal_point

    def find_projection_range(self, hit_point, backwards_focal_point, draw):
        left_sensor = util.ray_line_intersection(
            self.lens_seg.start,
            backwards_focal_point - self.lens_seg.start,
            self.sensor_seg.start,
            self.sensor_seg.end,
        )

        right_sensor = util.ray_line_intersection(
            self.lens_seg.end,
            backwards_focal_point - self.lens_seg.end,
            self.sensor_seg.start,
            self.sensor_seg.end,
        )

        if left_sensor is None or right_sensor is None:
            print("Error with ray ray intersection calculation")
            return

        if draw:
            self.plt.scatter(
                [left_sensor[0], right_sensor[0]], [left_sensor[1], right_sensor[1]]
            )
            self.plt.plot(
                [hit_point[0], self.lens_seg.start[0], left_sensor[0]],
                [hit_point[1], self.lens_seg.start[1], left_sensor[1]],
                "r",
                label="Image extremity rays",
            )
            self.plt.plot(
                [hit_point[0], self.lens_seg.end[0], right_sensor[0]],
                [hit_point[1], self.lens_seg.end[1], right_sensor[1]],
                "r",
            )


class Seg:
    def __init__(self, start: NDArray[Any], end: NDArray[Any]):
        self.start = start
        self.end = end

    def draw(self):
        plt.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], "bo-")


if __name__ == "__main__":
    with plt.ioff():
        fig, axs = plt.subplots()

    plt.axis("equal")
    axs.set_xlim(-0.25 * M_TO_MM, 0.25 * M_TO_MM)
    axs.set_axis_off()

    lidar = Lidar(plt, axs)
    lidar.draw()

    walls = [
        Seg(
            np.array([-10.0 * M_TO_MM, 0.5 * M_TO_MM]),
            np.array([10.0 * M_TO_MM, 0.5 * M_TO_MM]),
        ),
    ]
    for wall in walls:
        wall.draw()

    hit_point = lidar.initial_raycast(walls, True)
    backwards_focal_point = lidar.find_backwards_focal_point(hit_point, True)
    lidar.find_projection_range(hit_point, backwards_focal_point, True)

    fig.tight_layout()
    axs.legend(loc="center left")
    plt.show()
