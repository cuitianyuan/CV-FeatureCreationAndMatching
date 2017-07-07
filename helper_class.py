import os
import cv2
import ps5

input_dir = "input"
output_dir = "output"


class Ps5Arrays:
    """Helper class that performs the problem set 5 operations.

    This class is used in order to help partition the parts in experiment.py. Almost all sections
    of this problem set depend on the previous ones. For this reason, each method stores the
    results relevant to the problem set requirements and returns them when necessary. The main purpose
    of this methodology is to prevent redoing the work carried out by previous sections when all parts
    are called at the same time. The class functions verify if a class variable has been already
    populated before running the process behind the problem set part.

    Warning: This file must not be modified. The submissions process will only collect the ps5.py and
    experiment.py as usual. The original version of this file will be used during the grading process.

    """

    def __init__(self):
        self.trans_a = cv2.imread(os.path.join(input_dir, "transA.jpg"), 0) / 255.
        self.trans_b = cv2.imread(os.path.join(input_dir, "transB.jpg"), 0) / 255.
        self.sim_a = cv2.imread(os.path.join(input_dir, "simA.jpg"), 0) / 255.
        self.sim_b = cv2.imread(os.path.join(input_dir, "simB.jpg"), 0) / 255.
        self.gradients_a = None
        self.gradients_b = None
        self.r_maps = None
        self.corners = None
        self.angles = None
        self.keypoints = None
        self.matches = None
        self.descriptors = None

    def get_input_images(self):
        return {"trans_a": self.trans_a, "trans_b": self.trans_b, "sim_a": self.sim_a, "sim_b": self.sim_b}

    def get_gradients_a(self):

        if self.gradients_a is None:
            trans_a_x = ps5.gradient_x(self.trans_a)
            trans_a_y = ps5.gradient_y(self.trans_a)
            sim_a_x = ps5.gradient_x(self.sim_a)
            sim_a_y = ps5.gradient_y(self.sim_a)

            self.gradients_a = {"t_x": trans_a_x, "t_y": trans_a_y, "s_x": sim_a_x, "s_y": sim_a_y}

        return self.gradients_a

    def get_gradients_b(self):

        if self.gradients_b is None:
            trans_b_x = ps5.gradient_x(self.trans_b)
            trans_b_y = ps5.gradient_y(self.trans_b)
            sim_b_x = ps5.gradient_x(self.sim_b)
            sim_b_y = ps5.gradient_y(self.sim_b)

            self.gradients_b = {"t_x": trans_b_x, "t_y": trans_b_y, "s_x": sim_b_x, "s_y": sim_b_y}

        return self.gradients_b

    def calculate_r_maps(self, k_dims, alpha):

        if self.r_maps is None:
            gradients = self.get_gradients_a()
            trans_a_x, trans_a_y = gradients["t_x"], gradients["t_y"]
            sim_a_x, sim_a_y = gradients["s_x"], gradients["s_y"]

            gradients_b = self.get_gradients_b()
            trans_b_x, trans_b_y = gradients_b["t_x"], gradients_b["t_y"]
            sim_b_x, sim_b_y = gradients_b["s_x"], gradients_b["s_y"]

            trans_a_r = ps5.harris_response(trans_a_x, trans_a_y, k_dims["trans_a"], alpha["trans_a"])
            trans_b_r = ps5.harris_response(trans_b_x, trans_b_y, k_dims["trans_b"], alpha["trans_b"])
            sim_a_r = ps5.harris_response(sim_a_x, sim_a_y, k_dims["sim_a"], alpha["sim_a"])
            sim_b_r = ps5.harris_response(sim_b_x, sim_b_y, k_dims["sim_b"], alpha["sim_b"])

            self.r_maps = {"trans_a": trans_a_r, "trans_b": trans_b_r, "sim_a": sim_a_r, "sim_b": sim_b_r}

    def get_r_maps(self):
        return self.r_maps

    def find_corners(self, threshold, radius):

        if self.corners is None:
            self.corners = {"trans_a": ps5.find_corners(self.r_maps["trans_a"], threshold["trans_a"],
                                                        radius["trans_a"]),
                            "trans_b": ps5.find_corners(self.r_maps["trans_b"], threshold["trans_b"],
                                                        radius["trans_b"]),
                            "sim_a": ps5.find_corners(self.r_maps["sim_a"], threshold["sim_a"],
                                                      radius["sim_a"]),
                            "sim_b": ps5.find_corners(self.r_maps["sim_b"], threshold["sim_b"],
                                                      radius["sim_b"])}

    def get_corners(self):
        return self.corners

    def compute_angles(self):

        if self.angles is None:
            gradients = self.get_gradients_a()
            trans_a_x, trans_a_y = gradients["t_x"], gradients["t_y"]
            sim_a_x, sim_a_y = gradients["s_x"], gradients["s_y"]

            gradients_b = self.get_gradients_b()
            trans_b_x, trans_b_y = gradients_b["t_x"], gradients_b["t_y"]
            sim_b_x, sim_b_y = gradients_b["s_x"], gradients_b["s_y"]

            self.angles = {"trans_a": ps5.gradient_angle(trans_a_x, trans_a_y),
                           "trans_b": ps5.gradient_angle(trans_b_x, trans_b_y),
                           "sim_a": ps5.gradient_angle(sim_a_x, sim_a_y),
                           "sim_b": ps5.gradient_angle(sim_b_x, sim_b_y)}

    def create_keypoints(self, size, octave):

        if self.keypoints is None:
            corners = self.corners
            angles = self.angles

            trans_a_kp = ps5.get_keypoints(corners["trans_a"], angles["trans_a"], size["trans_a"],
                                                        octave["trans_a"])
            trans_b_kp = ps5.get_keypoints(corners["trans_b"], angles["trans_b"], size["trans_b"],
                                                        octave["trans_b"])
            sim_a_kp = ps5.get_keypoints(corners["sim_a"], angles["sim_a"], size["sim_a"],
                                                    octave["sim_a"])
            sim_b_kp = ps5.get_keypoints(corners["sim_b"], angles["sim_b"], size["sim_b"],
                                                    octave["sim_b"])

            self.keypoints = {"trans_a": trans_a_kp, "trans_b": trans_b_kp, "sim_a": sim_a_kp, "sim_b": sim_b_kp}

    def get_keypoints(self):
        return self.keypoints

    def get_descriptors(self):

        if self.descriptors is None:
            images = self.get_input_images()
            k_pts = self.get_keypoints()
            trans_a_des, trans_a_kp = ps5.get_descriptors(images["trans_a"], k_pts["trans_a"])
            trans_b_des, trans_b_kp = ps5.get_descriptors(images["trans_b"], k_pts["trans_b"])
            sim_a_des, sim_a_kp = ps5.get_descriptors(images["sim_a"], k_pts["sim_a"])
            sim_b_des, sim_b_kp = ps5.get_descriptors(images["sim_b"], k_pts["sim_b"])

            self.descriptors = {"trans_a": trans_a_des, "trans_b": trans_b_des, "sim_a": sim_a_des, "sim_b": sim_b_des}
            self.keypoints = {"trans_a": trans_a_kp, "trans_b": trans_b_kp, "sim_a": sim_a_kp, "sim_b": sim_b_kp}

        return self.descriptors

    def get_matches(self):

        if self.matches is None:

            self.matches = {"trans": ps5.match_descriptors(self.descriptors["trans_a"], self.descriptors["trans_b"]),
                            "sim": ps5.match_descriptors(self.descriptors["sim_a"], self.descriptors["sim_b"])}

        return self.matches
