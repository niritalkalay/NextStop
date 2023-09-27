import numpy as np, time
from numba import jit
from scipy.spatial import ConvexHull
from tracking.box import Box3D  # from AB3DMOT_libs.box import Box3D
from sklearn.neighbors import KDTree
from tracking.visual_utils import draw_scenes, draw_scenes_raw
from numpy.linalg import lstsq as optimizer


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) >= (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s): outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0: return None
    return (outputList)


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        if len(inter_p) >= 3 and PolyArea2D(np.array(inter_p)) > 1e-4:
            # print('convex_hull_intersection : p1=',p1)
            # print('convex_hull_intersection : p2=', p2)
            # print('convex_hull_intersection : inter_p=', inter_p_independent)
            # print('##################################################')

            hull_inter = ConvexHull(inter_p)
        else:
            return None, 0.0
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def compute_inter_2D(boxa_bottom, boxb_bottom):
    # computer intersection over union of two sets of bottom corner points

    _, I_2D = convex_hull_intersection(boxa_bottom, boxb_bottom)

    # a slower version
    # from shapely.geometry import Polygon
    # reca, recb = Polygon(boxa_bottom), Polygon(boxb_bottom)
    # I_2D = reca.intersection(recb).area

    return I_2D


def compute_height(box_a, box_b, inter=True):
    corners1 = Box3D.box2corners3d_camcoord(box_a)  # 8 x 3
    corners2 = Box3D.box2corners3d_camcoord(box_b)  # 8 x 3
    # this always holds : cornersX[0] << cornersX[4]

    if inter:  # compute overlap height
        ymax = min(corners1[0, 1], corners2[0, 1])
        ymin = max(corners1[4, 1], corners2[4, 1])
        height = max(0.0, ymax - ymin)
    else:
        # compute union height
        ymax = max(corners1[0, 1], corners2[0, 1])
        ymin = min(corners1[4, 1], corners2[4, 1])
        height = max(0.0, ymax - ymin)

    return height


def compute_bottom(box_a, box_b):
    # obtain ground corners and area, not containing the height

    corners1 = Box3D.box2corners3d_camcoord(box_a)  # 8 x 3
    corners2 = Box3D.box2corners3d_camcoord(box_b)  # 8 x 3

    # get bottom corners and inverse order so that they are in the
    # counter-clockwise order to fulfill polygon_clip
    boxa_bot = corners1[-5::-1, [0, 2]]  # 4 x 2
    boxb_bot = corners2[-5::-1, [0, 2]]  # 4 x 2

    return boxa_bot, boxb_bot


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


def convex_area(boxa_bottom, boxb_bottom):
    # compute the convex area
    all_corners = np.vstack((boxa_bottom, boxb_bottom))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)

    return convex_area


#################### distance metric
def iou_raw(box_a_raw, box_b_raw, metric='giou_3d'):
    h, w, l, cx, cy, cz, ry = box_a_raw
    box_a = Box3D(x=cx, y=cy, z=cz, h=h, w=w, l=l, ry=ry)
    h, w, l, cx, cy, cz, ry = box_b_raw
    box_b = Box3D(x=cx, y=cy, z=cz, h=h, w=w, l=l, ry=ry)
    return iou(box_a, box_b, metric=metric)


def iou_not_in_use(box_a, box_b, metric='iou_3d'):
    if metric != 'iou_3d' and metric != 'diou_3d':
        raise ValueError('unsupported method')
    else:

        print('iou : box_a (cx,cy,xz,theta,l,w,h)=', box_a.x, box_a.y, box_a.z, box_a.ry, box_a.l, box_a.w, box_a.h)
        print('iou : box_b (cx,cy,xz,theta,l,w,h)=', box_b.x, box_b.y, box_b.z, box_b.ry, box_b.l, box_b.w, box_b.h)

        obj = IoU(box_a, box_b)
        if metric == 'iou_3d':
            return obj.iou()
        else:
            return obj.diou()


def iou(box_a, box_b, metric='giou_3d'):
    ''' Compute 3D/2D bounding box IoU, only working for object parallel to ground

    Input:
        Box3D instances
    Output:
        iou_3d: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    box corner order is like follows
            1 -------- 0 		 top is bottom because y direction is negative
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    rect/ref camera coord:
    right x, down y, front z
    '''

    # compute 2D related measures
    # print('iou : box_a (cx,cy,xz,theta,l,w,h)=', box_a.x,box_a.y,box_a.z, box_a.ry,box_a.l,box_a.w,box_a.h)
    # print('iou : box_b (cx,cy,xz,theta,l,w,h)=', box_b.x, box_b.y, box_b.z, box_b.ry, box_b.l, box_b.w, box_b.h)
    # print('~~~~~~~~~~~~~~')

    boxa_bot, boxb_bot = compute_bottom(box_a, box_b)
    I_2D = compute_inter_2D(boxa_bot, boxb_bot)

    # only needed for GIoU & DIou
    if 'giou' in metric or 'diou' in metric:
        C_2D = convex_area(boxa_bot, boxb_bot)

    if '2d' in metric:  # return 2D IoU/GIoU
        U_2D = box_a.w * box_a.l + box_b.w * box_b.l - I_2D
        if metric == 'iou_2d':  return I_2D / U_2D
        if metric == 'giou_2d': return I_2D / U_2D - (C_2D - U_2D) / C_2D
        if metric == 'diou_2d':
            # distance between box centers
            d = np.linalg.norm([box_a.x - box_b.x, box_a.y - box_b.y])

            # diagonal length off the smallest encclosing box covering the two boxes:
            # https://medium.com/visionwizard/understanding-diou-loss-a-quick-read-a4a0fbcbf0f0

            xmin = np.min((box_a.x - 0.5 * box_a.l, box_b.x - 0.5 * box_b.l))
            xmax = np.max((box_a.x + 0.5 * box_a.l, box_b.x + 0.5 * box_b.l))

            ymin = np.min((box_a.y - 0.5 * box_a.w, box_b.y - 0.5 * box_b.w))
            ymax = np.max((box_a.y + 0.5 * box_a.w, box_b.y + 0.5 * box_b.w))

            c = np.linalg.norm([xmax - xmin, ymax - ymin])

            return I_2D / U_2D - (d * d) / (c * c)

    elif '3d' in metric:  # return 3D IoU/GIoU
        overlap_height = compute_height(box_a, box_b)
        I_3D = I_2D * overlap_height
        U_3D = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - I_3D
        if metric == 'iou_3d':  return I_3D / U_3D
        if metric == 'giou_3d':
            union_height = compute_height(box_a, box_b, inter=False)
            C_3D = C_2D * union_height
            return I_3D / U_3D - (C_3D - U_3D) / C_3D
        if metric == 'diou_3d':
            # union_height = compute_height(box_a, box_b, inter=False)
            # C_3D = C_2D * union_height

            # distance between box centers
            d = np.linalg.norm([box_a.x - box_b.x, box_a.y - box_b.y, box_a.z - box_b.z])

            # diagonal length off the smallest encclosing box covering the two boxes:
            # https://medium.com/visionwizard/understanding-diou-loss-a-quick-read-a4a0fbcbf0f0

            xmin = np.min((box_a.x - 0.5 * box_a.l, box_b.x - 0.5 * box_b.l))
            xmax = np.max((box_a.x + 0.5 * box_a.l, box_b.x + 0.5 * box_b.l))

            ymin = np.min((box_a.y - 0.5 * box_a.w, box_b.y - 0.5 * box_b.w))
            ymax = np.max((box_a.y + 0.5 * box_a.w, box_b.y + 0.5 * box_b.w))

            zmin = np.min((box_a.z - 0.5 * box_a.h, box_b.z - 0.5 * box_b.h))
            zmax = np.max((box_a.z + 0.5 * box_a.h, box_b.z + 0.5 * box_b.h))

            c = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])

            return I_3D / U_3D - (d * d) / (c * c)

    else:
        assert False, '%s is not supported'


def dist_ground(bbox1, bbox2):
    # Compute distance of bottom center in 3D space, NOT considering the difference in height

    c1 = Box3D.bbox2array(bbox1)[[0, 2]]
    c2 = Box3D.bbox2array(bbox2)[[0, 2]]
    dist = np.linalg.norm(c1 - c2)

    return dist


def dist3d_bottom(bbox1, bbox2):
    # Compute distance of bottom center in 3D space, considering the difference in height / 2

    c1 = Box3D.bbox2array(bbox1)[:3]
    c2 = Box3D.bbox2array(bbox2)[:3]
    dist = np.linalg.norm(c1 - c2)

    return dist


def dist3d(bbox1, bbox2):
    # Compute distance of actual center in 3D space, considering the difference in height

    corners1 = Box3D.box2corners3d_camcoord(bbox1)  # 8 x 3
    corners2 = Box3D.box2corners3d_camcoord(bbox2)  # 8 x 3

    # compute center point based on 8 corners
    c1 = np.average(corners1, axis=0)
    c2 = np.average(corners2, axis=0)

    dist = np.linalg.norm(c1 - c2)

    return dist


def diff_orientation_correction(diff):
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    if diff > np.pi / 2:  diff -= np.pi
    if diff < -np.pi / 2: diff += np.pi
    return diff


def m_distance(det, trk, trk_inv_innovation_matrix=None):
    # compute difference
    det_array = Box3D.bbox2array(det)[:7]
    trk_array = Box3D.bbox2array(trk)[:7]  # (7, )
    diff = np.expand_dims(det_array - trk_array, axis=1)  # 7 x 1

    # correct orientation
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        dist = np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
    else:
        dist = np.sqrt(np.dot(diff.T, diff))  # distance along 7 dimension
    return dist


def nearestDistance(tree, pt):
    dist, ind = tree.query([pt], k=1)
    return dist[0]


def _similarity(cloudA, cloudB, threshold):
    # compare B to A
    # use threshold for identifying outliers and not considering those for the similarity
    # a good value for threshold is 5 * <cloud_resolution>, e.g. 10cm for a cloud with 2cm resolution

    num_outlier = 0
    tree = KDTree(cloudA, leaf_size=2)
    sum = 0
    for pb in cloudB:
        dist = nearestDistance(tree, pb)
        if dist < threshold:
            sum += dist
        else:
            num_outlier += 1

    return sum / (len(cloudB) - num_outlier + +0.01)


def PointsSimilarity(det_cloud, trk_cloud, threshold=10):
    cloudA = det_cloud
    cloudB = trk_cloud
    # https://stackoverflow.com/questions/55913968/metric-to-compare-two-point-clouds-similarity

    if len(cloudA) == 0 or len(cloudB) == 0:
        return 100  # max distance  value

    Similarity_det2trk = _similarity(cloudA, cloudB, threshold)

    Similarity_trk2det = _similarity(cloudB, cloudA, threshold)

    return (Similarity_det2trk * 0.5) + (Similarity_trk2det * 0.5)


def getPointsInsideBox(points, box):
    h = box.h
    w = box.w
    l = box.l
    cx = box.x
    cy = box.y
    cz = box.z
    ry = box.ry

    xmin = cx - l * 0.5
    xmax = xmin + l

    ymin = cy - w * 0.5
    ymax = ymin + w

    zmin = cz - h * 0.5
    zmax = zmin + h

    idx_x = np.intersect1d(np.where(points[:, 0] <= xmax)[0], np.where(points[:, 0] >= xmin)[0])
    idx_y = np.intersect1d(np.where(points[:, 1] <= ymax)[0], np.where(points[:, 1] >= ymin)[0])
    idx_z = np.intersect1d(np.where(points[:, 2] <= zmax)[0], np.where(points[:, 2] >= zmin)[0])

    idx = np.intersect1d(np.intersect1d(idx_x, idx_y), idx_z)

    return points[idx, :]


#############################################################################
class IoU(object):
    """General Intersection Over Union cost for Oriented 3D bounding boxes."""

    def __init__(self, box1, box2):
        self._box1 = box1
        self._box2 = box2
        self._intersection_points = []

    def get_box_volume(self, box):
        return box.w * box.h * box.l

    def iou(self):
        """Computes the exact IoU using Sutherland-Hodgman algorithm."""
        self._intersection_points = []
        self._compute_intersection_points(self._box1, self._box2)
        self._compute_intersection_points(self._box2, self._box1)
        if self._intersection_points:
            intersection_volume = ConvexHull(self._intersection_points).volume
            box1_volume = self.get_box_volume(self._box1)
            box2_volume = self.get_box_volume(self._box2)
            union_volume = box1_volume + box2_volume - intersection_volume
            return intersection_volume / union_volume
        else:
            return 0.

    def diou(self):
        iou = self.iou()

        box_a = self._box1
        box_b = self._box2
        # distance between box centers
        d = np.linalg.norm([box_a.x - box_b.x, box_a.y - box_b.y, box_a.z - box_b.z])

        # diagonal length off the smallest encclosing box covering the two boxes:
        # https://medium.com/visionwizard/understanding-diou-loss-a-quick-read-a4a0fbcbf0f0

        xmin = np.min((box_a.x - 0.5 * box_a.l, box_b.x - 0.5 * box_b.l))
        xmax = np.max((box_a.x + 0.5 * box_a.l, box_b.x + 0.5 * box_b.l))

        ymin = np.min((box_a.y - 0.5 * box_a.w, box_b.y - 0.5 * box_b.w))
        ymax = np.max((box_a.y + 0.5 * box_a.w, box_b.y + 0.5 * box_b.w))

        zmin = np.min((box_a.z - 0.5 * box_a.h, box_b.z - 0.5 * box_b.h))
        zmax = np.max((box_a.z + 0.5 * box_a.h, box_b.z + 0.5 * box_b.h))

        c = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])

        return iou - (d * d) / (c * c)

    def iou_sampling(self, num_samples=10000):
        """Computes intersection over union by sampling points.

        Generate n samples inside each box and check if those samples are inside
        the other box. Each box has a different volume, therefore the number o
        samples in box1 is estimating a different volume than box2. To address
        this issue, we normalize the iou estimation based on the ratio of the
        volume of the two boxes.

        Args:
          num_samples: Number of generated samples in each box

        Returns:
          IoU Estimate (float)
        """
        p1 = [self._box1.sample() for _ in range(num_samples)]
        p2 = [self._box2.sample() for _ in range(num_samples)]
        box1_volume = self._box1.volume
        box2_volume = self._box2.volume
        box1_intersection_estimate = 0
        box2_intersection_estimate = 0
        for point in p1:
            if self._box2.inside(point):
                box1_intersection_estimate += 1
        for point in p2:
            if self._box1.inside(point):
                box2_intersection_estimate += 1
        # We are counting the volume of intersection twice.
        intersection_volume_estimate = (
                                               box1_volume * box1_intersection_estimate +
                                               box2_volume * box2_intersection_estimate) / 2.0
        union_volume_estimate = (box1_volume * num_samples + box2_volume *
                                 num_samples) - intersection_volume_estimate
        iou_estimate = intersection_volume_estimate / union_volume_estimate
        return iou_estimate

    @classmethod
    def box_scaled_axis_aligned_vertices(self, scale):
        """Returns an axis-aligned set of verticies for a box of the given scale.

        Args:
          scale: A 3*1 vector, specifiying the size of the box in x-y-z dimension.
        """
        w = scale[0] / 2.
        h = scale[1] / 2.
        d = scale[2] / 2.

        # Define the local coordinate system, w.r.t. the center of the box
        aabb = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                         [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                         [+w, +h, +d]])
        return aabb

    @classmethod
    def box_fit(cls, vertices):
        """Estimates a box 9-dof parameters from the given vertices.

        Directly computes the scale of the box, then solves for orientation and
        translation.

        Args:
          vertices: A 9*3 array of points. Points are arranged as 1 + 8 (center
            keypoint + 8 box vertices) matrix.

        Returns:
          orientation: 3*3 rotation matrix.
          translation: 3*1 translation vector.
          scale: 3*1 scale vector.
        """
        """
        box corner order is like follows
                1 -------- 0 		 top is bottom because y direction is negative
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
  
        rect/ref camera coord:
        right x, down y, front z
        nirit
        """
        EDGES = (
            [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
            [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
            [1, 2], [3, 4], [5, 6], [7, 8]  # lines along z-axis
        )
        NUM_KEYPOINTS = 9

        orientation = np.identity(3)
        translation = np.zeros((3, 1))
        scale = np.zeros(3)

        # The scale would remain invariant under rotation and translation.
        # We can safely estimate the scale from the oriented box.
        for axis in range(3):
            for edge_id in range(4):
                # The edges are stored in quadruples according to each axis
                begin, end = EDGES[axis * 4 + edge_id]
                scale[axis] += np.linalg.norm(vertices[begin, :] - vertices[end, :])
            scale[axis] /= 4.

        x = cls.box_scaled_axis_aligned_vertices(scale)
        system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
        solution, _, _, _ = optimizer(system, vertices, rcond=None)
        orientation = solution[:3, :3].T
        translation = solution[3, :3]
        return orientation, translation, scale

    def box_transformation(self, box):
        corners = Box3D.box2corners3d_camcoord(box)  # 8 x 3

        vertices = np.vstack((np.array([box.x, box.y, box.z]), corners))
        #  vertices: A 9*3 array of points. Points are arranged as 1 + 8 (center
        #           keypoint + 8 box vertices) matri

        rotation, translation, scale = self.box_fit(vertices)
        transformation = np.identity(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        return transformation

    def isinside(self, box, point):
        """Tests whether a given point is inside the box.

          Brings the 3D point into the local coordinate of the box. In the local
          coordinate, the looks like an axis-aligned bounding box. Next checks if
          the box contains the point.
        Args:
          point: A 3*1 numpy vector.

        Returns:
          True if the point is inside the box, False otherwise.
        """
        corners = Box3D.box2corners3d_camcoord(box)  # 8x3
        x_min = np.min(corners[:, 0])
        x_max = np.max(corners[:, 0])

        y_min = np.min(corners[:, 1])
        y_max = np.max(corners[:, 1])

        z_min = np.min(corners[:, 2])
        z_max = np.max(corners[:, 2])

        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max and z_min <= point[2] <= z_max:
            return True

        return False

    def _compute_intersection_points(self, box_src, box_template):
        """Computes the intersection of two boxes."""

        """
        box corner order is like follows
                1 -------- 0 		 top is bottom because y direction is negative
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
    
        rect/ref camera coord:
        right x, down y, front z
        """

        FACES = np.array([
            [5, 6, 8, 7],  # +x on yz plane
            [1, 3, 4, 2],  # -x on yz plane
            [3, 7, 8, 4],  # +y on xz plane = top
            [1, 2, 6, 5],  # -y on xz plane
            [2, 4, 8, 6],  # +z on xy plane = front
            [1, 5, 7, 3],  # -z on xy plane
        ])

        NUM_KEYPOINTS = 9

        src_corners = Box3D.box2corners3d_camcoord(box_src)  # 8 x 3
        src_vertices = np.vstack((np.array([box_src.x, box_src.y, box_src.z]), src_corners))

        template_corners = Box3D.box2corners3d_camcoord(box_template)  # 8 x 3
        template_vertices = np.vstack((np.array([box_template.x, box_template.y, box_template.z]), template_corners))

        for face in range(len(FACES)):
            indices = FACES[face, :]
            poly = [template_vertices[indices[i], :] for i in range(4)]
            clip = self.intersect_box_poly(src_vertices, poly)
            for point in clip:
                # Transform the intersection point back to the world coordinate
                # point_w = np.matmul(box_src.rotation, point) + box_src.translation
                self._intersection_points.append(point)  # point_w)

        for point_id in range(NUM_KEYPOINTS):
            v = template_vertices[point_id, :]
            if self.isinside(box_src, v):
                # point_w = np.matmul(box_src.rotation, v) + box_src.translation
                self._intersection_points.append(v)  # point_w)

    def intersect_box_poly(self, box_vertices, poly):
        """Clips the polygon against the faces of the axis-aligned box."""
        for axis in range(3):
            poly = self._clip_poly(poly, box_vertices[1, :], 1.0, axis)
            poly = self._clip_poly(poly, box_vertices[8, :], -1.0, axis)
        return poly

    def _clip_poly(self, poly, plane, normal, axis):
        """Clips the polygon with the plane using the Sutherland-Hodgman algorithm.

        See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of
        the Sutherland-Hodgman algorithm. Here we adopted a robust implementation
        from "Real-Time Collision Detection", by Christer Ericson, page 370.

        Args:
          poly: List of 3D vertices defining the polygon.
          plane: The 3D vertices of the (2D) axis-aligned plane.
          normal: normal
          axis: A tuple defining a 2D axis.

        Returns:
          List of 3D vertices of the clipped polygon.
        """

        _PLANE_THICKNESS_EPSILON = 0.000001
        _POINT_IN_FRONT_OF_PLANE = 1
        _POINT_ON_PLANE = 0
        _POINT_BEHIND_PLANE = -1

        # The vertices of the clipped polygon are stored in the result list.
        result = []
        if len(poly) <= 1:
            return result

        # polygon is fully located on clipping plane
        poly_in_plane = True

        # Test all the edges in the polygon against the clipping plane.
        for i, current_poly_point in enumerate(poly):
            prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
            d1 = self._classify_point_to_plane(prev_poly_point, plane, normal, axis)
            d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                               axis)
            if d2 == _POINT_BEHIND_PLANE:
                poly_in_plane = False
                if d1 == _POINT_IN_FRONT_OF_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)
            elif d2 == _POINT_IN_FRONT_OF_PLANE:
                poly_in_plane = False
                if d1 == _POINT_BEHIND_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)

                result.append(current_poly_point)
            else:
                if d1 != _POINT_ON_PLANE:
                    result.append(current_poly_point)

        if poly_in_plane:
            return poly
        else:
            return result

    def _intersect(self, plane, prev_point, current_point, axis):
        """Computes the intersection of a line with an axis-aligned plane.

        Args:
          plane: Formulated as two 3D points on the plane.
          prev_point: The point on the edge of the line.
          current_point: The other end of the line.
          axis: A tuple defining a 2D axis.

        Returns:
          A 3D point intersection of the poly edge with the plane.
        """
        alpha = (current_point[axis] - plane[axis]) / (
                current_point[axis] - prev_point[axis])
        # Compute the intersecting points using linear interpolation (lerp)
        intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
        return intersection_point

    def _inside(self, plane, point, axis):
        """Check whether a given point is on a 2D plane."""
        # Cross products to determine the side of the plane the point lie.
        x, y = axis
        u = plane[0] - point
        v = plane[1] - point

        a = u[x] * v[y]
        b = u[y] * v[x]
        return a >= b

    def _classify_point_to_plane(self, point, plane, normal, axis):
        """Classify position of a point w.r.t the given plane.

        See Real-Time Collision Detection, by Christer Ericson, page 364.

        Args:
          point: 3x1 vector indicating the point
          plane: 3x1 vector indicating a point on the plane
          normal: scalar (+1, or -1) indicating the normal to the vector
          axis: scalar (0, 1, or 2) indicating the xyz axis

        Returns:
          Side: which side of the plane the point is located.
        """
        _PLANE_THICKNESS_EPSILON = 0.000001
        _POINT_IN_FRONT_OF_PLANE = 1
        _POINT_ON_PLANE = 0
        _POINT_BEHIND_PLANE = -1

        signed_distance = normal * (point[axis] - plane[axis])
        if signed_distance > _PLANE_THICKNESS_EPSILON:
            return _POINT_IN_FRONT_OF_PLANE
        elif signed_distance < -_PLANE_THICKNESS_EPSILON:
            return _POINT_BEHIND_PLANE
        else:
            return _POINT_ON_PLANE

    @property
    def intersection_points(self):
        return self._intersection_points
