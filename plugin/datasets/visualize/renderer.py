import os.path as osp
import os
import av2.geometry.interpolate as interp_utils
import numpy as np
import copy
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use('agg') # prevent memory leak for drawing figures in a loop

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def points_ego2img(pts_ego, extrinsics, intrinsics):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = extrinsics @ pts_ego_4d.T
    
    uv = (intrinsics @ pts_cam_4d[:3, :]).T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def draw_polyline_ego_on_img(polyline_ego, img_bgr, extrinsics, intrinsics, color_bgr, thickness):
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego)
    
    uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    uv = np.round(uv[is_valid_points]).astype(np.int32)

    draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)


COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (0, 0, 255),
    'boundary': (0, 255, 0),
    'ped_crossing': (255, 0, 0),
    'centerline': (51, 183, 255),
    'drivable_area': (171, 255, 255)
}

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
}

CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    ]
CAM_NAMES_NUSC = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

class Renderer(object):
    """Render map elements on image views.

    Args:
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        dataset (str): 'av2' or 'nusc'
    """

    def __init__(self, cat2id, roi_size, dataset='av2'):
        self.roi_size = roi_size
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        if dataset == 'av2':
            self.cam_names = CAM_NAMES_AV2
        else:
            self.cam_names = CAM_NAMES_NUSC

    def render_bev_from_vectors(self, vectors, out_dir, draw_scores=False, specified_path=None,
            id_info=None):
        '''Render bev segmentation using vectorized map elements.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            out_dir (str): output directory
        '''

        car_img = Image.open('resources/car.png')
        #car_img = Image.open('resources/car_lidar_coord.png')
        if specified_path:
            map_path = specified_path
        else:
            map_path = os.path.join(out_dir, 'map.jpg')

        fig = plt.figure(figsize=(self.roi_size[0], self.roi_size[1]))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-self.roi_size[0] / 2, self.roi_size[0] / 2)
        ax.set_ylim(-self.roi_size[1] / 2, self.roi_size[1] / 2)
        ax.axis('off')
        #ax.imshow(car_img, extent=[-2.0, 2.0, -2.5, 2.5])
        ax.imshow(car_img, extent=[-2.5, 2.5, -2.0, 2.0])

        for label, vector_list in vectors.items():
            cat = self.id2cat[label]
            color = COLOR_MAPS_PLT[cat]
            for vec_i, vector in enumerate(vector_list):
                if draw_scores:
                    vector, score, prop = vector
                if isinstance(vector, list):
                    vector = np.array(vector)
                    from shapely.geometry import LineString
                    vector = np.array(LineString(vector).simplify(0.2).coords)
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
                #     scale_units='xy', scale=1)
                # for i in range(len(x)):
                ax.plot(x, y, 'o-', color=color, linewidth=20, markersize=50)
                if draw_scores:
                    #print('Prop:', prop, 'Label:', label)
                    if prop:
                        p = 'p'
                    else:
                        p = ''
                    score = round(score, 2)
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], str(score)+p, fontsize=100, color=color)
                if id_info:
                    vec_id = id_info[label][vec_i]
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], f'{cat[:1].upper()}{vec_id}', fontsize=100, color=color)
                    
        # Save and properly close the figure to free memory.
        # ``plt.clf`` only clears the current figure contents but the Figure
        # object (and its large canvas) remains alive in Matplotlib's global
        # state.  Over hundreds of scenes this results in significant memory
        # growth eventually leading to the process being *Killed* by the
        # operating system (OOM).  Using ``plt.close(fig)`` fully releases
        # the resources associated with this figure.

        # plt.savefig(map_path, bbox_inches='tight', dpi=40)
        fig.savefig(map_path, bbox_inches='tight', dpi=20)

        # Free memory held by the figure.
        plt.close(fig)
        
    def render_camera_views_from_vectors(self, vectors, imgs, extrinsics, 
            intrinsics, ego2cams, thickness, out_dir):
        '''Project vectorized map elements to camera views.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            imgs (tensor): images in bgr color.
            extrinsics (array): ego2img extrinsics, shape (4, 4)
            intrinsics (array): intrinsics, shape (3, 3) 
            thickness (int): thickness of lines to draw on images.
            out_dir (str): output directory
        '''

        for i in range(len(imgs)):
            img = imgs[i]
            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]
            ego2cam = ego2cams[i]
            img_bgr = copy.deepcopy(img)

            for label, vector_list in vectors.items():
                cat = self.id2cat[label]
                color = COLOR_MAPS_BGR[cat]
                for vector in vector_list:
                    img_bgr = np.ascontiguousarray(img_bgr)
                    if isinstance(vector, list):
                        vector = np.array(vector)
                    draw_polyline_ego_on_img(vector, img_bgr, ego2cam, intrinsic, color, thickness)
                    
            out_path = osp.join(out_dir, self.cam_names[i]) + '.jpg'
            cv2.imwrite(out_path, img_bgr)

    def render_bev_from_mask(self, semantic_mask, out_dir, flip=False):
        '''Render bev segmentation from semantic_mask.
        
        Args:
            semantic_mask (array): semantic mask.
            out_dir (str): output directory
        '''

        
        if len(semantic_mask.shape) == 3:
            c, h, w = semantic_mask.shape
        else:
            h, w = semantic_mask.shape
        
        bev_img = np.ones((3, h, w), dtype=np.uint8) * 255
        if 'drivable_area' in self.cat2id:
            drivable_area_mask = semantic_mask[self.cat2id['drivable_area']]
            bev_img[:, drivable_area_mask == 1] = \
                    np.array(COLOR_MAPS_BGR['drivable_area']).reshape(3, 1)
        
        for label in self.id2cat:
            cat = self.id2cat[label]
            if cat == 'drivable_area':
                continue
            if len(semantic_mask.shape) == 3:
                valid = (semantic_mask[label] == 1)
            else:
                valid = semantic_mask == (label + 1)
            bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)

        #for label in range(c):
        #    cat = self.id2cat[label]
        #    if cat == 'drivable_area':
        #        continue
        #    mask = semantic_mask[label]
        #    valid = mask == 1
        #    bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)

        out_path = osp.join(out_dir, 'semantic_map.jpg')
        if flip:
            bev_img_flipud = np.array([np.flipud(i) for i in bev_img], dtype=np.uint8)
            cv2.imwrite(out_path, bev_img_flipud.transpose((1, 2, 0)))
        else:
            cv2.imwrite(out_path, bev_img.transpose((1, 2, 0)))
            
        