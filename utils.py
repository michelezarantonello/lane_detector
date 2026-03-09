import typing as tp
import numpy as np
import cv2 as cv


# definitions on ausiliary functions

# function to perform the conversion between polar and cartesian coordinates
def polar2cartesian(radius: np.ndarray, angle: np.ndarray) -> tp.Tuple[np.ndarray]:
    return radius * np.array([np.cos(angle), np.sin(angle)]), np.array([np.sin(angle), -np.cos(angle)])
# funtion to add lines to an image
def draw_lines(img: np.ndarray, lines: np.ndarray, color: tp.List[int] = [0, 0, 255], thickness: int = 1) -> tp.Tuple[np.ndarray]:
    new_image = np.copy(img)
    empty_image = np.zeros(img.shape[:2])

    if len(lines.shape) == 1:
        lines = lines[None, ...]

    # draw found lines
    for rho, theta in lines:
        cx, direction = polar2cartesian(rho, theta)
        pt1 = np.round(cx + 10000*direction).astype(int)
        pt2 = np.round(cx - 10000*direction).astype(int)
        empty_image = cv.line(img=empty_image,pt1=pt1, pt2=pt2, color=255, thickness=thickness)

    # keep lower part of each line until intersection
    mask_lines = empty_image != 0
    max_line = 0
    valid = flag = False # check that we found 2 lines at least once before 
    for i, line in enumerate(mask_lines): # iterate each line and search for a batch of contiguous idxs
        indices = np.argwhere(line)
        if len(indices) > 1:
            flag = True
            for ii in range(len(indices)-1):
                flag = flag and indices[ii+1] == indices[ii] + 1
                if not flag:
                    valid = True 
                    break
                if valid:
                    max_line = i
        elif len(indices) == 1 and valid:
            max_line = i
        if flag and valid:
            break

    mask_boundaries = np.zeros_like(empty_image)
    mask_boundaries[max_line:] = 1
    mask = (mask_lines * mask_boundaries).astype(bool)

    new_image[mask_lines] = np.array(color)
    
    return new_image, mask

# function that given the image and the mask of the lines, fill the area between the lines
def fill_lines(img: np.ndarray, mask: np.ndarray, color: tp.List[int] = [0, 0, 255]) -> np.ndarray:
    border = np.where(mask)

    possible_vertex = np.where(border[0] == np.min(border[0]))
    vertex = np.array([border[0][int(len(possible_vertex[0]) / 2)], border[1][int(len(possible_vertex[0]) / 2)]])[::-1]

    bottom_pos = [np.min(np.where(border[1] == np.min(border[1]))), np.max(np.where(border[1] == np.max(border[1])))]
    bottom_left = np.array([border[0][bottom_pos[0]], border[1][bottom_pos[0]]])[::-1]
    bottom_right = np.array([border[0][bottom_pos[1]], border[1][bottom_pos[1]]])[::-1]
    points = np.array([vertex, bottom_left, bottom_right])

    return cv.fillConvexPoly(np.copy(img), points=points, color=color)

def fill_between_lines(image, lines1, lines2, color=[0,0,255]):
    img_filled = image.copy()
    h, w = image.shape[:2]

    # Calcola i punti estremi di ciascuna linea
    points = []

    for rho, theta in lines1:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        points.append(pt1)
        points.append(pt2)

    for rho, theta in lines2:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        points.append(pt1)
        points.append(pt2)

    # Ordina i punti per formare un poligono convesso
    points = np.array(points)
    hull = cv.convexHull(points)

    # Riempi il poligono
    cv.fillConvexPoly(img_filled, hull, color)

    return img_filled

def cluster_lines(lines, rho_thresh=10, theta_thresh=np.deg2rad(15)):
    clusters = []

    for rho, theta in lines:
        for i, cluster in enumerate(clusters):
            rho_c = cluster[:, 0].mean()
            theta_c = cluster[:, 1].mean()

            if abs(rho - rho_c) < rho_thresh and abs(theta - theta_c) < theta_thresh:
                clusters[i] = np.vstack([cluster, [rho, theta]])
                break
        else:
            clusters.append(np.array([[rho, theta]]))

    return clusters


def pointsum_in_line(line, edges, tol=2):
    ys, xs = np.where(edges > 0)
    dist = np.abs(xs*np.cos(line[1]) + ys*np.sin(line[1]) - line[0])
    return np.sum(dist < tol)

def points_in_line(line, edges, tol=2):
    ys, xs = np.where(edges > 0)
    rho, theta = line
    dist = np.abs(xs*np.cos(theta) + ys*np.sin(theta) - rho)
    mask = dist < tol
    xs_line = xs[mask]
    ys_line = ys[mask]
    t = xs_line * (-np.sin(theta)) + ys_line * np.cos(theta)
    t_sorted = np.sort(t)
    gaps = np.diff(t_sorted)
    return gaps

def is_road_line(line, image, tol=7, offset=15, min_delta_L=12):
    """
    Simple road line validation using relative LAB lightness.
    """

    rho, theta = line
    h, w = image.shape[:2]

    # --- 1. Sample pixels close to the line
    ys, xs = np.indices((h, w))
    dist = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho)
    line_mask = dist < tol

    xs_line = xs[line_mask]
    ys_line = ys[line_mask]

    if len(xs_line) < 50:
        return False

    # --- 2. Convert to LAB
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    L_line = lab[ys_line, xs_line, 0].astype(np.float32)

    # --- 3. Sample background pixels (above and below the line)
    nx = np.cos(theta)
    ny = np.sin(theta)

    xs_bg1 = (xs_line + offset * nx).astype(int)
    ys_bg1 = (ys_line + offset * ny).astype(int)

    xs_bg2 = (xs_line - offset * nx).astype(int)
    ys_bg2 = (ys_line - offset * ny).astype(int)

    # keep only valid pixels
    valid1 = (xs_bg1 >= 0) & (xs_bg1 < w) & (ys_bg1 >= 0) & (ys_bg1 < h)
    valid2 = (xs_bg2 >= 0) & (xs_bg2 < w) & (ys_bg2 >= 0) & (ys_bg2 < h)

    L_bg = np.concatenate([
        lab[ys_bg1[valid1], xs_bg1[valid1], 0],
        lab[ys_bg2[valid2], xs_bg2[valid2], 0]
    ]).astype(np.float32)

    if len(L_bg) == 0:
        return False

    # --- 4. Relative brightness test
    delta_L = np.mean(L_line) - np.mean(L_bg)

    return delta_L > min_delta_L



def is_dash(line, edges, min_points=250):
    print(pointsum_in_line(line, edges, tol=2))
    if pointsum_in_line(line, edges, tol=2) < min_points:
        return True
    else:
        return False