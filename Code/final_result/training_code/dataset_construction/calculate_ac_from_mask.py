import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

def fit_ellipse_ac_mm(mask_path, show_plot=False):
    spacing = 0.28       # 每像素毫米数
    scale_factor = 1.0   # 椭圆修正因子

    # 读取 mask 并二值化
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    # 边缘 & 去线
    edges = cv2.Canny(binary, 50, 150)
    h, w = binary.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=int(min(h, w) * 0.2), maxLineGap=10)
    arc_mask = edges.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(arc_mask, (x1, y1), (x2, y2), 0, thickness=3)
        arc_mask = cv2.morphologyEx(arc_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 拟合椭圆
    contours, _ = cv2.findContours(arc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.vstack(max(contours, key=cv2.contourArea)).reshape(-1, 2)
    ellipse = cv2.fitEllipse(points.astype(np.int32))
    (xc, yc), (maj_d, min_d), angle_deg = ellipse
    a, b = maj_d / 2.0, min_d / 2.0

    # 周长与毫米单位转换（改用精度更高的第一公式）
    # perimeter = pi * (3 * (a + b) - sqrt((3 * a + b) * (a + 3 * b)))  #
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = pi * (a + b) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)))

    ac_mm = perimeter * spacing * scale_factor

    # 可视化
    if show_plot:
        image_path = mask_path.replace("masks", "images")
        raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        raw_rgb = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)

        # 创建 mask 覆盖层（绿色）
        mask_color = np.zeros_like(raw_rgb)
        mask_color[binary > 0] = [0, 255, 0]

        # 叠加 mask 到原图
        overlay = cv2.addWeighted(raw_rgb, 1.0, mask_color, 0.4, 0)

        # 叠加椭圆（红色）
        cv2.ellipse(overlay, (int(xc), int(yc)), (int(a), int(b)), angle_deg, 0, 360, (0, 0, 255), 2)

        # 显示图像
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        # plt.title("Overlay: Original + Mask + Ellipse")
        plt.title(f"fitEllipse AC = {ac_mm:.2f} mm")
        plt.axis('off')
        plt.show()

    return ac_mm

# def preprocess_and_save_smoothed_mask(path, out_path="temp_smooth_mask.png", ksize=(71, 71), sigma=3):
#     mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     mask_bin = np.where(mask == 255, 1, mask).astype(np.uint8)
#     _, binary = cv2.threshold(mask_bin, 0, 255, cv2.THRESH_BINARY)
#
#     blurred = cv2.GaussianBlur(binary, ksize, sigma)
#     _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
#
#     cv2.imwrite(out_path, smooth_mask)  # 保存处理后的 mask
#     return out_path

def preprocess_and_save_smoothed_mask(path, out_path="temp_smooth_mask.png", ksize=(71, 71), sigma=3, morph_kernel_size=5, morph_iter=1, area_threshold=500):
    # Step 1: 读取原始 mask 并进行二值化
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask_bin = np.where(mask == 255, 1, mask).astype(np.uint8)
    _, binary = cv2.threshold(mask_bin, 0, 255, cv2.THRESH_BINARY)

    # Step 2: 高斯模糊去除随机边缘噪点
    blurred = cv2.GaussianBlur(binary, ksize, sigma)
    _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Step 3: 形态学闭操作平滑边界并修补断口
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    # Step 4: 轮廓面积过滤，去除小碎块噪点
    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(smooth_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= area_threshold:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Step 5: 保存最终处理结果
    cv2.imwrite(out_path, clean_mask)
    return out_path

############################################################################################################
# from skimage.segmentation import active_contour
# from skimage.filters import sobel
# from skimage.io import imread, imsave
# from skimage import img_as_ubyte
#
# def refine_mask_with_snake(mask_path, image_path=None, out_path="refined_mask_snake.png", alpha=0.01, beta=0.1, gamma=0.01):
#     # 读取 mask
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask_bin = np.where(mask == 255, 1, mask).astype(np.uint8)
#     _, binary = cv2.threshold(mask_bin, 0, 255, cv2.THRESH_BINARY)
#
#     # 获取边界并初始化 snake 起点（用最大轮廓）
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if len(contours) == 0:
#         raise ValueError("No contours found in mask.")
#     contour = max(contours, key=cv2.contourArea)
#     snake_init = contour[:, 0, :]  # N x 2 shape
#
#     # 读取用于梯度图像（edge map）
#     if image_path is None:
#         image_path = mask_path.replace("masks_pred", "segmentation_dataset/images")
#     image = imread(image_path, as_gray=True)
#     if image.max() > 1:
#         image = image / 255.0  # 归一化
#
#     edge_map = sobel(image)
#
#     # 执行 snake 微调
#     snake = active_contour(edge_map, snake_init, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=2500, convergence=0.1)
#
#     # 绘制 snake 到 mask 上
#     refined_mask = np.zeros_like(image, dtype=np.uint8)
#     snake_int = np.round(snake).astype(np.int32)
#     cv2.fillPoly(refined_mask, [snake_int], 255)
#     imsave(out_path, img_as_ubyte(refined_mask > 0))
#
#     return out_path
############################################################################################################


def fit_ellipse_ac_mm_pred(mask_path, show_plot=False, spacing = 0.28, scale_factor = 1, epsilon=0.01):

    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_gray = np.where(mask_gray == 255, 1, mask_gray).astype(np.uint8)
    _, binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binary, 50, 150)
    h, w = binary.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=int(min(h, w) * 0.2), maxLineGap=10)

    arc_mask = edges.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if min(x1, x2, y1, y2) < 10 or max(x1, x2) > w - 10 or max(y1, y2) > h - 10:
                continue  # 忽略边缘线段
            cv2.line(arc_mask, (x1, y1), (x2, y2), 0, thickness=1)

    arc_mask = cv2.morphologyEx(arc_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    arc_mask = cv2.dilate(arc_mask, np.ones((3, 3), np.uint8), iterations=1)  # 修补断点

    # 3. 提取并平滑轮廓
    contours, _ = cv2.findContours(arc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon * cv2.arcLength(contour, True)
    points = cv2.approxPolyDP(contour, epsilon, True)
    points = np.vstack(points).reshape(-1, 2)

    ellipse = cv2.fitEllipse(points.astype(np.int32))
    (xc, yc), (maj_d, min_d), angle_deg = ellipse
    a, b = maj_d / 2.0, min_d / 2.0

    # 5. Ramanujan 周长估计
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = pi * (a + b) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)))
    ac_mm = perimeter * spacing * scale_factor

    # 6. 可视化
    if show_plot:
        image_path = mask_path.replace("masks_pred", "segmentation_dataset/images")
        raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        raw_rgb = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)

        mask_color = np.zeros_like(raw_rgb)
        mask_color[binary > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(raw_rgb, 1.0, mask_color, 0.4, 0)

        cv2.ellipse(overlay, (int(xc), int(yc)), (int(a), int(b)), angle_deg, 0, 360, (0, 0, 255), 2)

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay: Original + Mask + Fitted Ellipse")
        plt.axis('off')
        plt.show()

    return ac_mm

########################################################################################################################

########################################################################################################################


def fit_circle_ac_mm(mask_path, show_plot=True, spacing=0.28, scale_factor=0.965):
    """
    仅用 OpenCV 的椭圆拟合（cv2.fitEllipse）计算 AC（mm）
    """
    # 原图路径（可选，仅用于可视化；按需替换目录名）
    raw_path = (mask_path
                .replace("masks_pred", "segmentation_dataset/images")
                .replace("masks", "images"))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # --- 关键修复：稳健二值化 ---
    # 方案A：阈值=0，>0 即前景
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # 如果前景像素太少（可能前景是黑色，背景是白色），就反转一次再试
    if cv2.countNonZero(binary) < 10:
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)

    # 取最大外轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask after binarisation. "
                         "Check mask foreground values (should be >0).")
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        raise ValueError("Contour has fewer than 5 points; cannot fit ellipse.")

    # 拟合椭圆：((cx, cy), (MA, ma), angle)
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
    a, b = MA / 2.0, ma / 2.0  # 半轴（px）

    # Ramanujan II 周长近似（px）
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    perimeter_px = pi * (a + b) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)))

    # 像素 -> 毫米，并做标定
    ac_mm = perimeter_px * spacing * scale_factor

    if show_plot:
        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raw = np.zeros_like(mask)
        vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        # 叠加掩膜
        mask_color = np.zeros_like(vis)
        mask_color[binary > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        # 叠加椭圆
        cv2.ellipse(overlay, ((cx, cy), (MA, ma), angle), (0, 0, 255), 2)

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"fitEllipse AC = {ac_mm:.2f} mm")
        plt.axis("off")
        plt.show()

    return ac_mm




# def fit_ellipse_ac_mm_pred(mask_path, show_plot=False):
#     spacing = 0.28
#     scale_factor = 1.0
#
#     mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     mask_gray = np.where(mask_gray == 255, 1, mask_gray).astype(np.uint8)
#     _, binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
#
#     edges = cv2.Canny(binary, 50, 150)
#     h, w = binary.shape
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=int(min(h, w) * 0.2), maxLineGap=10)
#     arc_mask = edges.copy()
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(arc_mask, (x1, y1), (x2, y2), 0, thickness=3)
#         arc_mask = cv2.morphologyEx(arc_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
#
#     contours, _ = cv2.findContours(arc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     points = np.vstack(max(contours, key=cv2.contourArea)).reshape(-1, 2)
#
#     ellipse = cv2.fitEllipse(points.astype(np.int32))
#     (xc, yc), (maj_d, min_d), angle_deg = ellipse
#     a, b = maj_d / 2.0, min_d / 2.0
#
#     h = ((a - b) ** 2) / ((a + b) ** 2)
#     perimeter = pi * (a + b) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)))
#     ac_mm = perimeter * spacing * scale_factor
#
#     if show_plot:
#         image_path = mask_path.replace("masks_pred", "segmentation_dataset/images")
#         raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         raw_rgb = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
#
#         mask_color = np.zeros_like(raw_rgb)
#         mask_color[binary > 0] = [0, 255, 0]
#         overlay = cv2.addWeighted(raw_rgb, 1.0, mask_color, 0.4, 0)
#
#         cv2.ellipse(overlay, (int(xc), int(yc)), (int(a), int(b)), angle_deg, 0, 360, (0, 0, 255), 2)
#
#         plt.figure(figsize=(6, 6))
#         plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#         plt.title("Overlay: Original + Mask + Ellipse")
#         plt.axis('off')
#         plt.show()
#
#     return ac_mm


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from math import pi, sqrt
#
# def fit_ellipse_ac_mm(mask_path, show_plot=False):
#     spacing = 0.28               # 每像素毫米数
#     scale_factor = 1.10          # 仅用于遮挡补偿
#     margin = 5                   # 边界容差像素
#
#     # 读取 mask 并二值化
#     mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     _, binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
#
#     # 边缘 & 去线
#     edges = cv2.Canny(binary, 50, 150)
#     h, w = binary.shape
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
#                             minLineLength=int(min(h, w) * 0.2), maxLineGap=10)
#     arc_mask = edges.copy()
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(arc_mask, (x1, y1), (x2, y2), 0, thickness=3)
#         arc_mask = cv2.morphologyEx(arc_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
#
#     # 拟合椭圆
#     contours, _ = cv2.findContours(arc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours or len(contours[0]) < 5:
#         return None  # 无法拟合
#     points = np.vstack(max(contours, key=cv2.contourArea)).reshape(-1, 2)
#     ellipse = cv2.fitEllipse(points.astype(np.int32))
#     (xc, yc), (maj_d, min_d), angle_deg = ellipse
#     a, b = maj_d / 2.0, min_d / 2.0
#
#     # 判断是否靠近图像边界（被裁剪）
#     is_truncated = (
#         xc - a < margin or xc + a > w - margin or
#         yc - b < margin or yc + b > h - margin
#     )
#
#     # 周长估计
#     perimeter = pi * (3 * (a + b) - sqrt((3 * a + b) * (a + 3 * b)))
#     ac_mm = perimeter * spacing
#     if is_truncated:
#         ac_mm *= scale_factor
#
#     # 可视化
#     if show_plot:
#         image_path = mask_path.replace("masks", "images")
#         raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if raw_image is None:
#             raw_image = np.zeros_like(mask_gray)
#         raw_rgb = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
#
#         mask_color = np.zeros_like(raw_rgb)
#         mask_color[binary > 0] = [0, 255, 0]
#         overlay = cv2.addWeighted(raw_rgb, 1.0, mask_color, 0.4, 0)
#         cv2.ellipse(overlay, (int(xc), int(yc)), (int(a), int(b)), angle_deg, 0, 360, (0, 0, 255), 2)
#
#         plt.figure(figsize=(6, 6))
#         plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#         plt.title(f"Overlay: {'裁剪补偿' if is_truncated else '完整拟合'}")
#         plt.axis('off')
#         plt.show()
#
#     return ac_mm
