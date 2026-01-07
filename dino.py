import cv2
import numpy as np

def segment_color_patch(img_rgb, target_color_name):

    if img_rgb is None:
        return None, None
    
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    target = target_color_name.lower()
    final_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

    if "red" in target:
        lower1 = np.array([0, 130, 90])
        upper1 = np.array([12, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower1, upper1)
        
        lower2 = np.array([170, 130, 90])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(img_hsv, lower2, upper2)
        
        final_mask = cv2.bitwise_or(mask1, mask2)
        
    elif "blue" in target:

        lower = np.array([0, 0, 70]) 
        upper = np.array([180, 135, 240]) 
        
        final_mask = cv2.inRange(img_hsv, lower, upper)
        
    elif "black" in target or "ink" in target:

        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 180]) 
        final_mask = cv2.inRange(img_hsv, lower, upper)

    kernel = np.ones((3,3), np.uint8) 
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        contour_mask = np.zeros_like(final_mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        final_mask = cv2.bitwise_and(final_mask, contour_mask)

    final_mask = cv2.erode(final_mask, kernel, iterations=2)

    return img_rgb, final_mask

def calculate_metrics(vis_rgb, ir_gray=None, uv_gray=None, mask=None):

    results = {}
    
    if mask is None or np.count_nonzero(mask) == 0:
        return {'ir_score': 0, 'uv_score': 0}

    vis_gray = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2GRAY)
    ink_vis = np.median(vis_gray[mask > 0])
    
    bg_mask = cv2.bitwise_not(mask)
    
    if ir_gray is not None:
        ink_ir = np.median(ir_gray[mask > 0])
        bg_ir = np.median(ir_gray[bg_mask > 0])
        
        vis_contrast = (np.median(vis_gray[bg_mask > 0]) - ink_vis) + 1e-5
        ir_contrast = (bg_ir - ink_ir)
        contrast_retention = np.clip(ir_contrast / vis_contrast, 0, 1)
        results['ir_score'] = 1.0 - contrast_retention
    else:
        results['ir_score'] = 0.0

    if uv_gray is not None:
        ink_uv = np.median(uv_gray[mask > 0])
        bg_uv = np.median(uv_gray[bg_mask > 0])
        
        results['uv_score'] = (ink_uv - bg_uv) / (bg_uv + 1e-5)
    else:
        results['uv_score'] = -1.0 
        
    return results