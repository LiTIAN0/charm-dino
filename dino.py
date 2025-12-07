import cv2
import numpy as np

def segment_color_patch(img_rgb, target_color_name):
    """
    v5.0: 基于日志数据的精准阈值分割 + 连通区域过滤
    """
    if img_rgb is None:
        return None, None
    
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    
    target = target_color_name.lower()
    final_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    
    # ==========================================
    # 1. 精准阈值定义 (基于你的日志分析)
    # ==========================================
    
    if "red" in target:
        # 你的红色：H=9.1, S=205. 背景：H=14.6, S=150.
        # 策略：Hue 必须在 0-12 之间 (避开背景的 14)，且 Saturation 必须高
        
        # 范围 1: 0-12 (严格避开背景)
        lower1 = np.array([0, 130, 90])
        upper1 = np.array([12, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower1, upper1)
        
        # 范围 2: 170-180 (处理红色跨界)
        lower2 = np.array([170, 130, 90])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(img_hsv, lower2, upper2)
        
        final_mask = cv2.bitwise_or(mask1, mask2)
        
    elif "blue" in target:
        # 【策略 v7.0：基于 S 和 V 的物理特征定位】
        # 根据你的观察：
        # 1. S图蓝色最深 -> 寻找低饱和度 (Low Saturation)
        # 2. V图蓝色比黑色亮，比背景暗 -> 寻找中等亮度 (Medium Value)
        
        # Hue: 忽略 (0-180)，因为它已经跑偏到红色区了
        # Sat: 0 - 90 (必须足够"灰"，避开 S高的红色背景和反光墨水)
        # Val: 120 - 240 (必须比黑色亮，但不能是极亮的反光点)
        
        lower = np.array([0, 0, 70]) 
        upper = np.array([180, 135, 240]) 
        
        final_mask = cv2.inRange(img_hsv, lower, upper)
        
    elif "black" in target or "ink" in target:
        # 策略：你的黑墨水 V=152, S=200 (注意：你的黑墨水其实很红/饱和！)
        # 但它的 V 比背景 (V=253) 低很多。
        # 我们主要靠 V (亮度) 来抓取
        
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 180]) # V < 180 (背景通常 > 200)
        final_mask = cv2.inRange(img_hsv, lower, upper)

    # ==========================================
    # 2. 强力去噪：只保留最大的色块
    # ==========================================


    # 先进行形态学运算
    kernel = np.ones((3,3), np.uint8) # 稍微改小一点核，保留更多细节
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # 寻找连通区域
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 创建一个 "轮廓掩膜" (Contour Mask)
        contour_mask = np.zeros_like(final_mask)
        # 注意：这里依然填充，用来定义"最大区域在哪里"
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # 【关键修改】
        # 取交集：必须既在"最大轮廓范围内"，又必须"符合颜色阈值"
        # 这样字母中间的洞（不符合颜色阈值）就会被扣掉！
        final_mask = cv2.bitwise_and(final_mask, contour_mask)

    final_mask = cv2.erode(final_mask, kernel, iterations=2)

    return img_rgb, final_mask

def calculate_metrics(vis_rgb, ir_gray=None, uv_gray=None, mask=None):
    """
    计算 IR 透明度分数 和 UV 荧光分数
    """
    results = {}
    
    if mask is None or np.count_nonzero(mask) == 0:
        return {'ir_score': 0, 'uv_score': 0}

    # 1. 准备 VIS 的灰度图 (用于作为分母)
    vis_gray = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2GRAY)
    ink_vis = np.median(vis_gray[mask > 0])
    
    # 背景掩膜 (反转)
    bg_mask = cv2.bitwise_not(mask)
    
    # --- 2. IR 分数计算 (适用于 Black/Blue) ---
    if ir_gray is not None:
        ink_ir = np.median(ir_gray[mask > 0])
        bg_ir = np.median(ir_gray[bg_mask > 0])
        
        # 使用对比度保留率逻辑 (1.0 = 消失, 0.2 = 还在)
        vis_contrast = (np.median(vis_gray[bg_mask > 0]) - ink_vis) + 1e-5
        ir_contrast = (bg_ir - ink_ir)
        contrast_retention = np.clip(ir_contrast / vis_contrast, 0, 1)
        results['ir_score'] = 1.0 - contrast_retention
    else:
        results['ir_score'] = 0.0

    # --- 3. UV 分数计算 (适用于 Red) ---
    if uv_gray is not None:
        ink_uv = np.median(uv_gray[mask > 0])
        bg_uv = np.median(uv_gray[bg_mask > 0])
        
        # 相对背景增量 (>0 发光, <0 不发光)
        results['uv_score'] = (ink_uv - bg_uv) / (bg_uv + 1e-5)
    else:
        results['uv_score'] = -1.0 # 默认不发光
        
    return results