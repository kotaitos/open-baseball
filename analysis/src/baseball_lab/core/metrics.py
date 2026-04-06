import math
from typing import Dict

import numpy as np


def calculate_distance_3d(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """3D空間における2点間のユークリッド距離を計算します"""
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + (p1["z"] - p2["z"]) ** 2
    )


def calculate_angle_3d(
    p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float]
) -> float:
    """
    3点からなる角度を計算します (p2が頂点)
    0〜180度の範囲で返します
    """
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"], p1["z"] - p2["z"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"], p3["z"] - p2["z"]])

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # 計算誤差で範囲外になるのを防ぐ

    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def calculate_rotation(p_left: Dict[str, float], p_right: Dict[str, float]) -> float:
    """
    左右の点（肩や腰）の座標から、カメラ平面に対する捻り角度を推定します。
    Z座標の差分とX座標の差分から角度を計算します。（正面向いて0度、真横向いて90度）
    """
    dx = p_right["x"] - p_left["x"]
    dz = p_right["z"] - p_left["z"]

    if dx == 0 and dz == 0:
        return 0.0

    angle_rad = math.atan2(abs(dz), abs(dx))
    return math.degrees(angle_rad)


def calculate_distance_2d(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """X-Y平面における2点間のユークリッド距離を計算します"""
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def calculate_speed_2d(
    p_curr: Dict[str, float], p_prev: Dict[str, float], fps: float
) -> float:
    """2D平面での1フレーム間の移動距離から速度を計算します"""
    dist = calculate_distance_2d(p_curr, p_prev)
    return dist * fps
