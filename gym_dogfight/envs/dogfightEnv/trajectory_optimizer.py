import numpy as np
import math
import json
from typing import List, Tuple, Dict, Any


class TrajectoryOptimizer:
    def __init__(self, tolerance: float = 1.0):
        self.tolerance = tolerance
        self.trajectory_data = []
        self.simplified_trajectory = []
        self.delta_encoded_data = []
    
    def add_trajectory_point(self, timestamp: float, position: Tuple[float, float, float], orientation: Tuple[float, float, float], velocity: float, additional_data: Dict[str, Any] = None):
        point = {
            'timestamp': timestamp,
            'position': position,
            'orientation': orientation,
            'velocity': velocity,
            'additional_data': additional_data or {}
        }
        self.trajectory_data.append(point)
    
    def calculate_triangle_area(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], p3: Tuple[float, float, float]) -> float:
        # 使用向量叉积计算三角形面积
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        v2 = np.array([p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]])
        cross_product = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross_product)
        return area
    
    def visvalingam_whyatt_simplify(self) -> List[Dict]:
        if len(self.trajectory_data) < 3:
            self.simplified_trajectory = self.trajectory_data.copy()
            return self.simplified_trajectory
        
        # 创建点的副本，包含有效面积信息
        points = []
        for i, point in enumerate(self.trajectory_data):
            points.append({'index': i, 'data': point, 'area': float('inf'), 'removed': False})
        
        # 计算每个中间点的有效面积
        for i in range(1, len(points) - 1):
            p1 = points[i-1]['data']['position']
            p2 = points[i]['data']['position']
            p3 = points[i+1]['data']['position']
            points[i]['area'] = self.calculate_triangle_area(p1, p2, p3)
        
        # 迭代移除面积最小的点
        while True:
            # 找到未被移除的点中面积最小的点
            min_area = float('inf')
            min_index = -1
            
            for i in range(1, len(points) - 1):
                if not points[i]['removed'] and points[i]['area'] < min_area:
                    min_area = points[i]['area']
                    min_index = i
            
            # 如果最小面积大于容差，停止简化
            if min_area > self.tolerance or min_index == -1:
                break
            points[min_index]['removed'] = True
            
            # 重新计算相邻点的面积
            for offset in [-1, 1]:
                neighbor_idx = min_index + offset
                if (1 <= neighbor_idx < len(points) - 1 and 
                    not points[neighbor_idx]['removed']):
                    
                    prev_idx = neighbor_idx - 1
                    while prev_idx >= 0 and points[prev_idx]['removed']:
                        prev_idx -= 1
                    
                    next_idx = neighbor_idx + 1
                    while next_idx < len(points) and points[next_idx]['removed']:
                        next_idx += 1
                    
                    if prev_idx >= 0 and next_idx < len(points):
                        p1 = points[prev_idx]['data']['position']
                        p2 = points[neighbor_idx]['data']['position']
                        p3 = points[next_idx]['data']['position']
                        points[neighbor_idx]['area'] = self.calculate_triangle_area(p1, p2, p3)
        
        self.simplified_trajectory = []
        for point in points:
            if not point['removed']:
                self.simplified_trajectory.append(point['data'])
        
        return self.simplified_trajectory
    
    def delta_encode(self, trajectory: List[Dict] = None) -> List[Dict]:
        if trajectory is None:
            trajectory = self.simplified_trajectory
        
        if not trajectory:
            self.delta_encoded_data = []
            return self.delta_encoded_data
        
        self.delta_encoded_data = []
        
        # 第一个点保持原样
        first_point = trajectory[0].copy()
        first_point['is_delta'] = False
        self.delta_encoded_data.append(first_point)
        
        # 后续点存储与前一点的差值
        for i in range(1, len(trajectory)):
            current = trajectory[i]
            previous = trajectory[i-1]
            
            delta_point = {
                'timestamp': current['timestamp'] - previous['timestamp'],
                'position': (
                    current['position'][0] - previous['position'][0],
                    current['position'][1] - previous['position'][1],
                    current['position'][2] - previous['position'][2]
                ),
                'orientation': (
                    self._angle_delta(current['orientation'][0], previous['orientation'][0]),
                    self._angle_delta(current['orientation'][1], previous['orientation'][1]),
                    self._angle_delta(current['orientation'][2], previous['orientation'][2])
                ),
                'velocity': current['velocity'] - previous['velocity'],
                'additional_data': self._delta_encode_dict(current['additional_data'], previous['additional_data']), 'is_delta': True}
            self.delta_encoded_data.append(delta_point)
        
        return self.delta_encoded_data
    
    def _angle_delta(self, current_angle: float, previous_angle: float) -> float:
        delta = current_angle - previous_angle
        # 处理角度跨越边界的情况
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        return delta
    
    def _delta_encode_dict(self, current_dict: Dict, previous_dict: Dict) -> Dict:
        delta_dict = {}
        
        # 处理数值类型的差值
        for key in current_dict:
            if key in previous_dict:
                if isinstance(current_dict[key], (int, float)):
                    delta_dict[key] = current_dict[key] - previous_dict[key]
                else:
                    delta_dict[key] = current_dict[key]
            else:
                delta_dict[key] = current_dict[key]
        
        # 添加新增的键
        for key in current_dict:
            if key not in previous_dict:
                delta_dict[key] = current_dict[key]
        
        return delta_dict
    
    def delta_decode(self, encoded_data: List[Dict] = None) -> List[Dict]:
        if encoded_data is None:
            encoded_data = self.delta_encoded_data
        
        if not encoded_data:
            return []
        
        decoded_data = []
        
        # 第一个点直接复制
        first_point = encoded_data[0].copy()
        first_point.pop('is_delta', None)
        decoded_data.append(first_point)
        
        # 解码后续点
        for i in range(1, len(encoded_data)):
            delta_point = encoded_data[i]
            previous_point = decoded_data[i-1]
            
            decoded_point = {
                'timestamp': previous_point['timestamp'] + delta_point['timestamp'],
                'position': (
                    previous_point['position'][0] + delta_point['position'][0],
                    previous_point['position'][1] + delta_point['position'][1],
                    previous_point['position'][2] + delta_point['position'][2]
                ),
                'orientation': (
                    self._angle_add(previous_point['orientation'][0], delta_point['orientation'][0]),
                    self._angle_add(previous_point['orientation'][1], delta_point['orientation'][1]),
                    self._angle_add(previous_point['orientation'][2], delta_point['orientation'][2])
                ),
                'velocity': previous_point['velocity'] + delta_point['velocity'],
                'additional_data': self._delta_decode_dict(delta_point['additional_data'], 
                                                         previous_point['additional_data'])
            }
            decoded_data.append(decoded_point)
        
        return decoded_data
    
    def _angle_add(self, base_angle: float, delta_angle: float) -> float:
        result = base_angle + delta_angle
        # 将角度规范化到[-180, 180]范围
        while result > 180:
            result -= 360
        while result <= -180:
            result += 360
        return result
    
    def _delta_decode_dict(self, delta_dict: Dict, previous_dict: Dict) -> Dict:
        decoded_dict = previous_dict.copy()
        
        for key, value in delta_dict.items():
            if key in previous_dict and isinstance(value, (int, float)) and isinstance(previous_dict[key], (int, float)):
                decoded_dict[key] = previous_dict[key] + value
            else:
                decoded_dict[key] = value
        
        return decoded_dict
    
    def save_to_file(self, filename: str, data_type: str = 'delta'):
        if data_type == 'original':
            data = self.trajectory_data
        elif data_type == 'simplified':
            data = self.simplified_trajectory
        elif data_type == 'delta':
            data = self.delta_encoded_data
        else:
            raise ValueError("data_type must be 'original', 'simplified', or 'delta'")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'data_type': data_type, 'tolerance': self.tolerance, 'trajectory': data}, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filename: str) -> str:
        with open(filename, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        data_type = file_data.get('data_type', 'original')
        self.tolerance = file_data.get('tolerance', 1.0)
        trajectory = file_data.get('trajectory', [])
        
        if data_type == 'original':
            self.trajectory_data = trajectory
        elif data_type == 'simplified':
            self.simplified_trajectory = trajectory
        elif data_type == 'delta':
            self.delta_encoded_data = trajectory
        
        return data_type
    
    def get_compression_stats(self) -> Dict[str, Any]:
        original_count = len(self.trajectory_data)
        simplified_count = len(self.simplified_trajectory)
        delta_count = len(self.delta_encoded_data)
        
        stats = {
            'original_points': original_count,
            'simplified_points': simplified_count,
            'delta_encoded_points': delta_count,
            'simplification_ratio': simplified_count / original_count if original_count > 0 else 0,
            'compression_ratio': simplified_count / original_count if original_count > 0 else 0,
            'tolerance': self.tolerance
        }
        
        return stats
    
    def clear(self):
        self.trajectory_data.clear()
        self.simplified_trajectory.clear()
        self.delta_encoded_data.clear()