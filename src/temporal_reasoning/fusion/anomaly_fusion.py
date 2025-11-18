# -*- coding: utf-8 -*-
"""
异常融合模块
"""

from typing import List, Dict


def fuse_multimodal_anomalies(
    aligned_anomalies: List[List[Dict]],
    multimodal_confidence_boost: float = 1.2,
    single_modality_threshold: float = 0.8
) -> List[Dict]:
    """
    融合多模态异常
    
    Args:
        aligned_anomalies: 对齐后的异常组列表
        multimodal_confidence_boost: 多模态一致时的置信度提升倍数
        single_modality_threshold: 单模态异常的最小置信度阈值
    
    Returns:
        融合后的异常列表
    """
    fused_anomalies = []
    filtered_count = 0  # 统计被过滤的异常组数
    
    for anomaly_group in aligned_anomalies:
        if not anomaly_group:
            continue
        
        # 统计各模态的异常
        modalities = [a.get('modality', 'unknown') for a in anomaly_group]
        modality_set = set(modalities)
        
        # 计算融合置信度
        confidences = [a.get('confidence', 0.0) for a in anomaly_group]
        base_confidence = max(confidences) if confidences else 0.0
        
        # 多模态一致时提升置信度
        if len(modality_set) >= 2:
            fused_confidence = min(1.0, base_confidence * multimodal_confidence_boost)
        else:
            # 单模态异常需要达到阈值
            if base_confidence < single_modality_threshold:
                filtered_count += 1
                continue  # 过滤低置信度单模态异常
            fused_confidence = base_confidence
        
        # 确定异常类型
        anomaly_type = _determine_anomaly_type(anomaly_group)
        
        # 确定严重程度
        severity = _determine_severity(fused_confidence, len(modality_set))
        
        # 使用第一个异常的时间戳和位置信息
        primary_anomaly = anomaly_group[0]
        
        # 合并所有异常的metadata（保留重要字段如object_id、class_name等）
        merged_metadata = {}
        for anomaly in anomaly_group:
            if 'metadata' in anomaly and isinstance(anomaly['metadata'], dict):
                # 优先保留第一个异常的metadata，但合并其他异常的重要字段
                if not merged_metadata:
                    merged_metadata = anomaly['metadata'].copy()
                else:
                    # 如果当前异常的metadata有object_id而merged_metadata没有，则添加
                    if 'object_id' in anomaly['metadata'] and 'object_id' not in merged_metadata:
                        merged_metadata['object_id'] = anomaly['metadata']['object_id']
                    if 'class_name' in anomaly['metadata'] and 'class_name' not in merged_metadata:
                        merged_metadata['class_name'] = anomaly['metadata']['class_name']
                    # 合并其他重要字段（如baseline_motion、motion_value等）
                    for key in ['baseline_motion', 'motion_value', 'motion_change', 
                               'hist_similarity', 'hist_diff', 'similarity_drop', 'flow_value', 'triggers']:
                        if key in anomaly['metadata'] and key not in merged_metadata:
                            merged_metadata[key] = anomaly['metadata'][key]
        
        fused_anomaly = {
            'type': anomaly_type,
            'timestamp': primary_anomaly.get('timestamp', ''),
            'frame_id': primary_anomaly.get('frame_id', 0),
            'confidence': fused_confidence,
            'description': _generate_description(anomaly_group),
            'modalities': list(modality_set),
            'severity': severity,
            'location': primary_anomaly.get('location', {}),
            'metadata': merged_metadata if merged_metadata else primary_anomaly.get('metadata', {})
        }
        
        fused_anomalies.append(fused_anomaly)
    
    if filtered_count > 0:
        print(f"[融合] 过滤了 {filtered_count} 个低置信度单模态异常组（置信度 < {single_modality_threshold}）")
    
    return fused_anomalies


def _determine_anomaly_type(anomaly_group: List[Dict]) -> str:
    """
    确定异常类型
    
    Args:
        anomaly_group: 异常组
    
    Returns:
        异常类型字符串
    """
    # 统计各类型异常
    type_counts = {}
    for anomaly in anomaly_group:
        anomaly_type = anomaly.get('type', 'unknown')
        type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
    
    # 返回最常见的类型
    if type_counts:
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    return 'unknown'


def _determine_severity(confidence: float, modality_count: int) -> str:
    """
    确定严重程度
    
    Args:
        confidence: 置信度
        modality_count: 模态数量
    
    Returns:
        严重程度字符串
    """
    if confidence >= 0.9 or modality_count >= 3:
        return 'Critical'
    elif confidence >= 0.7 or modality_count >= 2:
        return 'Moderate'
    else:
        return 'Minor'


def _generate_description(anomaly_group: List[Dict]) -> str:
    """
    生成异常描述
    
    Args:
        anomaly_group: 异常组
    
    Returns:
        描述字符串
    """
    if len(anomaly_group) == 1:
        return anomaly_group[0].get('description', '检测到异常')
    
    # 多模态异常
    modalities = set(a.get('modality', 'unknown') for a in anomaly_group)
    modality_names = {
        'motion': '运动',
        'structure': '结构',
        'physiological': '生理'
    }
    
    modality_str = '、'.join([modality_names.get(m, m) for m in modalities])
    return f"多模态异常检测：{modality_str}"

