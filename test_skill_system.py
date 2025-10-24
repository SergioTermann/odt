#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能系统测试脚本
测试oneVSoneEnv_ap环境中的activate_skill函数集成
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'gym_dogfight'))

try:
    from gym_dogfight.envs.dogfightEnv.oneVSoneEnv_ap import oneVSoneEnv_ap
except ImportError as e:
    print(f"导入环境失败: {e}")
    print("请确保gym_dogfight包已正确安装")
    sys.exit(1)

def test_skill_system():
    """测试技能系统"""
    print("=== 技能系统测试 ===")
    
    try:
        # 创建环境实例
        env = oneVSoneEnv_ap(rendering=False)
        print("✓ 环境创建成功")
        
        # 检查动作空间维度
        print(f"动作空间维度: {env.action_space.shape}")
        print(f"动作空间范围: low={env.action_space.low}, high={env.action_space.high}")
        
        # 检查观测空间维度
        print(f"观测空间维度: {env.observation_space.shape}")
        
        # 重置环境
        obs = env.reset()
        print(f"✓ 环境重置成功，观测维度: {obs.shape}")
        
        # 测试不同技能激活
        test_actions = [
            [0.5, 0.5, 0.5, 0.0, 0.9],  # 激活爬升技能
            [0.5, 0.5, 0.5, 0.0, 0.7],  # 激活俯冲技能
            [0.5, 0.5, 0.5, 0.0, 0.5],  # 激活转弯技能
            [0.5, 0.5, 0.5, 0.0, 0.3],  # 激活追踪技能
            [0.5, 0.5, 0.5, 0.0, 0.1],  # 无技能激活
        ]
        
        skill_names = ["爬升", "俯冲", "转弯", "追踪", "无技能"]
        
        for i, (action, skill_name) in enumerate(zip(test_actions, skill_names)):
            print(f"\n--- 测试 {skill_name} 技能 ---")
            print(f"动作值: {action}")
            
            # 执行动作
            obs, reward, done, info = env.steps([action])
            
            # 检查技能状态
            skill_type = obs[-2]  # 倒数第二个元素是技能类型
            skill_duration = obs[-1]  # 最后一个元素是技能持续时间
            
            print(f"当前技能类型编码: {skill_type}")
            print(f"技能剩余持续时间: {skill_duration}")
            print(f"环境当前技能: {env.current_skill}")
            print(f"技能持续时间计数器: {env.skill_duration}")
            
            if done:
                print("环境已结束，重置环境")
                obs = env.reset()
        
        print("\n=== 技能系统测试完成 ===")
        print("✓ 所有测试通过")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_skill_functions():
    """测试技能函数"""
    print("\n=== 技能函数测试 ===")
    
    try:
        env = oneVSoneEnv_ap(rendering=False)
        
        # 测试各个技能函数
        skills_to_test = [
            ('climb', {'intensity': 0.8, 'duration': 30}),
            ('dive', {'intensity': 0.7, 'duration': 25}),
            ('turn', {'turn_rate': 45, 'duration': 20}),
            ('track', {'duration': 60})
        ]
        
        for skill_name, params in skills_to_test:
            print(f"\n测试 {skill_name} 技能函数...")
            result = env.activate_skill(skill_name, **params)
            print(f"激活结果: {result}")
            print(f"当前技能: {env.current_skill}")
            print(f"技能持续时间: {env.skill_duration}")
            print(f"技能参数: {getattr(env, 'skill_params', {})}")
        
        print("\n✓ 技能函数测试完成")
        
    except Exception as e:
        print(f"✗ 技能函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("开始技能系统集成测试...\n")
    
    # 测试技能函数
    if not test_skill_functions():
        sys.exit(1)
    
    # 测试技能系统集成
    if not test_skill_system():
        sys.exit(1)
    
    print("\n🎉 所有测试通过！技能系统已成功集成到环境中。")
    print("\n使用说明:")
    print("- 动作空间现在是5维: [推力, 高度变化, 航向偏移, 导弹发射, 技能激活]")
    print("- 技能激活值范围: 0.8-1.0=爬升, 0.6-0.8=俯冲, 0.4-0.6=转弯, 0.2-0.4=追踪")
    print("- 观测空间增加了2维: 技能类型编码和剩余持续时间")
    print("- 技能会在指定持续时间后自动结束")