#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data analysis module for Chinese Text Correction task.
This module provides functions for analyzing error patterns in the dataset.
"""

import re
import json
import numpy as np
import difflib
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

# Try to import optional dependencies for visualization
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")

def find_differences(str1, str2):
    differ = difflib.ndiff(str1, str2)
    changes = [char[0]+char[2] for char in differ if char[0] in ('-', '+')]
    return changes

def analyze_character_confusion(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the dataset to extract statistics and error patterns.

    Args:
        data: List of dictionaries containing the data.

    Returns:
        Dictionary containing analysis results.
    """
    error_counter = Counter()
    for instance in data:
        if instance['label'] == 1:
            diff = find_differences(instance['source'], instance['target'])
            # diff[0]表示错误的文字，diff[1]表示正确的文字    
            if len(diff) > 2:
                error_message = "其他"
                error_counter[error_message] += 1
            elif len(diff) == 1:
                error_message = diff[0]
                error_counter[error_message] += 1
            elif len(diff) == 2:
                error_message = diff[0] + diff[1]
                error_counter[error_message] += 1
        
    return dict(error_counter)

def analyze_word_confusion(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Analyze the dataset to extract word-level confusion patterns.

    Args:
        data: List of dictionaries containing the data.

    Returns:
        Dictionary with word confusion pairs and their frequency.
    """
    word_confusion_counter = Counter()

    for instance in data:
        if instance['label'] == 1:
            source = instance['source']
            target = instance['target']

            matcher = difflib.SequenceMatcher(None, source, target)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ('replace', 'delete', 'insert'):
                    wrong = source[i1:i2]
                    correct = target[j1:j2]
                    
                    # 限定长度避免字符混用影响（我们只看2字及以上的混用）
                    if len(wrong) >= 2 and len(correct) >= 2:
                        confusion_pair = f"{wrong} -> {correct}"
                        word_confusion_counter[confusion_pair] += 1
    print(dict(word_confusion_counter))
    return dict(word_confusion_counter)

def visualize_character_confusion(analysis_results: Dict[str, Any]) -> None:
    """
    Visualize the error distribution from analysis results.

    Args:
        analysis_results: Dictionary containing analysis results.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot visualize results: matplotlib not available.")
        return

    plt.rcParams['font.sans-serif'] = ['Songti SC'] # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False

    analysis_results = {k:v for k,v in analysis_results.items() if v > 1}

    sorted_items = sorted(analysis_results.items(), key=lambda x: x[1], reverse=True)
    sorted_keys, sorted_values = zip(*sorted_items)

    plt.figure(figsize=(14, 8))
    plt.bar(sorted_keys, sorted_values)
    plt.xlabel('Error Type')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.xticks(rotation=270)
    plt.savefig('错误数据分布直方图（字混淆）.png')
    plt.show()

def visualize_word_confusion(analysis_results: Dict[str, int]) -> None:
    """
    Visualize word-level confusion patterns.

    Args:
        analysis_results: Dictionary with word confusion pair statistics.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot visualize results: matplotlib not available.")
        return

    # 过滤低频词混用
    filtered_results = {k: v for k, v in analysis_results.items() if v >= 1}

    if not filtered_results:
        print("没有足够高频的词混用对可视化")
        return

    sorted_items = sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)

    plt.rcParams['font.sans-serif'] = ['Songti SC']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(14, 8))
    plt.bar(labels, values)
    plt.xlabel('Confusion Word Pair')
    plt.ylabel('Frequency')
    plt.title('Word-Level Confusion Distribution')
    plt.xticks(rotation=270)
    plt.savefig('错误数据分布直方图（词混淆）.png')
    plt.show()