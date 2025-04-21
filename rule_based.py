#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-based corrector for Chinese Text Correction task.
This module implements rule-based methods for correcting errors in Chinese text.
"""

import re
import json
import difflib
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

# Try to import optional dependencies
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Some features will be disabled.")


class RuleBasedCorrector:
    """
    A rule-based corrector for Chinese text.
    """

    def __init__(self):
        """
        Initialize the rule-based corrector.
        """
        # Common confusion pairs (similar characters)
        self.confusion_pairs = {}

        # Punctuation rules
        self.punctuation_rules = {}

        # Grammar rules
        self.grammar_rules = {}

        # Common word pairs (for word-level correction)
        self.word_confusion = {}

        # Quantifier-noun pairs (for measure word correction)
        self.quantifier_noun_pairs = {}

        # or else

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # TODO 完成规则方法的实现，可以参考如下的方法，或者自行设计
        self._extract_confusion_pairs(train_data)
        self._extract_punctuation_rules(train_data)
        self._extract_grammar_rules(train_data)
        self._extract_word_confusion(train_data)

    def _extract_confusion_pairs(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract character confusion pairs from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # Extract character-level confusion pairs from error examples
        for sample in train_data:
            if sample['label'] == 1:  # Only for sentences with errors
                source = sample['source']
                target = sample['target']

                # For character substitution errors (when lengths are equal)
                if len(source) == len(target):
                    for i, (s_char, t_char) in enumerate(zip(source, target)):
                        if s_char != t_char:
                            # Get context (surrounding characters)
                            left_context = source[max(0, i - 2) : i]
                            right_context = source[i + 1 : min(len(source), i + 3)]
                            context = left_context + '_' + right_context

                            # Add to confusion pairs with context
                            if s_char not in self.confusion_pairs:
                                self.confusion_pairs[s_char] = defaultdict(int)
                            self.confusion_pairs[s_char][t_char] += 1

        # Filter confusion pairs to keep only the most common ones
        filtered_pairs = {}
        for wrong_char, corrections in self.confusion_pairs.items():
            # Keep only corrections that appear at least twice
            common_corrections = {correct: count for correct, count in corrections.items() if count >= 2}
            if common_corrections:
                filtered_pairs[wrong_char] = common_corrections

        # print(filtered_pairs)
        self.confusion_pairs = filtered_pairs

    def _extract_punctuation_rules(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract punctuation correction rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # TODO
        return

    def _extract_grammar_rules(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract grammar correction rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # TODO
        return

    def _extract_word_confusion(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract word-level confusion pairs from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        self.word_confusion = defaultdict(lambda: defaultdict(int))  # 嵌套字典统计

        for sample in train_data:
            if sample['label'] == 1:  # 只处理标记为错误的句子
                source = sample['source']
                target = sample['target']

                # 利用 SequenceMatcher 查找不匹配的片段
                matcher = difflib.SequenceMatcher(None, source, target)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag in ('replace', 'delete', 'insert'):
                        wrong = source[i1:i2]
                        correct = target[j1:j2]

                        # 只考虑长度至少为2的词（避免和字符混淆）
                        if len(wrong) >= 2 and len(correct) >= 2:
                            self.word_confusion[wrong][correct] += 1

        # 过滤掉出现次数过少的混淆词对
        filtered_word_confusion = {}
        for wrong_word, corrections in self.word_confusion.items():
            common = {c: count for c, count in corrections.items() if count >= 1}
            if common:
                filtered_word_confusion[wrong_word] = common

        # print(filtered_word_confusion)
        self.word_confusion = filtered_word_confusion

    def correct(self, text: str) -> str:
        """
        Apply rule-based correction to the input text.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        # Apply different correction rules in sequence
        # TODO 对应规则方法的实现，完成修正部分（可以参考如下的方法，或者自行设计）
        corrected = self._correct_punctuation(text)
        corrected = self._correct_confusion_chars(corrected)
        corrected = self._correct_grammar(corrected)
        corrected = self._correct_word_confusion(corrected)

        return corrected

    def _correct_punctuation(self, text: str) -> str:
        """
        Correct punctuation errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected punctuation.
        """
        # TODO
        return text

    def _correct_confusion_chars(self, text: str) -> str:
        """
        Correct character confusion errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected characters.
        """
        corrected_text = list(text)  # Convert to list for character-by-character editing

        # Check each character for potential confusion
        for i, char in enumerate(text):
            if char in self.confusion_pairs and self.confusion_pairs[char]:
                # Get the most common correction for this character
                correct_char = max(self.confusion_pairs[char].items(), key=lambda x: x[1])[0]

                prev_char = text[i - 1] if i > 0 else ''
                next_char = text[i + 1] if i < len(text) - 1 else ''

                # ---------- 规则：的 / 地 / 得 ----------
                if char in ['的', '地', '得']:
                    pre_2_char = text[i - 2: i] if i > 1 else ''
                    next_2_char = text[i + 1: i + 3] if i < len(text) - 2 else '' # 前两个字 和 后两个字
                    if (char != '地' and prev_char in '美快慢高低认真幸福愉快清晰熟练冷漠缓' and next_char in '走跑跳看想写读吃喝干做说叫来去进出飞打唱扔开球震') or \
                        (char != '地' and pre_2_char in ['全面', '慢慢', '有效']):
                        corrected_text[i] = '地'
                        continue
                    if char != '得' and prev_char in '跑写做说跳走唱吃喝看打学想笑哭累忙急疼困觉变获来' and next_char in '快慢清楚好多少高低早晚对厉害认真漂亮自然稳妥熟练生动准确不行':
                        corrected_text[i] = '得'
                        continue
                    if char != '的' and pre_2_char in ['重要', '城市', '侵略']:
                        corrected_text[i] = '的'
                        continue

                # ---------- 规则：在 / 再 ----------
                elif char in ['在', '再'] and correct_char in ['再', '在']:
                    if char != '再' and next_char in '次遍回见试尝问来去':
                        corrected_text[i] = '再'
                        continue
                    if char != '在' and prev_char in '正就将已也仍还正在' and next_char not in '次遍回':
                        corrected_text[i] = '在'
                        continue

                # # ---------- 规则：哪 / 那 ----------
                elif char in ['哪', '那'] and correct_char in ['哪', '那']:
                    if char != '那' and prev_char in '这和就对跟从比' and next_char in '个些位边种':
                        corrected_text[i] = '那'
                        continue

                # ---------- 规则：它 / 他 ----------
                elif char in ['它', '他'] and correct_char in ['它', '他']:
                    # “它”多用于非人事物，“他”用于人
                    if char != '它' and next_char in '咬推撞冲飞':
                        corrected_text[i] = '它'
                        continue
                    if char != '他' and next_char in '说走来去写做干打叫给唱问':
                        corrected_text[i] = '他'
                        continue
                
                elif char in ['已', '己'] and correct_char in ['已', '己']:
                    if char != '已' and next_char in '经然':
                        corrected_text[i] = '已'
                        print("nihao 已", next_char)
                        continue
                    if (char != '己' and prev_char in '自' and prev_char != '') or (char != '己' and prev_char in '自' and next_char == '的'):
                        corrected_text[i] = '己'
                        continue
                
                # ---------- 规则：像 / 象 ----------
                elif char in ['像', '象'] and correct_char in ['像', '象']:
                    if (char != '像' and next_char in '极是样个我你他她它这') or (char != '像' and prev_char in '好不'):
                        corrected_text[i] = '像'
                        continue
                    
                # ---------- 规则：做 / 作 ----------
                elif char in ['做', '作'] and correct_char in ['作', '做']:
                    if (char != '作' and next_char in '文诗画曲品家风战派秀出') or (char != '作' and prev_char in '当看用制'):
                        # print("yes")
                        corrected_text[i] = '作'
                        continue
                    if (char != '做' and next_char in '好对'):
                        corrected_text[i] = '作'
                        continue

                # ---------- 高置信度的替换 ----------
                elif self.confusion_pairs[char][correct_char] > 5:
                    corrected_text[i] = correct_char

        return ''.join(corrected_text)

    def _correct_grammar(self, text: str) -> str:
        """
        Correct grammar errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected grammar.
        """
        # TODO
        return text

    def _correct_word_confusion(self, text: str) -> str:
        """
        Correct word-level confusion errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected words.
        """
        # 通过检查数据，发现训练集中的word-level的混用的数据较少，甚至没有涵盖测试集中word-level的任意一种错误。
        # 这里我不再利用其他的先验的知识添加规则，只使用训练集中的数据进行学习，结合上面的观察，该模块对最终的结果并没有提升。
        corrected_text = text
        for wrong_word, correction_dict in self.word_confusion.items():
            if wrong_word in text:
                correct_word = max(correction_dict.items(), key=lambda x: x[1])[0]
                count = correction_dict[correct_word]

                if count >= 2 and wrong_word in corrected_text:
                    corrected_text = corrected_text.replace(wrong_word, correct_word)

        return corrected_text
