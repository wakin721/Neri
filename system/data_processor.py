# system/data_processor.py

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from collections import Counter

from system.config import INDEPENDENT_DETECTION_THRESHOLD
from system.utils import resource_path

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理类，处理图像信息集合"""

    @staticmethod
    def calculate_working_days(image_info_list: List[Dict], earliest_date: Optional[datetime]) -> List[Dict]:
        """计算每张图片的工作天数

        Args:
            image_info_list: 图像信息列表
            earliest_date: 最早的拍摄日期

        Returns:
            更新后的图像信息列表
        """
        if not earliest_date:
            logger.warning("无法计算工作天数：未找到任何有效拍摄日期")
            return image_info_list

        for info in image_info_list:
            date_taken = info.get('拍摄日期对象')
            if date_taken:
                working_days = (date_taken.date() - earliest_date.date()).days + 1
                info['工作天数'] = working_days

        return image_info_list

    @staticmethod
    def process_independent_detection(image_info_list: List[Dict], confidence_settings: Dict[str, float]) -> List[Dict]:
        """处理独立探测首只标记

        Args:
            image_info_list: 图像信息列表
            confidence_settings: 物种置信度阈值设置

        Returns:
            更新后的图像信息列表
        """
        # 按拍摄日期排序
        sorted_images = sorted(
            [img for img in image_info_list if img.get('拍摄日期对象')],
            key=lambda x: x['拍摄日期对象']
        )

        species_last_detected = {}  # 记录每个物种的最后探测时间

        for img_info in sorted_images:
            # --- 新增逻辑：根据置信度过滤当前图片的物种 ---
            if img_info.get('最低置信度') == '人工校验':
                # 对于人工校验过的数据，直接使用已有的物种名称
                species_names = img_info.get('物种名称', '').split(',')
            else:
                confidences = img_info.get('all_confidences', [])
                classes = img_info.get('all_classes', [])
                names_map = img_info.get('names_map', {})

                if not confidences or not classes or not names_map:
                    img_info['独立探测首只'] = ''
                    continue

                final_species_counts = Counter()
                for cls, conf in zip(classes, confidences):
                    species_name = names_map.get(str(int(cls)))
                    if species_name:
                        # 优先使用物种特定阈值，否则使用全局阈值
                        threshold = confidence_settings.get(species_name, confidence_settings.get("global", 0.25))
                        if conf >= threshold:
                            final_species_counts[species_name] += 1

                if not final_species_counts:
                    species_names = ['空']
                else:
                    species_names = list(final_species_counts.keys())
            # --- 过滤逻辑结束 ---

            current_time = img_info.get('拍摄日期对象')

            if not current_time or not species_names or species_names == [''] or species_names == ['空']:
                img_info['独立探测首只'] = ''
                continue

            is_independent = False

            for species in species_names:
                if species in species_last_detected:
                    # 检查时间差是否超过阈值
                    time_diff = (current_time - species_last_detected[species]).total_seconds()
                    if time_diff > INDEPENDENT_DETECTION_THRESHOLD:
                        is_independent = True
                else:
                    # 首次探测该物种
                    is_independent = True

                # 更新最后探测时间
                species_last_detected[species] = current_time

            img_info['独立探测首只'] = '是' if is_independent else ''

        return image_info_list

    @staticmethod
    def export_to_excel(image_info_list: List[Dict], output_path: str, confidence_settings: Dict[str, float],
                        file_format: str = 'excel') -> bool:
        """将图像信息导出为Excel或CSV文件

        Args:
            image_info_list: 图像信息列表
            output_path: 输出文件路径
            confidence_settings: 物种置信度阈值设置
            file_format: 文件格式 ('excel' 或 'csv')

        Returns:
            是否成功导出
        """
        if not image_info_list:
            logger.warning("没有数据可导出")
            return False

        # 加载鸟类名录用于物种类型分类
        bird_list_path = resource_path(os.path.join("res", "中国鸟类名录.xlsx"))
        bird_names = set()
        if os.path.exists(bird_list_path):
            try:
                df_birds = pd.read_excel(bird_list_path)
                # 假设物种名称在C列
                if df_birds.shape[1] > 2:
                    bird_names = set(df_birds.iloc[:, 2].dropna().astype(str).tolist())
            except Exception as e:
                logger.error(f"加载鸟类名录失败: {e}")

        personnel_names = {"人", "牧民", "人员"}

        try:
            # 在导出前根据置信度阈值更新数据
            for info in image_info_list:
                species_names_str = info.get('物种名称', '')
                if info.get('最低置信度') == '人工校验':
                    if species_names_str and species_names_str != '空':
                        species_list = [s.strip() for s in species_names_str.split(',')]
                        type_list = []
                        for species in species_list:
                            if species in personnel_names:
                                type_list.append("人员")
                            elif species in bird_names:
                                type_list.append("鸟")
                            else:
                                type_list.append("兽")
                        # --- 修改开始 ---
                        # 去重并排序后合并，实现您的需求
                        unique_types = sorted(list(set(type_list)))
                        info['物种类型'] = ','.join(unique_types)
                        # --- 修改结束 ---
                    else:
                        info['物种类型'] = ''
                    continue

                species_names_original = info.get('物种名称', '').split(',')
                if not species_names_original or species_names_original == ['']:
                    info['物种类型'] = ''
                    continue

                confidences = info.get('all_confidences', [])
                classes = info.get('all_classes', [])
                names_map = info.get('names_map', {})

                if not confidences or not classes or not names_map:
                    info['物种类型'] = ''
                    continue

                final_species_counts = Counter()
                valid_confidences = []

                for cls, conf in zip(classes, confidences):
                    species_name = names_map.get(str(int(cls)))  # JSON keys are strings
                    if species_name:
                        # 优先使用物种特定阈值，否则使用全局阈值
                        threshold = confidence_settings.get(species_name, confidence_settings.get("global", 0.25))
                        if conf >= threshold:
                            final_species_counts[species_name] += 1
                            valid_confidences.append(conf)

                type_list = []
                if not final_species_counts:
                    info['物种名称'] = '空'
                    info['物种数量'] = '空'
                    info['最低置信度'] = ''
                    info['物种类型'] = ''
                else:
                    filtered_species_list = list(final_species_counts.keys())
                    for species in filtered_species_list:
                        if species in personnel_names:
                            type_list.append("人员")
                        elif species in bird_names:
                            type_list.append("鸟")
                        else:
                            type_list.append("兽")
                    # --- 修改开始 ---
                    # 去重并排序后合并，实现您的需求
                    unique_types = sorted(list(set(type_list)))
                    info['物种类型'] = ','.join(unique_types)
                    # --- 修改结束 ---

                    info['物种名称'] = ','.join(filtered_species_list)
                    info['物种数量'] = ','.join(map(str, final_species_counts.values()))
                    if valid_confidences:
                        info['最低置信度'] = f"{min(valid_confidences):.3f}"
                    else:
                        info['最低置信度'] = ''

            # 使用pandas创建DataFrame更高效
            df = pd.DataFrame(image_info_list)

            # 选择需要的列并按顺序排列
            columns = ['文件名', '格式', '拍摄日期', '拍摄时间', '工作天数',
                       '物种名称', '物种类型', '物种数量', '最低置信度', '独立探测首只', '备注']

            # 确保所有列都存在，不存在的列填充空值
            for col in columns:
                if col not in df.columns:
                    df[col] = ''

            # 只保留需要的列并排序
            df = df[columns]

            # 根据选择的格式导出文件
            if file_format.lower() == 'excel':
                df.to_excel(output_path, sheet_name="物种检测信息", index=False)
            elif file_format.lower() == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 以便 Excel 正确显示中文

            return True
        except Exception as e:
            logger.error(f"导出文件失败: {e}")
            return False