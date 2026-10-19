# system/data_processor.py

import os
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from collections import Counter

from system.config import INDEPENDENT_DETECTION_THRESHOLD
from system.utils import resource_path

logger = logging.getLogger(__name__)


def select_classification_candidate(
    candidates: object,
    confidence_settings: Dict[str, float],
) -> tuple[str, float] | None:
    """Return the highest-ranked candidate that passes its species threshold."""

    if not isinstance(candidates, list):
        return None
    global_threshold = float(confidence_settings.get("global", 0.25))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get('accepted') is False:
            continue
        name = str(candidate.get("name") or "").strip()
        if not name:
            continue
        try:
            confidence = float(
                candidate.get("conf", candidate.get("confidence", 0))
            )
        except (TypeError, ValueError):
            continue
        threshold = float(confidence_settings.get(name, global_threshold))
        if confidence >= threshold:
            return name, confidence
    return None


def select_image_detections(
    info: Dict,
    confidence_settings: Dict[str, float],
) -> tuple[Counter, list[float]]:
    """Use final boxes/candidates; only pre-box-format data may use detector arrays."""
    counts = Counter()
    confidences = []
    global_threshold = confidence_settings.get('global', 0.25)
    boxes = info.get('检测框') or []
    if boxes:
        for box in boxes:
            species = box.get('物种')
            if species == '拒识/Unknown':
                # Rejection is a final decision, with no numeric confidence.
                counts[species] += 1
                continue
            if box.get('候选项'):
                chosen = select_classification_candidate(box['候选项'], confidence_settings)
                if chosen is None:
                    continue
                species, confidence = chosen
            else:
                if not species or species == '空':
                    continue
                raw_confidence = box.get('置信度', 0)
                confidence = float(raw_confidence) if raw_confidence is not None else None
                if confidence is not None and confidence < confidence_settings.get(species, global_threshold):
                    continue
            counts[species] += 1
            if confidence is not None:
                confidences.append(confidence)
    elif info.get('分类候选项'):
        # Classifier-only jobs have candidates but no detection boxes.
        chosen = select_classification_candidate(info['分类候选项'], confidence_settings)
        if chosen is not None:
            species, confidence = chosen
            counts[species] += 1
            confidences.append(confidence)
    elif '检测框' not in info:
        names = info.get('names_map', {})
        for cls, confidence in zip(info.get('all_classes', []), info.get('all_confidences', [])):
            class_id = int(cls)
            species = names.get(str(class_id), names.get(class_id))
            if species and confidence >= confidence_settings.get(species, global_threshold):
                counts[species] += 1
                confidences.append(confidence)
    return counts, confidences


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
    def process_independent_detection(image_info_list: List[Dict], confidence_settings: Dict[str, float],
                                      min_frame_ratio: float = 0.0) -> List[Dict]:
        """处理独立探测首只标记 """
        # 按拍摄日期排序
        sorted_images = sorted(
            [img for img in image_info_list if img.get('拍摄日期对象')],
            key=lambda x: x['拍摄日期对象']
        )

        species_last_detected = {}  # 记录每个物种的最后探测时间

        for img_info in sorted_images:
            species_names = []

            # === 修改：增加视频tracks处理逻辑 ===
            if img_info.get('最低置信度') == '人工校验':
                # 对于人工校验过的数据，直接使用已有的物种名称
                names_str = img_info.get('物种名称', '')
                if names_str and names_str != '空':
                    species_names = [s.strip() for s in names_str.split(',')]

            elif 'tracks' in img_info:
                # 视频文件处理
                total_frames = img_info.get('total_frames_processed', 1)
                tracks = img_info.get('tracks', {})
                threshold = total_frames * min_frame_ratio

                track_species_list = []
                for t_id, points in tracks.items():
                    if len(points) < threshold: continue

                    # 收集该轨迹的有效投票
                    votes = []
                    for p in points:
                        sp = p.get('species', 'Unknown')
                        conf = p.get('confidence', 0)
                        thresh = confidence_settings.get(sp, confidence_settings.get("global", 0.25))
                        if conf >= thresh:
                            votes.append(sp)

                    if votes:
                        # 选出该轨迹的最终物种
                        most_common = Counter(votes).most_common(1)[0][0]
                        track_species_list.append(most_common)

                species_names = list(set(track_species_list)) if track_species_list else ['空']

            else:
                final_species_counts, _ = select_image_detections(img_info, confidence_settings)
                species_names = [
                    name for name in final_species_counts
                    if name not in {'空', 'Unknown', '拒识/Unknown'}
                ]
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

            img_info['独立探测首只'] = '1' if is_independent else ''

        return image_info_list

    @staticmethod
    def export_to_excel(image_info_list: List[Dict], output_path: str, confidence_settings: Dict[str, float],
                        file_format: str = 'excel', columns_to_export: Optional[List[str]] = None,
                        min_frame_ratio: float = 0.0, progress_callback=None) -> bool:
        """将图像信息导出为Excel或CSV文件 (增加候选物种过滤逻辑)"""
        if not image_info_list:
            logger.warning("没有数据可导出")
            return False

        requested_confidence_settings = dict(confidence_settings or {})
        confidence_settings = {}
        conf_path = os.path.join("temp", "conf.json")
        if os.path.exists(conf_path):
            try:
                with open(conf_path, 'r', encoding='utf-8') as f:
                    file_conf = json.load(f)
                    if isinstance(file_conf, dict):
                        confidence_settings.update(file_conf)
                        logger.info(f"导出数据时已加载并应用阈值配置文件: {conf_path}")
            except Exception as e:
                logger.error(f"加载阈值配置文件失败: {e}，将继续使用默认设置", exc_info=True)
        # Values selected in the current UI are newer and must override the
        # optional legacy conf.json defaults.
        confidence_settings.update(requested_confidence_settings)

        # --- 加载生物物种名录 ---
        species_info_map = {}
        species_db_path = resource_path(os.path.join("res", "species_database.db"))

        if os.path.exists(species_db_path):
            logger.info(f"正在从 {species_db_path} 加载物种名录...")
            try:
                import sqlite3
                import pandas as pd
                conn = sqlite3.connect(species_db_path)

                # 直接通过 SQL 查询所有需要的列
                query = "SELECT 中文名, 学名, 纲, 目中文名, 目拉丁名, 科中文名, 科拉丁名, 属中文名, 属拉丁名, 物种类型 FROM species"
                df_species = pd.read_sql_query(query, conn)
                conn.close()

                for _, row in df_species.iterrows():
                    chinese_name = row.get('中文名')
                    if pd.notna(chinese_name) and str(chinese_name).strip():
                        cleaned_name = str(chinese_name).strip()
                        species_info_map[cleaned_name] = {
                            '学名': str(row.get('学名', '')).strip(),
                            '纲': str(row.get('纲', '')).strip(),
                            '目名': str(row.get('目中文名', '')).strip(),
                            '目拉丁名': str(row.get('目拉丁名', '')).strip(),
                            '科名': str(row.get('科中文名', '')).strip(),
                            '科拉丁名': str(row.get('科拉丁名', '')).strip(),
                            '属名': str(row.get('属中文名', '')).strip(),
                            '属拉丁名': str(row.get('属拉丁名', '')).strip(),
                            '物种类型': str(row.get('物种类型', '')).strip()
                        }

                if species_info_map:
                    logger.info(f"成功加载 {len(species_info_map)} 条物种信息。")
                else:
                    logger.warning("物种名录已加载，但未能提取任何物种信息，请检查数据库表内容。")
            except Exception as e:
                logger.error(f"加载或处理物种数据库失败: {e}", exc_info=True)
        else:
            logger.warning(f"未找到物种数据库文件: {species_db_path}，分类信息将为空。")

        personnel_names = {"人", "牧民", "人员"}

        try:
            total_items = len(image_info_list)  # 获取总数

            # 在导出前根据置信度阈值更新数据
            for idx, info in enumerate(image_info_list):
                # ===== 触发进度条回调 =====
                if progress_callback:
                    progress_callback(idx + 1, total_items)

                # --- 为新列初始化 ---
                info['学名'], info['目名'], info['目拉丁名'], info['科名'], info['科拉丁名'], info['属名'], info[
                    '属拉丁名'] = [''] * 7

                species_names_str = info.get('物种名称', '')

                if info.get('最低置信度') == '人工校验':
                    if species_names_str and species_names_str != '空':
                        species_list = [s.strip() for s in species_names_str.split(',')]
                    else:
                        species_list = []

                elif 'tracks' in info:
                    # === 视频文件处理：投票与过滤 ===
                    total_frames = info.get('total_frames_processed', 1)
                    tracks = info.get('tracks', {})
                    threshold = total_frames * min_frame_ratio

                    final_species_counts = Counter()
                    valid_confidences = []

                    for t_id, points in tracks.items():
                        # 1. 帧数过滤
                        if len(points) < threshold: continue

                        # 2. 收集轨迹内的有效投票
                        votes = []
                        for p in points:
                            sp = p.get('species', 'Unknown')
                            conf = p.get('confidence', 0)
                            thresh = confidence_settings.get(sp, confidence_settings.get("global", 0.25))

                            if conf >= thresh:
                                votes.append(sp)
                                valid_confidences.append(conf)

                        # 3. 轨迹投票
                        if votes:
                            most_common = Counter(votes).most_common(1)[0][0]
                            final_species_counts[most_common] += 1

                    species_list = sorted(list(final_species_counts.keys()))
                    if not species_list:
                        info['物种名称'], info['物种数量'], info['最低置信度'], info['物种类型'] = '空', '空', '', ''
                    else:
                        info['物种名称'] = ','.join(species_list)
                        info['物种数量'] = ','.join([str(final_species_counts[s]) for s in species_list])
                        info['最低置信度'] = f"{min(valid_confidences):.3f}" if valid_confidences else ''


                else:
                    final_species_counts, valid_confidences = select_image_detections(
                        info, confidence_settings,
                    )

                    # 更新 info 中的统计结果
                    species_list = list(final_species_counts.keys())
                    if not species_list:
                        info['物种名称'], info['物种数量'], info['最低置信度'], info['物种类型'] = '空', '空', '', ''

                    else:
                        info['物种名称'] = ','.join(species_list)
                        info['物种数量'] = ','.join(map(str, final_species_counts.values()))
                        info['最低置信度'] = f"{min(valid_confidences):.3f}" if valid_confidences else ''

                # --- 开始填充分类信息 ---
                if species_list:
                    type_list = []
                    sci_info_lists = {k: [] for k in
                                      ['学名', '纲', '目名', '目拉丁名', '科名', '科拉丁名', '属名', '属拉丁名']}

                    for species in species_list:
                        if species in personnel_names:
                            type_list.append("人员")
                        elif species in species_info_map:  # 核心匹配逻辑
                            s_info = species_info_map[species]
                            p_type = s_info.get('物种类型', '')
                            if p_type:
                                type_list.append(p_type)
                            else:
                                # 如果 P 列为空，则回退到原有的根据纲判断的逻辑（可选）
                                if s_info.get('纲') in ['鸟纲', '鸟']:
                                    type_list.append("鸟")
                                elif s_info.get('纲') in ['哺乳纲', '兽']:
                                    type_list.append("兽")
                                elif s_info.get('纲') == '家畜':
                                    type_list.append("家畜")

                            for key in sci_info_lists.keys():
                                if key != '纲': sci_info_lists[key].append(s_info.get(key, ''))
                        else:
                            # 如果物种不在名录中，记录警告并填充空值
                            if species not in personnel_names:
                                logger.warning(f"物种名称 '{species}' 无法在名录中找到匹配项。")
                            for key in sci_info_lists.keys():
                                if key != '纲': sci_info_lists[key].append('')

                    info['物种类型'] = ','.join(sorted(list(set(type_list))))
                    info['学名'] = ','.join(sci_info_lists['学名'])
                    info['目名'] = ','.join(sci_info_lists['目名'])
                    info['目拉丁名'] = ','.join(sci_info_lists['目拉丁名'])
                    info['科名'] = ','.join(sci_info_lists['科名'])
                    info['科拉丁名'] = ','.join(sci_info_lists['科拉丁名'])
                    info['属名'] = ','.join(sci_info_lists['属名'])
                    info['属拉丁名'] = ','.join(sci_info_lists['属拉丁名'])
                else:
                    info['物种类型'] = ''

            # --- 导出到文件 ---
            df = pd.DataFrame(image_info_list)

            # 默认的完整列顺序
            default_columns = ['文件名', '格式', '拍摄日期', '拍摄时间', '工作天数',
                               '物种名称', '学名',
                               '目名', '目拉丁名', '科名', '科拉丁名', '属名', '属拉丁名',
                               '物种类型', '物种数量', '最低置信度', '独立探测首只', '备注']

            # 如果用户传入了要导出的列列表，则使用该列表；否则，使用默认的完整列表
            columns = columns_to_export if columns_to_export is not None and len(
                columns_to_export) > 0 else default_columns

            # 确保所有需要的列都存在于DataFrame中，不存在则填充为空字符串
            for col in columns:
                if col not in df.columns:
                    df[col] = ''

            # 仅选择用户指定的列进行导出
            df = df[columns]

            if file_format.lower() == 'excel':
                df.to_excel(output_path, sheet_name="物种检测信息", index=False)
            elif file_format.lower() == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8-sig')

            logger.info(f"文件已成功导出到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出文件失败: {e}", exc_info=True)
            return False
