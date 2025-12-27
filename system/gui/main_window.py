import os
import sys
import platform
import logging
import threading
import json
import time
from datetime import datetime
import gc
import hashlib
import shutil
from PIL import Image
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMessageBox, QFileDialog, QApplication, QStackedWidget,
    QDialog, QLabel, QTextBrowser, QDialogButtonBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, Slot
from PySide6.QtGui import QIcon, QPixmap, QPalette, QTextDocument

from system.config import APP_TITLE, APP_VERSION, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from system.utils import resource_path
from system.image_processor import ImageProcessor
from system.metadata_extractor import ImageMetadataExtractor
from system.data_processor import DataProcessor
from system.settings_manager import SettingsManager
from system.update_checker import check_for_updates, get_latest_version_info, compare_versions, start_download_thread, \
    _show_messagebox

# Import GUI components
from system.gui.sidebar import Sidebar
from system.gui.start_page import StartPage
from system.gui.preview_page import PreviewPage
from system.gui.species_validation_page import SpeciesValidationPage
from system.gui.advanced_page import AdvancedPage
from system.gui.about_page import AboutPage
from system.gui.ui_components import InfoBar, Win11Colors, ThemeManager

if platform.system() == "Windows":
    import ctypes

logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """图像处理线程"""
    progress_updated = Signal(int, int, float, float, float)  # current, total, elapsed, remaining, speed
    file_processed = Signal(str, object, str)
    processing_complete = Signal(bool)
    status_message = Signal(str)
    file_processing_result = Signal(str, dict)
    current_file_changed = Signal(str, int, int)
    current_file_preview = Signal(str, dict)
    console_log = Signal(str, str)  # 控制台日志信号 (message, color)

    def __init__(self,
                 controller,
                 file_path,
                 save_path,
                 use_fp16,
                 resume_from=0):
        super().__init__()
        self.controller = controller
        self.file_path = file_path
        self.save_path = save_path
        self.use_fp16 = use_fp16
        self.resume_from = resume_from
        self.stop_flag = False
        self.force_stop_flag = False
        # 添加用于保存进度的变量
        self.current_excel_data = []
        self.current_processed_files = 0
        self.current_total_files = 0

    def stop(self):
        """停止处理线程并保存进度"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console_log.emit(f"[INFO] {current_time} 正在停止处理并保存进度...", "#ffff00")
        QThread.msleep(10)

        self.stop_flag = True

        # 保存当前进度
        if self.current_processed_files > 0:
            self._save_processing_cache(
                self.current_excel_data,
                self.current_processed_files,
                self.current_total_files
            )
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.console_log.emit(
                f"[INFO] {current_time} 进度已保存: {self.current_processed_files}/{self.current_total_files}",
                "#00ff00"
            )
            QThread.msleep(10)

    def run(self):
        import time
        import math
        import cv2  # 引入cv2用于获取视频帧数

        class ForceStopError(Exception):
            pass

        start_time = time.time()
        excel_data = [] if self.resume_from == 0 else self.controller.excel_data

        # 记录已处理的文件数（用于索引文件列表）
        processed_files_count = self.resume_from
        stopped_manually = False
        earliest_date = None
        temp_photo_dir = self.controller.get_temp_photo_dir()

        try:
            iou = self.controller.advanced_page.iou_var
            conf = self.controller.advanced_page.conf_var
            augment = self.controller.advanced_page.use_augment_var
            agnostic_nms = self.controller.advanced_page.use_agnostic_nms_var
            vid_stride = getattr(self.controller.advanced_page, 'vid_stride_var', 1) # 获取跳帧参数，默认为1

            from system.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
            all_extensions = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS

            # 获取文件夹下所有支持的文件
            all_files_list = sorted([f for f in os.listdir(self.file_path)
                                     if f.lower().endswith(all_extensions)])
            total_files_count = len(all_files_list)

            # =========================================================================
            # [修改] 预计算总工作单元（统计视频帧数 + 图片数量）
            # =========================================================================
            self.console_log.emit("=" * 118, None)
            self.console_log.emit(f"[INFO] 正在预扫描文件以计算总工作量(统计视频帧数)...", "#00ff00")
            QThread.msleep(10)

            total_work_units = 0  # 总工作量（帧数+图片数）
            file_unit_map = {}  # 记录每个文件对应的工作量

            for f in all_files_list:
                f_path = os.path.join(self.file_path, f)
                units = 1
                if f.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                    try:
                        cap = cv2.VideoCapture(f_path)
                        if cap.isOpened():
                            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            # 使用math.ceil向上取整计算实际需要处理的帧数 (考虑跳帧)
                            units = math.ceil(frames / vid_stride) if frames > 0 else 1
                        cap.release()
                    except Exception:
                        units = 1

                file_unit_map[f] = units
                total_work_units += units

            # 计算断点续传前的已完成工作量
            processed_work_units = 0
            for i in range(self.resume_from):
                processed_work_units += file_unit_map.get(all_files_list[i], 1)

            # 记录本次会话开始时的已完成量，用于计算实时速度
            start_work_units = processed_work_units

            # 初始化进度变量
            self.current_total_files = total_files_count
            self.current_excel_data = excel_data
            self.current_processed_files = processed_files_count

            # 输出开始信息
            self.console_log.emit("=" * 118, None)
            QThread.msleep(10)

            # 居中显示绿色的 START
            start_text = "START"
            padding = 31
            centered_start = "＊" * padding + start_text + "＊" * padding
            self.console_log.emit(centered_start, "#00ff00")
            QThread.msleep(10)

            self.console_log.emit("=" * 118, None)
            QThread.msleep(10)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # [修改] 日志显示总帧数/图数
            self.console_log.emit(
                f"[INFO] {current_time} 开始处理 {total_files_count} 个文件 (总计 {total_work_units} 帧/图)", "#00ff00")
            QThread.msleep(10)

            display_file_path = os.path.normpath(self.file_path)
            display_save_path = os.path.normpath(self.save_path)

            self.console_log.emit(f"[INFO] {current_time} 源路径: {display_file_path}", "#aaaaaa")
            QThread.msleep(10)
            self.console_log.emit(f"[INFO] {current_time} 保存路径: {display_save_path}", "#aaaaaa")
            QThread.msleep(10)
            self.console_log.emit(
                f"[INFO] {current_time} 参数配置: IOU={iou}, CONF={conf}, FP16={self.use_fp16}, AUGMENT={augment}, AGNOSTIC_NMS={agnostic_nms}, VID_STRIDE={vid_stride}",
                "#aaaaaa")
            QThread.msleep(10)
            self.console_log.emit("=" * 118, None)
            QThread.msleep(10)

            if self.resume_from > 0:
                # 获取本次需要处理的文件列表
                files_to_process = all_files_list[self.resume_from:]
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.console_log.emit(f"[INFO] {current_time} 从第 {self.resume_from + 1} 个文件继续处理", "#ffff00")
                QThread.msleep(10)
                if excel_data:
                    valid_dates = [item['拍摄日期对象'] for item in excel_data if item.get('拍摄日期对象')]
                    if valid_dates:
                        earliest_date = min(valid_dates)
            else:
                files_to_process = all_files_list

            # 遍历处理文件
            for idx, filename in enumerate(files_to_process):
                # 1. 检查停止标志 (循环开始处)
                # 如果 stop_flag 被设置(第一次点击)，但没有强制停止(第二次点击)
                # 说明用户希望"等待当前处理完毕后停止"。此时上一个文件已跑完，可以安全退出了。
                if self.stop_flag and not self.force_stop_flag:
                    stopped_manually = True
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.console_log.emit(f"[INFO] {current_time} 当前文件处理完毕，正在停止...", "#ffff00")
                    # 保存进度 (此时 processed_files_count 已包含刚完成的文件)
                    self._save_processing_cache(excel_data, processed_files_count, total_files_count)
                    break

                # 如果是强制停止，直接退出 (虽然通常会被异常捕获，这里做双重保险)
                if self.force_stop_flag:
                    stopped_manually = True
                    break

                current_file_index_display = processed_files_count + 1
                img_path = os.path.join(self.file_path, filename)
                is_video = filename.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS)

                # 发射当前处理文件变化信号
                self.current_file_changed.emit(img_path, current_file_index_display, total_files_count)

                try:
                    if is_video:
                        # === 视频处理逻辑 ===
                        image_info = {
                            '文件名': filename,
                            '格式': filename.split('.')[-1].lower(),
                            '拍摄日期': None,
                            '拍摄时间': None,
                            '拍摄日期对象': None,
                            '工作天数': None,
                            '物种名称': '',
                            '物种数量': '',
                            'detect_results': None,
                            '最低置信度': None,
                            '独立探测首只': '',
                            '检测时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # 定义视频状态回调函数，集成进度更新
                        def video_log_callback(frame_idx, total_frames, w, h, counts, speed_ms):
                            # 只有在"强制停止"时才抛出异常中断视频
                            # 如果只是 stop_flag (第一次点击)，则忽略，让视频跑完
                            if self.force_stop_flag:
                                raise ForceStopError("用户强制停止")
                            # frame_idx 是当前绝对帧位置(或接近)，需将其转换为处理过的“单元”数
                            processed_frames_so_far = math.ceil(frame_idx / vid_stride)

                            # 计算当前总进度：之前文件完成的单元 + 当前视频已处理帧数
                            current_total_done = processed_work_units + processed_frames_so_far

                            elapsed_time = time.time() - start_time

                            # 计算速度 (基于本次会话已处理的单元数)
                            session_units_done = current_total_done - start_work_units

                            if session_units_done > 0 and elapsed_time > 0:
                                speed = session_units_done / elapsed_time  # 单位：帧/秒
                                remaining_time = (total_work_units - current_total_done) / speed
                            else:
                                speed = 0
                                remaining_time = float('inf')

                            # 发送进度更新
                            self.progress_updated.emit(current_total_done, total_work_units, elapsed_time,
                                                       remaining_time, speed)

                            # 原始日志逻辑
                            if counts:
                                species_str = ", ".join([f"{c} {n}" for n, c in counts.items()])
                            else:
                                species_str = "无目标"

                            display_path = img_path.replace('/', '\\')
                            msg = (f"video {current_file_index_display}/{total_files_count} "
                                   f"(frame {frame_idx}/{total_frames}) "
                                   f"{display_path}: {w}x{h} "
                                   f"{species_str}, {speed_ms:.1f}ms")
                            self.console_log.emit(msg, None)

                        video_output_dir = os.path.join(self.save_path, "video_results")
                        detection_start = time.time()

                        # 调用检测方法并传入回调
                        video_result = self.controller.image_processor.detect_video_species(
                            img_path,
                            video_output_dir,
                            bool(self.use_fp16),
                            iou, conf, augment, agnostic_nms,
                            status_callback=video_log_callback,
                            vid_stride=vid_stride,  # 传入跳帧参数
                            temp_video_dir=temp_photo_dir
                        )

                        detection_time = (time.time() - detection_start) * 1000

                        if video_result.get('status') == 'success':
                            # 解析视频结果
                            json_path = video_result.get('json_path')
                            if json_path and os.path.exists(json_path):
                                try:
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        v_data = json.load(f)
                                    tracks = v_data.get('tracks', {})
                                    v_counts = {}
                                    for t_list in tracks.values():
                                        if t_list:
                                            s_name = t_list[0].get('species', 'Unknown')
                                            v_counts[s_name] = v_counts.get(s_name, 0) + 1

                                    if v_counts:
                                        image_info['物种名称'] = ','.join(v_counts.keys())
                                        image_info['物种数量'] = ','.join(map(str, v_counts.values()))
                                    else:
                                        image_info['物种名称'] = '空'
                                        image_info['物种数量'] = '空'
                                except Exception as e:
                                    logger.error(f"解析视频结果JSON失败: {e}")
                        else:
                            image_info['错误'] = video_result.get('error', 'Video processing failed')

                        # 输出视频处理完成日志
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        display_img_path = os.path.normpath(img_path)
                        log_message = (
                            f"[INFO] {current_time} {display_img_path} [视频] | "
                            f"检测结果:[{image_info.get('物种名称', '未知')}] | "
                            f"耗时:{detection_time:.1f}ms"
                        )
                        self.console_log.emit(log_message, "#00ff00")
                        QThread.msleep(5)

                        excel_data.append(image_info)

                        # [修改] 视频处理完成后，累加该视频的总帧数到已完成工作量
                        processed_work_units += file_unit_map.get(filename, 1)

                    else:
                        # === 图片处理逻辑 ===
                        if self.force_stop_flag:
                            raise ForceStopError("用户强制停止")
                        # 提取元数据
                        image_info, img = ImageMetadataExtractor.extract_metadata(img_path, filename)

                        img_height, img_width = 0, 0
                        if img is not None:
                            if hasattr(img, 'shape') and len(img.shape) >= 2:
                                img_height, img_width = img.shape[:2]
                            else:
                                try:
                                    with Image.open(img_path) as pil_img:
                                        img_width, img_height = pil_img.size
                                except Exception as e:
                                    logger.warning(f"无法获取图像尺寸 {filename}: {e}")

                        if img_height == 0 or img_width == 0:
                            raise ValueError(f"无法获取有效的图像尺寸")

                        detection_start = time.time()
                        species_info = self.controller.image_processor.detect_species(
                            img_path, bool(self.use_fp16), iou, conf, augment, agnostic_nms
                        )
                        detection_time = (time.time() - detection_start) * 1000
                        species_info['检测时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        detect_results = species_info.get('detect_results')

                        # 获取翻译字典并统计
                        translation_dict = self.controller.image_processor.translation_dict
                        species_counts = {}
                        if detect_results:
                            for result in detect_results:
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    for box in result.boxes:
                                        cls_id = int(box.cls.item())
                                        english_name = result.names.get(cls_id, 'Unknown')
                                        translated_name = translation_dict.get(english_name, english_name)
                                        species_counts[translated_name] = species_counts.get(translated_name, 0) + 1

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        display_img_path = os.path.normpath(img_path)

                        if species_counts:
                            detection_summary = ', '.join([f"{count}x{name}" for name, count in species_counts.items()])
                            log_message = (
                                f"[INFO] {current_time} {display_img_path} | "
                                f"尺寸:{img_height}x{img_width} | "
                                f"检测结果:[{detection_summary}] | "
                                f"检测耗时:{detection_time:.1f}ms"
                            )
                            self.console_log.emit(log_message, "#00ff00")
                        else:
                            log_message = (
                                f"[INFO] {current_time} {display_img_path} | "
                                f"尺寸:{img_height}x{img_width} | "
                                f"检测结果:[无目标] | "
                                f"检测耗时:{detection_time:.1f}ms"
                            )
                            self.console_log.emit(log_message, "#ffaa00")
                        QThread.msleep(5)

                        self.controller.image_processor.save_detection_info_json(
                            detect_results, filename, species_info, temp_photo_dir
                        )
                        self.file_processed.emit(img_path, detect_results, filename)

                        complete_detection_info = {
                            **species_info,
                            'detect_results': detect_results,
                            'filename': filename
                        }
                        self.current_file_preview.emit(img_path, complete_detection_info)

                        if 'detect_results' in species_info:
                            del species_info['detect_results']

                        image_info.update(species_info)
                        excel_data.append(image_info)

                        # [修改] 图片处理完成后，工作量 +1
                        processed_work_units += 1

                        # [修改] 更新进度条 (图片)
                        elapsed_time = time.time() - start_time
                        session_units_done = processed_work_units - start_work_units

                        if session_units_done > 0 and elapsed_time > 0:
                            speed = session_units_done / elapsed_time  # 图/秒
                            remaining_time = (total_work_units - processed_work_units) / speed
                        else:
                            speed = 0
                            remaining_time = float('inf')

                        self.progress_updated.emit(processed_work_units, total_work_units, elapsed_time, remaining_time,
                                                   speed)

                except ForceStopError:
                    stopped_manually = True
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.console_log.emit(f"[WARN] {current_time} 检测到强制停止信号，正在中断后台处理...", "#ff0000")
                    break  # 跳出文件循环

                except Exception as e:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    display_img_path = os.path.normpath(img_path)
                    error_message = (
                        f"[WARN] {current_time} {display_img_path} | "
                        f"处理失败 | "
                        f"错误信息:{str(e)}"
                    )
                    logger.error(f"处理文件 {filename} 失败: {e}", exc_info=True)
                    self.console_log.emit(error_message, "#ff0000")
                    QThread.msleep(5)

                    try:
                        image_info = {'文件名': filename, '错误': str(e)}
                        excel_data.append(image_info)
                    except:
                        pass

                    # 出错时也要更新进度，防止卡死
                    if is_video:
                        processed_work_units += file_unit_map.get(filename, 1)
                    else:
                        processed_work_units += 1

                processed_files_count += 1
                self.current_processed_files = processed_files_count
                self.current_excel_data = excel_data

                if processed_files_count % 10 == 0:
                    self._save_processing_cache(excel_data, processed_files_count, total_files_count)
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.console_log.emit(
                        f"[INFO] {current_time} 已保存进度缓存 ({processed_files_count}/{total_files_count} 文件)",
                        "#888888")
                    QThread.msleep(5)

                try:
                    del img_path, image_info, img, species_info, detect_results
                except NameError:
                    pass
                gc.collect()

            if not stopped_manually:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.console_log.emit("=" * 118, None)
                QThread.msleep(10)

                total_time = time.time() - start_time
                # [修改] 使用工作单元（帧/图）来计算最终平均速度
                actual_processed_units = processed_work_units - start_work_units
                avg_speed = actual_processed_units / total_time if total_time > 0 else 0

                self.console_log.emit(
                    f"[INFO] {current_time} 所有文件处理完成! | "
                    f"总计:{total_files_count}个文件 ({total_work_units} 单元) | "
                    f"总耗时:{total_time:.2f}秒 | "
                    f"平均速度:{avg_speed:.2f} 帧(图)/秒",
                    "#00ff00"
                )
                QThread.msleep(10)

                self.progress_updated.emit(total_work_units, total_work_units, total_time, 0, avg_speed)
                self.controller.excel_data = excel_data

                # 日期格式化与数据处理
                for item in excel_data:
                    if '拍摄日期对象' in item:
                        date_obj = item['拍摄日期对象']
                        if isinstance(date_obj, str):
                            try:
                                item['拍摄日期对象'] = datetime.fromisoformat(date_obj)
                            except (ValueError, AttributeError):
                                try:
                                    item['拍摄日期对象'] = datetime.strptime(date_obj, "%Y-%m-%d %H:%M:%S")
                                except (ValueError, AttributeError):
                                    item['拍摄日期对象'] = None
                        elif not isinstance(date_obj, datetime):
                            item['拍摄日期对象'] = None

                excel_data = DataProcessor.process_independent_detection(excel_data,
                                                                         self.controller.confidence_settings)
                if earliest_date:
                    excel_data = DataProcessor.calculate_working_days(excel_data, earliest_date)
                self._delete_processing_cache()
                self.status_message.emit("处理完成！")
                QTimer.singleShot(0, lambda: QMessageBox.information(None, "成功", "图像处理完成！"))

            self.processing_complete.emit(not stopped_manually)

        except Exception as e:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.error(f"处理过程中发生错误: {e}", exc_info=True)
            self.console_log.emit(f"[WARN] {current_time} 处理过程发生严重错误: {str(e)}", "#ff0000")
            QThread.msleep(10)
            QTimer.singleShot(0, lambda: QMessageBox.critical(None, "错误", f"处理过程中发生错误: {e}"))
            self.processing_complete.emit(False)
        finally:
            gc.collect()

    def _save_processing_cache(self, excel_data, processed_files, total_files):
        """保存处理缓存"""
        try:
            cache_dir = self.controller.settings_manager.settings_dir
            cache_file = os.path.join(cache_dir, "cache.json")

            cache_data = {
                'processed_files': processed_files,
                'total_files': total_files,
                'file_path': self.file_path,
                'save_path': self.file_path,
                'excel_data': excel_data,
                'timestamp': datetime.now().isoformat()
            }

            # 转换日期对象为字符串
            for item in cache_data.get('excel_data', []):
                if '拍摄日期对象' in item and isinstance(item['拍摄日期对象'], datetime):
                    item['拍摄日期对象'] = item['拍摄日期对象'].isoformat()

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"处理缓存已保存: {processed_files}/{total_files}")
        except Exception as e:
            logger.error(f"保存处理缓存失败: {e}")

    def _delete_processing_cache(self):
        """删除处理缓存"""
        try:
            cache_dir = self.controller.settings_manager.settings_dir
            cache_file = os.path.join(cache_dir, "cache.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info("处理缓存已删除")
        except Exception as e:
            logger.error(f"删除处理缓存失败: {e}")


class ObjectDetectionGUI(QMainWindow):
    """主应用程序窗口 - PySide6版本"""

    def __init__(self, settings_manager: SettingsManager, settings: dict, resume_processing: bool, cache_data: dict):
        super().__init__()
        self.settings_manager = settings_manager
        self.settings = settings
        self.resume_processing = resume_processing
        self.cache_data = cache_data
        self.current_temp_photo_dir = None
        self.is_dark_mode = False
        self.accent_color = Win11Colors.LIGHT_ACCENT  # 默认使用浅色模式的颜色

        # 检查CUDA可用性
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
        except ImportError:
            self.cuda_available = False

        self.is_processing = False
        self.processing_thread = None
        self.excel_data = []
        self.current_page = "settings"
        self.update_channel_var = "稳定版 (Release)"
        self.model_var = ""
        self.confidence_settings = self.settings_manager.load_confidence_settings()

        # 新增：用于存储最新进度信息
        self.last_progress_value = 0
        self.last_progress_total = 0
        self.last_progress_speed = 0.0
        self.last_progress_remaining_time = float('inf')
        self.last_progress_elapsed_time = 0.0
        self.current_processing_file = None

        self.last_preview_image = None
        self.last_validation_species = None
        self.last_validation_image = None

        # 主题检测和应用
        self._apply_system_theme()
        self._setup_window()
        self._initialize_model(settings)
        self._create_ui_elements()
        self._setup_connections()

        # 在窗口初始化后设置标题栏颜色
        self._set_title_bar_color()

        # 加载设置
        if self.settings:
            self._load_settings_to_ui(self.settings)
        else:
            self._create_default_settings()

        # 设置定时器进行启动后的初始化
        QTimer.singleShot(100, self._post_init)

    def _apply_system_theme(self):
        """应用系统主题"""
        try:
            import darkdetect
            system_theme = darkdetect.theme().lower()
            if system_theme == 'dark':
                self.is_dark_mode = True
                self.accent_color = Win11Colors.DARK_ACCENT
            else:
                self.is_dark_mode = False
                self.accent_color = Win11Colors.LIGHT_ACCENT
        except Exception as e:
            self.is_dark_mode = False
            self.accent_color = Win11Colors.LIGHT_ACCENT
            logger.warning(f"无法检测系统主题: {e}")

        # 应用Win11样式
        ThemeManager.apply_win11_style(QApplication.instance())

    def _setup_window(self):
        """设置窗口"""
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1100, 750)
        self.resize(1100, 750)

        # 居中显示
        screen = QApplication.primaryScreen().geometry()
        window_rect = self.geometry()
        x = (screen.width() - window_rect.width()) // 2
        y = (screen.height() - window_rect.height()) // 2
        self.move(x, y)

        # --- 关键修改 1: 必须最先设置 AppID ---
        # 这告诉 Windows 这是一个独立的程序，不应该和 python.exe 混在一起
        if platform.system() == "Windows":
            try:
                # 如果修改代码后图标还不更新，尝试在下方字符串末尾加个 ".v2" 强制刷新 Windows 缓存
                myappid = f'mycompany.{APP_TITLE}.{APP_VERSION}'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception as e:
                logger.warning(f"设置 AppID 失败: {e}")

        # --- 关键修改 2: 同时设置 应用程序图标 和 窗口图标 ---
        try:
            ico_path = resource_path("res/ico.ico")

            # 增加文件存在性检查，确保路径正确
            if os.path.exists(ico_path):
                icon = QIcon(ico_path)

                # A. 设置主窗口图标 (界面左上角)
                self.setWindowIcon(icon)

                # B. ！！！设置应用程序全局图标 (任务栏图标)！！！
                QApplication.instance().setWindowIcon(icon)
            else:
                logger.warning(f"图标文件未找到: {ico_path}")

        except Exception as e:
            logger.warning(f"无法加载窗口图标: {e}")

    def _initialize_model(self, settings: dict):
        """初始化模型"""
        saved_model_name = settings.get("selected_model") if settings else None
        model_path = None
        res_dir = resource_path("res")

        # 尝试从设置中加载模型
        if saved_model_name:
            potential_path = os.path.join(res_dir, "model", saved_model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                logger.info(f"从设置加载模型: {saved_model_name}")
            else:
                logger.warning(f"设置中保存的模型文件不存在: {saved_model_name}。将尝试加载默认模型。")

        # 如果设置中没有模型或文件不存在，则查找第一个可用的模型
        if not model_path:
            model_path = self._find_model_file()
            if model_path:
                logger.info(f"加载找到的第一个模型: {os.path.basename(model_path)}")

        # 初始化 ImageProcessor
        self.image_processor = ImageProcessor(model_path)
        if model_path:
            self.image_processor.model_path = model_path
            self.model_var = os.path.basename(model_path)
        else:
            self.image_processor.model = None
            self.image_processor.model_path = None
            self.model_var = ""
            logger.error("在 res/model 目录中未找到任何有效的模型文件 (.pt)。")

    def _find_model_file(self) -> str:
        """查找模型文件"""
        try:
            model_dir = os.path.join(resource_path("res") ,"model")
            if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
                return None
            model_files = [f for f in os.listdir(model_dir) if f.lower().endswith('.pt')]
            if not model_files:
                return None
            return os.path.join(model_dir, model_files[0])
        except Exception as e:
            logger.error(f"查找模型文件时出错: {e}")
            return None

    def _create_ui_elements(self):
        """创建UI元素"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 创建侧边栏
        self.sidebar = Sidebar(self)
        splitter.addWidget(self.sidebar)

        # 创建内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 创建页面堆栈
        self.content_stack = QStackedWidget()
        content_layout.addWidget(self.content_stack)

        # 创建状态栏
        self.status_bar = InfoBar()
        content_layout.addWidget(self.status_bar)

        splitter.addWidget(content_widget)

        # 设置分割器比例
        splitter.setStretchFactor(0, 0)  # 侧边栏不拉伸
        splitter.setStretchFactor(1, 1)  # 内容区域拉伸

        # 创建页面
        self.start_page = StartPage(self)
        self.preview_page = PreviewPage(self, self)
        self.species_validation_page = SpeciesValidationPage(self)
        self.advanced_page = AdvancedPage(self, self)
        self.about_page = AboutPage(self)

        # 添加页面到堆栈
        self.content_stack.addWidget(self.start_page)
        self.content_stack.addWidget(self.preview_page)
        self.content_stack.addWidget(self.species_validation_page)
        self.content_stack.addWidget(self.advanced_page)
        self.content_stack.addWidget(self.about_page)

        # 显示默认页面
        self._show_page("settings")

    def _setup_connections(self):
        """设置信号连接"""
        # 侧边栏导航
        self.sidebar.page_requested.connect(self._show_page)

        # 开始页面
        self.start_page.browse_file_path_requested.connect(self.browse_file_path)
        self.start_page.file_path_changed.connect(self._validate_and_update_file_path)
        self.start_page.toggle_processing_requested.connect(self.toggle_processing_state)

        self.start_page.settings_changed.connect(self._sync_start_to_advanced)
        self.start_page.settings_changed.connect(self._save_current_settings)

        # 高级页面
        if hasattr(self.advanced_page, 'settings_changed'):
            self.advanced_page.settings_changed.connect(self._save_current_settings)
            self.advanced_page.settings_changed.connect(self._sync_advanced_to_start)

        self.advanced_page.update_check_requested.connect(self.check_for_updates_from_ui)
        self.advanced_page.theme_changed.connect(self.change_theme)
        self.advanced_page.params_help_requested.connect(self.show_params_help)
        self.advanced_page.cache_clear_requested.connect(self.clear_image_cache)

        # 预览页面
        self.preview_page.settings_changed.connect(self._save_current_settings)

        # 物种校验页面
        self.species_validation_page.settings_changed.connect(self._save_current_settings)
        self.species_validation_page.quick_marks_updated.connect(
            self.advanced_page.load_quick_mark_settings)

    def _sync_advanced_to_start(self):
        """将高级页面的设置实时同步到开始页面"""
        # 获取高级页面的当前设置
        settings = self.advanced_page.get_settings()

        # 提取模型和跳帧参数
        model = settings.get("selected_model")
        stride = settings.get("vid_stride")

        # 调用开始页面的更新方法
        if hasattr(self.start_page, 'update_quick_settings'):
            self.start_page.update_quick_settings(model, stride)

    def _sync_start_to_advanced(self):
        """将开始页面的设置同步到高级页面"""
        # 获取开始页面的当前设置
        settings = self.start_page.get_settings()

        # 提取模型和跳帧参数
        model = settings.get("selected_model")
        stride = settings.get("vid_stride")

        # 调用高级页面的同步方法
        if hasattr(self.advanced_page, 'update_quick_settings_sync'):
            self.advanced_page.update_quick_settings_sync(model, stride)

    def _post_init(self):
        """后期初始化"""
        # 检查模型
        if not self.image_processor.model:
            QMessageBox.critical(
                self, "错误",
                "未找到有效的模型文件(.pt)。请在res/model目录中放入至少一个模型文件。"
            )
            if hasattr(self.start_page, 'set_processing_enabled'):
                self.start_page.set_processing_enabled(False)

        # 设置主题监控
        self.setup_theme_monitoring()

        # 加载验证数据
        if hasattr(self.preview_page, '_load_validation_data'):
            self.preview_page._load_validation_data()

        QTimer.singleShot(2000, lambda: self._check_for_updates(silent=True))

        # 恢复处理
        if self.resume_processing and self.cache_data:
            QTimer.singleShot(1000, self._resume_processing)

    def _create_default_settings(self):
        """创建默认设置"""
        logger.info("未找到配置文件，正在使用默认值创建 'setting.json'。")
        default_settings = self._get_current_settings()
        self.settings_manager.save_settings(default_settings)
        self.settings = default_settings
        self.change_theme()

    def _check_for_updates(self, silent=False):
        """检查更新"""
        # 获取通道
        if hasattr(self, 'advanced_page') and hasattr(self.advanced_page, 'update_channel_combo'):
            channel_selection = self.advanced_page.update_channel_combo.currentText()
            # 获取镜像源选择
            mirror_selection = self.advanced_page.update_mirror_combo.currentText()
        else:
            channel_selection = self.update_channel_var
            # 默认镜像
            mirror_selection = "国内源 (KKGitHub)"

        channel = 'preview' if '预览版' in channel_selection else 'stable'

        # 解析镜像源参数
        mirror = 'official'
        if 'KKGitHub' in mirror_selection:
            mirror = 'kkgithub'

        # 使用线程来运行检查
        update_thread = threading.Thread(
            target=check_for_updates,
            # [修改] 传递 mirror 参数
            args=(self, silent, channel, mirror),
            daemon=True
        )
        update_thread.start()

    def check_for_updates_from_ui(self):
        """从UI手动检查更新"""
        try:
            # 从高级设置页面直接获取最新的更新通道选择
            channel_selection = self.advanced_page.update_channel_combo.currentText()
            channel = 'preview' if '预览版' in channel_selection else 'stable'

            # 从高级页面获取镜像源
            mirror_selection = self.advanced_page.update_mirror_combo.currentText()
            mirror = 'official'
            if 'KKGitHub' in mirror_selection:
                mirror = 'kkgithub'

            self.status_bar.status_label.setText(f"正在检查更新 ({mirror_selection})...")

            # 使用线程来运行检查
            update_thread = threading.Thread(
                target=check_for_updates,
                # 传递 mirror 参数
                args=(self, False, channel, mirror),
                daemon=True
            )
            update_thread.start()
        except Exception as e:
            logger.error(f"启动更新检查失败: {e}")
            QMessageBox.critical(self, "错误", f"无法开始更新检查: {e}")

    def change_theme(self):
        """更改主题"""
        if hasattr(self.advanced_page, 'get_theme_selection'):
            selected_theme = self.advanced_page.get_theme_selection()
        else:
            selected_theme = "自动"

        if selected_theme == "自动":
            self._apply_system_theme()
        elif selected_theme == "深色":
            self.is_dark_mode = True
            self.accent_color = Win11Colors.DARK_ACCENT
        else:  # "浅色"
            self.is_dark_mode = False
            self.accent_color = Win11Colors.LIGHT_ACCENT

        # 应用主题
        ThemeManager.apply_win11_style(QApplication.instance())
        # 在主题更改后设置标题栏颜色
        self._set_title_bar_color()
        self._save_current_settings()

    def setup_theme_monitoring(self):
        """设置主题监控"""
        if platform.system() in ["Windows", "Darwin"]:
            self.theme_timer = QTimer()
            self.theme_timer.timeout.connect(self._check_theme_change)
            self.theme_timer.start(10000)  # 每10秒检查一次

    def _check_theme_change(self):
        """检查主题变化"""
        try:
            if hasattr(self.advanced_page, 'get_theme_selection') and \
                    self.advanced_page.get_theme_selection() == "自动":
                import darkdetect
                current_theme = darkdetect.theme().lower()
                if (current_theme == 'dark' and not self.is_dark_mode) or \
                        (current_theme == 'light' and self.is_dark_mode):
                    self._apply_system_theme()
                    ThemeManager.apply_win11_style(QApplication.instance())
        except Exception as e:
            logger.warning(f"检查主题变化失败: {e}")

    def _show_page(self, page_id: str):
        """显示页面（修复版本）"""
        if self.current_page == "preview" and page_id != "preview":
            if hasattr(self.preview_page, 'image_loader_thread'):
                if self.preview_page.image_loader_thread and self.preview_page.image_loader_thread.isRunning():
                    self.preview_page.image_loader_thread.cancel()
                    try:
                        self.preview_page.image_loader_thread.image_loaded.disconnect()
                        self.preview_page.image_loader_thread.loading_failed.disconnect()
                    except:
                        pass
                    self.preview_page.image_loader_thread.wait(100)
                    self.preview_page.image_loader_thread = None

        # 在切换页面前,保存当前页面的选择
        if self.current_page == "preview":
            if self.preview_page.file_listbox.currentItem():
                self.last_preview_image = self.preview_page.file_listbox.currentItem().text()
        elif self.current_page == "species_validation":
            if self.species_validation_page.species_listbox.currentItem():
                # 从 "物种 (数量)" 的格式中提取物种名
                self.last_validation_species = \
                    self.species_validation_page.species_listbox.currentItem().text().split(' (')[0]
            if self.species_validation_page.species_photo_listbox.currentItem():
                self.last_validation_image = self.species_validation_page.species_photo_listbox.currentItem().text()

        # 在处理时限制页面访问
        if self.is_processing and page_id not in ["settings"]:
            return

        self.sidebar.set_active_button(page_id)

        # 设置当前页面索引
        page_index = {
            "settings": 0,
            "preview": 1,
            "species_validation": 2,
            "advanced": 3,
            "about": 4
        }.get(page_id, 0)

        self.content_stack.setCurrentIndex(page_index)
        self.current_page = page_id

        # 更新状态栏并恢复选择
        if page_id == "settings":
            if self.is_processing:
                # 如果正在处理，则恢复状态栏的进度显示
                # 计算已用时间
                elapsed_time = 0  # 如果需要精确时间，需要在类中添加 start_time 属性
                self._on_progress_updated(
                    self.last_progress_value,
                    self.last_progress_total,
                    elapsed_time,  # 添加 elapsed_time 参数
                    self.last_progress_remaining_time,
                    self.last_progress_speed
                )
            else:
                self.status_bar.status_label.setText("就绪")
        elif page_id == "preview":
            if hasattr(self.start_page, 'get_file_path'):
                file_path = self.start_page.get_file_path()
                if file_path and os.path.isdir(file_path):
                    if hasattr(self.preview_page, 'update_file_list'):
                        self.preview_page.update_file_list(file_path)
                    file_count = self.preview_page.get_file_count() if hasattr(self.preview_page,
                                                                               'get_file_count') else 0
                    self.status_bar.status_label.setText(f"当前文件夹下有 {file_count} 个图像文件")

                    # 恢复预览页面的选择
                    if self.last_preview_image:
                        # 使用QTimer确保列表加载完毕后再选择
                        QTimer.singleShot(100, lambda: self.preview_page.select_file_by_name(self.last_preview_image))
                else:
                    self.status_bar.status_label.setText('请在"开始"页面中设置有效的图像文件路径')
        elif page_id == "species_validation":
            if not self.is_processing:
                self.status_bar.status_label.setText("就绪")
            # 确保物种校验页面的数据被加载
            if hasattr(self.species_validation_page, '_load_species_data'):
                # 使用QTimer延迟执行，确保页面完全显示后再加载数据
                QTimer.singleShot(100, self.species_validation_page._load_species_data)
            if hasattr(self.species_validation_page, '_load_species_buttons'):
                QTimer.singleShot(150, self.species_validation_page._load_species_buttons)

            # 恢复物种校验页面的选择
            if self.last_validation_species and self.last_validation_image:
                # 使用更长的延迟以确保物种和照片列表都已加载
                QTimer.singleShot(250, lambda: self.species_validation_page.select_species_and_image(
                    self.last_validation_species, self.last_validation_image))

        elif page_id in ["advanced", "about"] and not self.is_processing:
            self.status_bar.status_label.setText("就绪")

    def browse_file_path(self):
        """浏览文件路径"""
        folder_selected = QFileDialog.getExistingDirectory(
            self, "选择图像文件所在文件夹"
        )
        if folder_selected:
            if hasattr(self.start_page, 'set_file_path'):
                self.start_page.set_file_path(folder_selected)
            self._validate_and_update_file_path(folder_selected)

    def _validate_and_update_file_path(self, folder_selected):
        """验证和更新文件路径"""
        if hasattr(self.preview_page, 'clear_preview'):
            self.preview_page.clear_preview()

        if not folder_selected:
            self.status_bar.status_label.setText("文件路径已清除")
            self._save_current_settings()
            return

        if os.path.isdir(folder_selected):
            self.get_temp_photo_dir(update=True)
            if hasattr(self.preview_page, 'update_file_list'):
                self.preview_page.update_file_list(folder_selected)
            file_count = self.preview_page.get_file_count() if hasattr(self.preview_page, 'get_file_count') else 0
            self.status_bar.status_label.setText(f"文件路径已设置，找到 {file_count} 个图像文件。")
            self._save_current_settings()
            if self.current_page == "preview":
                self._show_page("preview")
        else:
            QMessageBox.critical(
                self, "路径错误",
                f"提供的图像文件路径不存在或不是一个文件夹:\n'{folder_selected}'"
            )
            self.status_bar.status_label.setText("无效的文件路径")

    def show_params_help(self):
        """显示参数帮助"""
        help_text = """
**检测阈值设置**
- **IOU阈值:** 控制对象检测中非极大值抑制（NMS）的重叠阈值。较高的值会减少重叠框，但可能导致部分目标漏检。
- **置信度阈值:** 检测对象的最小置信度分数。较高的值只显示高置信度的检测结果，减少误检。

**模型加速选项**
- **使用FP16加速:** 使用半精度浮点数进行推理，可以加快速度但可能会略微降低精度。需要兼容的NVIDIA GPU。

**高级检测选项**
- **使用数据增强:** 在测试时使用数据增强（TTA），通过对输入图像进行多种变换并综合结果，可能会提高准确性，但会显著降低处理速度。
- **使用类别无关NMS:** 在所有类别上一起执行NMS，对于检测多种相互重叠的物种可能有用。
    """
        QMessageBox.information(self, "参数说明", help_text.strip())

    def get_temp_photo_dir(self, update=False):
        """获取临时图片目录"""
        if hasattr(self.start_page, 'get_file_path'):
            source_path = self.start_page.get_file_path()
        else:
            return None

        if not source_path:
            return None

        path_hash = hashlib.md5(source_path.encode()).hexdigest()
        base_dir = self.settings_manager.base_dir
        temp_dir = os.path.join(base_dir, "temp", "photo", path_hash)

        if update:
            self.current_temp_photo_dir = temp_dir

        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def clear_image_cache(self):
        """清除图像缓存"""
        cache_dir = os.path.join(self.settings_manager.base_dir, "temp", "photo")
        reply = QMessageBox.question(
            self, "确认清除缓存",
            f"是否清空图片缓存？\n\n此操作将删除以下文件夹及其所有内容：\n{cache_dir}\n\n注意：这不会影响您的原始图片或已保存的结果。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    QMessageBox.information(self, "成功", "图片缓存已成功清除。")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"清除缓存时发生错误：\n{e}")
            else:
                QMessageBox.information(self, "提示", "缓存目录不存在，无需清除。")

    def toggle_processing_state(self):
        """切换处理状态"""
        if not self.is_processing:
            self.check_for_cache_and_process()
        else:
            self.stop_processing()

    def check_for_cache_and_process(self):
        """检查缓存并处理"""
        cache_file = os.path.join(self.settings_manager.settings_dir, "cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if 'processed_files' in cache_data and 'total_files' in cache_data:
                    processed = cache_data.get('processed_files', 0)
                    total = cache_data.get('total_files', 0)
                    file_path = cache_data.get('file_path', '')
                    reply = QMessageBox.question(
                        self, "发现未完成任务",
                        f"检测到上次有未完成的任务，是否继续？\n已处理：{processed}/{total} at {file_path}",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self._load_cache_data_from_file(cache_data)
                        self.start_processing(resume_from=processed)
                        return
            except Exception as e:
                logger.error(f"读取缓存文件失败: {e}")
        self.start_processing()

    def start_processing(self, resume_from=0):
        """开始处理"""
        # 获取处理参数
        file_path = self.start_page.get_file_path() if hasattr(self.start_page, 'get_file_path') else ""
        save_path = file_path
        use_fp16 = self.advanced_page.get_use_fp16() if hasattr(self.advanced_page, 'get_use_fp16') else False

        # 验证输入时不再传入 save_path
        if not self._validate_inputs(file_path):
            return

        if self.is_processing:
            return

        # 显示控制台（但不清空）
        if hasattr(self.start_page, 'show_console'):
            self.start_page.show_console()

        # 只在非恢复模式下清空数据
        if resume_from == 0:
            self.excel_data = []
            self._clear_current_validation_file()
            # 可选：如果你希望在新任务开始时清空控制台，可以在这里添加
            # if hasattr(self.start_page, 'console_output'):
            #     self.start_page.console_output.clear()

        # 设置处理状态
        self._set_processing_state(True)

        self.processing_thread = ProcessingThread(
            self, file_path, save_path, use_fp16, resume_from
        )

        # 连接信号 - 确保使用正确的签名
        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.file_processed.connect(self._on_file_processed)
        self.processing_thread.processing_complete.connect(self._on_processing_complete)
        self.processing_thread.status_message.connect(self._on_status_message)
        self.processing_thread.file_processing_result.connect(self._on_file_processing_result)
        self.processing_thread.current_file_changed.connect(self._on_current_file_changed)
        self.processing_thread.current_file_preview.connect(self._on_current_file_preview)
        self.processing_thread.console_log.connect(self._on_console_log)

        self.processing_thread.start()

    def _on_console_log(self, message: str, color: str = None):
        """处理控制台日志"""
        if hasattr(self.start_page, 'append_console_log'):
            self.start_page.append_console_log(message, color)

    def _set_processing_state(self, is_processing: bool):
        """设置处理状态"""
        self.is_processing = is_processing

        if hasattr(self.start_page, 'set_processing_state'):
            self.start_page.set_processing_state(is_processing)
        if hasattr(self.sidebar, 'set_processing_state'):
            self.sidebar.set_processing_state(is_processing)

        if is_processing:
            # 显示进度条
            self.status_bar.show_progress()
            # 隐藏或更新状态标签
            self.status_bar.status_label.setText("正在处理...")
            if hasattr(self.preview_page, 'set_show_detection'):
                self.preview_page.set_show_detection(True)
        else:
            # 隐藏进度条
            self.status_bar.hide_progress()
            if self.status_bar.status_label.text() != "处理完成!":
                self.status_bar.status_label.setText("就绪")

    def _on_progress_updated(self, current, total, elapsed_time, remaining_time, speed):
        """处理进度更新"""
        # 存储最新的进度信息
        self.last_progress_value = current
        self.last_progress_total = total
        self.last_progress_speed = speed
        self.last_progress_remaining_time = remaining_time
        self.last_progress_elapsed_time = elapsed_time

        # 更新状态栏的 tqdm 风格进度条
        self.status_bar.update_progress(current, total, elapsed_time, remaining_time, speed)

    def _on_processing_complete(self, success):
        """处理完成"""
        self._set_processing_state(False)

        # 隐藏控制台（延迟3秒）
        if hasattr(self.start_page, 'hide_console'):
            QTimer.singleShot(3000, self.start_page.hide_console)

        # 如果当前在物种校验页面，刷新数据
        if self.current_page == "species_validation" and success:
            if hasattr(self.species_validation_page, '_load_species_data'):
                QTimer.singleShot(500, self.species_validation_page._load_species_data)

    def _on_current_file_changed(self, img_path, current_index, total_files):
        """当前处理文件变化处理"""
        filename = os.path.basename(img_path)
        self.current_processing_file = filename  # 保存当前文件名

        # 如果当前在预览页面，则同步显示
        if self.current_page == "preview":
            if hasattr(self.preview_page, 'sync_current_processing_file'):
                self.preview_page.sync_current_processing_file(img_path, current_index, total_files)

    def _on_current_file_preview(self, img_path, detection_info):
        """当前文件预览处理"""
        # 如果当前在预览页面，则显示检测结果
        if self.current_page == "preview":
            if hasattr(self.preview_page, 'sync_current_processing_result'):
                self.preview_page.sync_current_processing_result(img_path, detection_info)

    def _on_file_processing_result(self, img_path, detection_info):
        """处理文件处理结果，同步到预览页面"""
        if hasattr(self.preview_page, 'sync_processing_result'):
            self.preview_page.sync_processing_result(img_path, detection_info)

    def stop_processing(self):
        """停止处理 (第一次点击：等待当前完成；第二次点击：强制退出保存上一进度)"""
        # 基础检查
        if not hasattr(self, 'processing_thread') or self.processing_thread is None or not self.processing_thread.isRunning():
            return

        # 获取当前状态
        # 如果 stop_flag 为 True，说明用户之前已经点过一次，现在是第二次点击
        is_already_stopping = self.processing_thread.stop_flag

        if not is_already_stopping:
            # === 第一次点击 ===
            reply = QMessageBox.question(
                self, "停止确认",
                "确定要停止吗？\n\n点击【是】将等待当前正在处理的图片/视频完成后再停止。\n(进度将包含当前文件)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.status_bar.status_label.setText("正在停止... (等待当前文件处理完毕，再次点击可强制停止)")
                self.processing_thread.stop_flag = True
                if hasattr(self.processing_thread, 'console_log'):
                    self.processing_thread.console_log.emit("[INFO] 已收到停止请求，将在当前文件完成后自动停止...", "#ffff00")

        else:
            # === 第二次点击 ===
            reply = QMessageBox.question(
                self, "强制停止",
                "检测到已发出停止请求。\n\n是否【强制立即退出】？\n(注意：当前正在处理的文件进度将丢失，进度将回滚到上一个文件)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.status_bar.status_label.setText("正在强制停止并回滚进度...")
                # 设置强制停止标志，这将触发 run 方法内部的 ForceStopError
                self.processing_thread.force_stop_flag = True

    def _validate_inputs(self, file_path: str) -> bool:
        """验证输入参数 """
        if not file_path or not os.path.isdir(file_path):
            QMessageBox.critical(self, "错误", "请提供有效的源文件夹路径。")
            return False
        return True

    def _on_file_processed(self, img_path, detection_results, filename):
        """处理文件完成"""
        if hasattr(self.preview_page, 'on_file_processed'):
            self.preview_page.on_file_processed(img_path, detection_results, filename)

    def _on_processing_complete(self, success):
        """处理完成"""
        self._set_processing_state(False)

        # 如果当前在物种校验页面，刷新数据
        if self.current_page == "species_validation" and success:
            if hasattr(self.species_validation_page, '_load_species_data'):
                QTimer.singleShot(500, self.species_validation_page._load_species_data)

    def _on_status_message(self, message):
        """状态消息 - 只处理非进度相关的消息"""
        if "正在处理:" not in message and "处理中:" not in message:
            self.status_bar.status_label.setText(message)

    def _save_current_settings(self):
        """保存当前设置"""
        if not self.settings_manager:
            return
        settings = self._get_current_settings()
        if self.settings_manager.save_settings(settings):
            logger.info("设置已保存")

    def _get_current_settings(self):
        """获取当前设置"""
        settings = {}

        # 从开始页面获取设置
        if hasattr(self.start_page, 'get_settings'):
            settings.update(self.start_page.get_settings())

        # 从高级页面获取设置
        if hasattr(self.advanced_page, 'get_settings'):
            settings.update(self.advanced_page.get_settings())

        # 从预览页面获取设置
        if hasattr(self.preview_page, 'get_settings'):
            settings.update(self.preview_page.get_settings())

            # 从物种校验页面获取设置
            if hasattr(self.species_validation_page, 'get_settings'):
                settings.update(self.species_validation_page.get_settings())

        # 添加其他设置，确保包含模型信息
        settings.update({
            "selected_model": getattr(self, 'model_var', ''),
        })

        # 如果model_var为空，尝试从image_processor获取
        if not settings.get("selected_model") and hasattr(self, 'image_processor'):
            if hasattr(self.image_processor, 'model_path') and self.image_processor.model_path:
                settings["selected_model"] = os.path.basename(self.image_processor.model_path)

        return settings

    def _load_settings_to_ui(self, settings: dict):
        """加载设置到UI"""
        if not settings:
            return
        try:
            # 加载到各个页面
            if hasattr(self.start_page, 'load_settings'):
                self.start_page.load_settings(settings)
            if hasattr(self.advanced_page, 'load_settings'):
                self.advanced_page.load_settings(settings)
            if hasattr(self.preview_page, 'load_settings'):
                self.preview_page.load_settings(settings)
            if hasattr(self.species_validation_page, 'load_settings'):
                self.species_validation_page.load_settings(settings)

            # 加载主题
            theme = settings.get("theme", "自动")
            if hasattr(self.advanced_page, 'set_theme_selection'):
                self.advanced_page.set_theme_selection(theme)
            self.change_theme()

        except Exception as e:
            logger.error(f"加载设置到UI失败: {e}")

    def _load_cache_data_from_file(self, cache_data):
        """从文件加载缓存数据"""
        self._load_settings_to_ui(cache_data)
        self.excel_data = cache_data.get('excel_data', [])
        for item in self.excel_data:
            if '拍摄日期对象' in item and isinstance(item['拍摄日期对象'], str):
                try:
                    item['拍摄日期对象'] = datetime.fromisoformat(item['拍摄日期对象'])
                except ValueError:
                    item['拍摄日期对象'] = None

    def _resume_processing(self):
        """恢复处理"""
        self._load_cache_data_from_file(self.cache_data)
        self.start_processing(resume_from=self.cache_data.get('processed_files', 0))

    def _clear_current_validation_file(self):
        """清除当前验证文件"""
        temp_photo_dir = self.get_temp_photo_dir()
        if temp_photo_dir:
            validation_file_path = os.path.join(temp_photo_dir, "validation.json")
            if os.path.exists(validation_file_path):
                try:
                    os.remove(validation_file_path)
                    logger.info(f"已清除旧的校验文件: {validation_file_path}")
                except Exception as e:
                    logger.error(f"清除旧的校验文件失败: {e}")

        # 清除内存中的数据
        if hasattr(self.preview_page, 'clear_validation_data'):
            self.preview_page.clear_validation_data()

    def closeEvent(self, event):
        """关闭事件"""
        if self.is_processing:
            reply = QMessageBox.question(
                self, "确认退出",
                "图像处理正在进行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            # 检查处理线程是否存在且有stop方法
            if hasattr(self, 'processing_thread') and self.processing_thread is not None:
                if hasattr(self.processing_thread, 'stop'):
                    self.processing_thread.stop()
                else:
                    # 如果没有stop方法，设置停止标志
                    if hasattr(self.processing_thread, 'stop_flag'):
                        self.processing_thread.stop_flag = True

        # 保存验证数据
        if hasattr(self.preview_page, '_save_validation_data'):
            self.preview_page._save_validation_data()

        # 保存设置
        self._save_current_settings()

        event.accept()

    def _set_title_bar_color(self):
        """设置窗口标题栏颜色 (仅限Windows)"""
        if platform.system() != "Windows":
            return

        try:
            from PySide6.QtGui import QColor

            # Constant for DWMWA_CAPTION_COLOR from Windows API
            DWMWA_CAPTION_COLOR = 35

            if self.is_dark_mode:
                # 深色模式标题栏颜色
                color_str = "#5d3a4f"
            else:
                # 浅色模式标题栏颜色
                color_str = "#f6dce0"

            color = QColor(color_str)
            # Windows API COLORREF is in 0x00BBGGRR format
            color_ref = color.blue() << 16 | color.green() << 8 | color.red()

            hwnd = self.winId()
            if hwnd:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    ctypes.c_void_p(hwnd),
                    DWMWA_CAPTION_COLOR,
                    ctypes.byref(ctypes.c_int(color_ref)),
                    ctypes.sizeof(ctypes.c_int)
                )
        except Exception as e:
            logger.warning(f"无法设置标题栏颜色: {e}")

    @Slot(str, str, str)
    def prompt_for_update(self, remote_version, download_url, release_notes):
        """弹窗询问用户是否更新，并在主线程中安全地启动下载。"""
        if not download_url:
            QMessageBox.critical(self, "更新错误", "找不到新版本的下载链接。")
            return

        # 创建自定义 Dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("发现新版本")
        dialog.setMinimumWidth(350)  # 设置最小宽度

        layout = QVBoxLayout(dialog)

        # 1. 标题和版本号
        title_label = QLabel(f"发现新版本 ({remote_version})")
        font = title_label.font()
        font.setBold(True)
        font.setPointSize(11)
        title_label.setFont(font)
        layout.addWidget(title_label)

        # 2. 询问文本
        layout.addWidget(QLabel("是否立即下载并更新？"))
        layout.addSpacing(10)

        # 3. 更新内容标题
        header_label = QLabel("=== 更新内容 ===")
        # 设置粗体
        header_font = header_label.font()
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # 4. 滚动文本区域 (支持 Markdown)
        content_browser = QTextBrowser()
        content_browser.setMarkdown(release_notes)
        content_browser.setOpenExternalLinks(True)  # 允许点击链接
        content_browser.setReadOnly(True)
        # [关键] 设置固定高度，内容过多时自动出现滚动条
        content_browser.setFixedHeight(300)
        layout.addWidget(content_browser)

        # 5. 按钮区域 (Yes/No)
        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        # 设置按钮显示的文本为中文（可选，取决于系统语言，通常 StandardButton 会自动适配）
        # 如果需要强制中文，可以手动添加按钮

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # 显示对话框并处理结果
        if dialog.exec() == QDialog.Accepted:
            start_download_thread(self, download_url)

    @Slot(str)
    def set_status_bar_message(self, message: str):
        """安全地设置状态栏消息（可从其他线程调用）"""
        self.status_bar.status_label.setText(message)