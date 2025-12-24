# system/image_processor.py

import os
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Union
from collections import Counter, defaultdict
from ultralytics import YOLO
import json
import torch
from system.utils import resource_path
import cv2

logger = logging.getLogger(__name__)


class ImageProcessor:
    """处理图像、检测物种及视频追踪的核心类"""

    def __init__(self, model_path: str):
        """初始化图像处理器"""
        self.model = self._load_model(model_path)
        self.translation_dict = self._load_translation_file()

    def _load_model(self, model_path: str) -> Optional[YOLO]:
        """加载YOLO模型"""
        try:
            logger.info(f"正在加载模型: {model_path}")
            return YOLO(model_path)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None

    def _load_translation_file(self) -> Dict[str, str]:
        """加载翻译文件"""
        try:
            translate_file_path = resource_path("res/translate.json")
            if os.path.exists(translate_file_path):
                with open(translate_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning("翻译文件 res/translate.json 未找到，将使用原始英文名称。")
                return {}
        except Exception as e:
            logger.error(f"加载或解析翻译文件失败: {e}")
            return {}

    def _check_cuda(self, use_fp16: bool) -> bool:
        """检查CUDA可用性并决定是否使用FP16"""
        try:
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                return False
            return use_fp16
        except ImportError:
            return False
        except Exception:
            return False

    def detect_species(self, img_path: str, use_fp16: bool = False, iou: float = 0.3,
                       conf: float = 0.25, augment: bool = True,
                       agnostic_nms: bool = True, timeout: float = 20.0) -> Dict[str, Any]:
        """检测图像中的物种并应用翻译"""
        use_fp16 = self._check_cuda(use_fp16)

        species_names = ""
        species_counts = ""
        detect_results = None
        min_confidence = None

        if not self.model:
            return {
                '物种名称': species_names,
                '物种数量': species_counts,
                'detect_results': detect_results,
                '最低置信度': min_confidence
            }

        def run_detection():
            nonlocal species_names, species_counts, detect_results, min_confidence
            try:
                results = self.model(
                    img_path,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    imgsz=1024,
                    half=use_fp16,
                    iou=iou,
                    conf=conf
                )
                detect_results = results

                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue

                    data_list = r.boxes.cls.tolist()
                    counts = Counter(data_list)
                    species_dict = r.names
                    confidences = r.boxes.conf.tolist()

                    if confidences:
                        current_min_confidence = min(confidences)
                        if min_confidence is None or current_min_confidence < min_confidence:
                            min_confidence = "%.3f" % current_min_confidence

                    detected_species_counts = {}
                    for element, count in counts.items():
                        english_name = species_dict.get(int(element), "unknown")
                        translated_name = self.translation_dict.get(english_name, english_name)

                        if translated_name in detected_species_counts:
                            detected_species_counts[translated_name] += count
                        else:
                            detected_species_counts[translated_name] = count

                    species_list = list(detected_species_counts.keys())
                    counts_list = list(map(str, detected_species_counts.values()))

                    species_names = ",".join(species_list)
                    species_counts = ",".join(counts_list)

                return True
            except Exception as e:
                logger.error(f"物种检测失败: {e}")
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_detection)
            try:
                success = future.result(timeout=timeout)
                if not success:
                    raise Exception("检测过程出错")
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"物种检测超时（>{timeout}秒）")

        return {
            '物种名称': species_names if species_names else "空",
            '物种数量': species_counts if species_counts else "空",
            'detect_results': detect_results,
            '最低置信度': min_confidence
        }

    # ==========================================
    # 新增视频检测方法
    # ==========================================
    def detect_video_species(self, video_source: str, output_dir: str,
                             use_fp16: bool = False, iou: float = 0.3,
                             conf: float = 0.25, augment: bool = True,
                             agnostic_nms: bool = True,
                             status_callback: Optional[Any] = None,
                             vid_stride: int = 1,
                             temp_video_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        对视频进行物种检测和追踪。
        """
        if hasattr(self, 'model_path') and self.model_path:
            try:
                logger.info(f"正在重置模型以清理追踪器状态: {self.model_path}")
                # 重新加载模型实例，彻底清空追踪器历史
                self.model = self._load_model(self.model_path)
            except Exception as e:
                logger.warning(f"重置模型状态失败，将尝试使用当前模型继续: {e}")

        if not self.model:
            logger.error("模型未加载，无法处理视频")
            return {'error': 'Model not loaded'}

        use_fp16 = self._check_cuda(use_fp16)

        # 获取视频总帧数
        total_frames = 0
        try:
            cap = cv2.VideoCapture(video_source)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            logger.warning(f"无法获取视频总帧数: {e}")

        # 1. 获取Tracker配置文件路径
        tracker_config = resource_path(os.path.join("res", "model", "tracker.yaml"))
        if not os.path.exists(tracker_config):
            logger.warning(f"Tracker配置文件未找到: {tracker_config}，将使用默认追踪器")
            tracker_config = "botsort.yaml"  # Ultralytics 默认值

        # 2. 准备路径
        # [修改] 仅标准化 output_dir 路径，但不立即创建文件夹，避免生成空的 video_results
        output_dir = os.path.normpath(output_dir)

        # [修改] 移除之前的 project_path 和 run_name 计算，不再让 YOLO 输出到该目录
        # project_path = os.path.dirname(output_dir)
        # run_name = os.path.basename(output_dir)

        video_name = os.path.splitext(os.path.basename(video_source))[0]
        if "http" in video_source:
            video_name = "stream_result"

        # 3. 运行追踪 (Stream=True 以减少内存占用)
        logger.info(f"开始视频追踪: {video_source} (跳帧: {vid_stride})")

        # [新增] 使用系统临时目录作为 YOLO 的运行目录，防止在用户目录下生成 runs 文件夹
        import tempfile
        temp_run_project = os.path.join(tempfile.gettempdir(), "neri_yolo_logs")

        try:
            # save=False 禁止生成视频文件
            results = self.model.track(
                source=video_source,
                tracker=tracker_config,
                augment=augment,
                agnostic_nms=agnostic_nms,
                imgsz=1024,
                half=use_fp16,
                iou=iou,
                conf=conf,
                persist=True,  # 视频追踪必须开启 persist
                save=False,  # [修改] 显式禁止保存视频
                project=temp_run_project,  # [修改] 重定向到临时目录
                name="track_log",  # [修改] 临时任务名
                exist_ok=True,
                stream=True,
                vid_stride=vid_stride  # 传入跳帧参数
            )

            # 4. 数据聚合结构
            tracks_data = defaultdict(list)
            frame_count = 0

            # 5. 逐帧处理结果
            for r in results:
                frame_count += 1

                # 计算当前实际对应的视频帧索引
                # frame_count 是处理过的帧数，需要乘以 stride 还原为原始视频帧索引
                current_real_frame_idx = (frame_count - 1) * vid_stride

                # 处理回调状态
                if status_callback:
                    try:
                        frame_counts = Counter()
                        if r.boxes and r.boxes.cls is not None:
                            for cls_id in r.boxes.cls.int().tolist():
                                name = r.names[cls_id]
                                trans_name = self.translation_dict.get(name, name)
                                frame_counts[trans_name] += 1

                        speed_ms = 0.0
                        if hasattr(r, 'speed') and isinstance(r.speed, dict):
                            speed_ms = sum(r.speed.values())

                        h, w = r.orig_shape if hasattr(r, 'orig_shape') else (0, 0)

                        # 回调中使用实际帧进度，以便进度条正确显示
                        current_progress = min(current_real_frame_idx + 1, total_frames)
                        status_callback(current_progress, total_frames, w, h, frame_counts, speed_ms)

                    except Exception as e:
                        # 检测是否为强制停止信号
                        # 如果异常信息中包含"强制停止"（这是我们在 main_window 中抛出的），
                        # 则不再视为错误，而是直接向上抛出异常，从而跳出 for r in results 循环，停止 YOLO 追踪。
                        if "强制停止" in str(e):
                            logger.info(f"响应用户停止信号，正在中断视频追踪: {e}")
                            raise e  # 重新抛出异常，让外层的 ForceStopError 捕获逻辑生效

                        # 只有非停止信号的异常才记录为错误
                        logger.error(f"视频状态回调出错: {e}")

                if r.boxes is None or r.boxes.id is None:
                    continue

                # 获取转为列表的数据
                ids = r.boxes.id.int().cpu().tolist()
                classes = r.boxes.cls.int().cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                boxes = r.boxes.xyxy.cpu().tolist()

                for track_id, cls_id, conf_val, box_val in zip(ids, classes, confs, boxes):
                    english_name = r.names[cls_id]
                    translated_name = self.translation_dict.get(english_name, english_name)

                    entry = {
                        "frame_index": current_real_frame_idx,  # [修改] 使用还原后的实际帧索引
                        "species": translated_name,
                        "original_species": english_name,
                        "confidence": float(conf_val),
                        "bbox": [float(x) for x in box_val]
                    }
                    tracks_data[track_id].append(entry)

            # 6. 保存JSON结果
            # 只有在需要保存 JSON 时才创建目标文件夹
            target_json_dir = temp_video_dir if temp_video_dir else output_dir
            os.makedirs(target_json_dir, exist_ok=True)

            json_output_path = os.path.join(target_json_dir, f"{video_name}.json")

            final_json_data = {
                "video_source": video_source,
                "total_frames_processed": frame_count,
                "vid_stride": vid_stride,
                "tracker_config": tracker_config,
                "tracks": dict(tracks_data)
            }

            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_data, f, ensure_ascii=False, indent=4)

            logger.info(f"视频处理完成，JSON已保存至: {json_output_path}")
            return {
                "json_path": json_output_path,
                "frame_count": frame_count,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"视频追踪失败: {e}")
            return {"error": str(e), "status": "failed"}

    def save_detection_result(self, results: Any, image_name: str, save_path: str) -> None:
        """保存探测结果图片"""
        if not results:
            return

        try:
            result_path = os.path.join(save_path, "result")
            os.makedirs(result_path, exist_ok=True)

            for c, h in enumerate(results):
                species_name = self._get_first_detected_species(results)
                result_file = os.path.join(result_path, f"{image_name}_result_{species_name}.jpg")
                h.save(filename=result_file)
        except Exception as e:
            logger.error(f"保存检测结果图片失败: {e}")

    def _get_first_detected_species(self, results: Any) -> str:
        """从检测结果中获取第一个物种的名称"""
        try:
            for r in results:
                if r.boxes and len(r.boxes.cls) > 0:
                    return r.names[int(r.boxes.cls[0].item())]
        except Exception as e:
            logger.error(f"获取物种名称失败: {e}")
        return "unknown"

    def save_detection_temp(self, results: Any, image_name: str, temp_photo_dir: str) -> str:
        """保存探测结果图片到指定的临时目录"""
        if not results or not temp_photo_dir:
            return ""

        try:
            os.makedirs(temp_photo_dir, exist_ok=True)
            result_file = os.path.join(temp_photo_dir, image_name)
            for h in results:
                from PIL import Image
                result_img = h.plot()
                result_img = Image.fromarray(result_img[..., ::-1])
                result_img.save(result_file, "JPEG", quality=95)
                return result_file
        except Exception as e:
            logger.error(f"保存临时检测结果图片失败: {e}")
            return ""

    def save_detection_info_json(self, results, image_name: str, species_info: dict, temp_photo_dir: str) -> str:
        """保存探测结果信息到指定的临时目录 (用于单张图片)"""
        if not results or not temp_photo_dir:
            return ""

        try:
            import json
            os.makedirs(temp_photo_dir, exist_ok=True)
            data_to_save = {
                "物种名称": species_info.get('物种名称', ''),
                "物种数量": species_info.get('物种数量', ''),
                "最低置信度": species_info.get('最低置信度', ''),
                "检测时间": species_info.get('检测时间', '')
            }
            boxes_info = []
            all_confidences = []
            all_classes = []
            names_map = {}

            if results:
                for r in results:
                    original_names_map = r.names
                    translated_names_map = {
                        class_id: self.translation_dict.get(english_name, english_name)
                        for class_id, english_name in original_names_map.items()
                    }
                    names_map = translated_names_map
                    if r.boxes is not None:
                        for i, box in enumerate(r.boxes):
                            cls_id = int(box.cls.item())
                            species_name = r.names[cls_id]

                            translated_name = self.translation_dict.get(species_name, species_name)

                            confidence = float(box.conf.item())
                            bbox = [float(x) for x in box.xyxy.tolist()[0]]

                            box_info = {"物种": translated_name, "置信度": confidence, "边界框": bbox}

                            boxes_info.append(box_info)
                        all_confidences = r.boxes.conf.tolist()
                        all_classes = r.boxes.cls.tolist()

            data_to_save["检测框"] = boxes_info
            data_to_save["all_confidences"] = all_confidences
            data_to_save["all_classes"] = all_classes
            data_to_save["names_map"] = names_map

            base_name, _ = os.path.splitext(image_name)
            json_path = os.path.join(temp_photo_dir, f"{base_name}.json")

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)

            return json_path
        except Exception as e:
            logger.error(f"保存检测结果JSON失败: {e}")
            return ""

    def load_model(self, model_path: str) -> None:
        """加载新的模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_path = model_path
            logger.info(f"模型已加载: {model_path}")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise Exception(f"加载模型失败: {e}")