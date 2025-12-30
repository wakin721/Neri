# system/image_processor.py

import os
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Union
from collections import Counter, defaultdict
from ultralytics import YOLO
import json
import torch
import numpy as np
from system.utils import resource_path
import cv2

logger = logging.getLogger(__name__)


class ImageProcessor:
    """处理图像、检测物种及视频追踪的核心类"""

    def __init__(self, model_path: str):
        """初始化图像处理器"""
        self.model = self._load_model(model_path)
        self.translation_dict = self._load_translation_file()
        self.cls_model = None

    def _load_model(self, model_path: str) -> Optional[YOLO]:
        """加载YOLO模型"""
        try:
            logger.info(f"正在加载模型: {model_path}")
            return YOLO(model_path)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None

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

    def load_cls_model(self, model_path: str) -> None:
        """加载分类模型"""
        try:
            if not model_path:
                self.cls_model = None
                logger.info("分类模型已卸载")
                return
            logger.info(f"正在加载分类模型: {model_path}")
            self.cls_model = YOLO(model_path)
        except Exception as e:
            logger.error(f"加载分类模型失败: {e}")
            self.cls_model = None

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

    def _preprocess_image(self, img: Any) -> Any:
        """
        图像预处理：LAB色彩空间增强 (L通道 CLAHE)
        适用于 BGR 彩色图像和 灰度图像
        """
        if img is None or img.size == 0:
            return None

        try:
            # [新增] 确保图像是 uint8 类型且内存连续，防止 YOLO 报错 Unsupported image type
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # 1. 灰度图处理 (2维数组)
            if len(img.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img)
                return np.ascontiguousarray(enhanced)

            # 2. 彩色图处理 (3维数组 BGR)
            elif len(img.shape) == 3:
                # BGR -> LAB
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # 只对 L 通道 (亮度) 进行 CLAHE 增强
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)

                # 合并通道并转回 BGR
                merged = cv2.merge((l_enhanced, a, b))
                bgr_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

                # [重要] 返回连续数组
                return np.ascontiguousarray(bgr_enhanced)

        except Exception as e:
            logger.warning(f"图像预处理失败，将使用原图: {e}")
            return np.ascontiguousarray(img) if img is not None else None

        return img

    def _apply_temperature_scaling(self, probs: torch.Tensor, temperature: float = 3.0) -> torch.Tensor:
        """
        标准温度缩放 (Temperature Scaling)：
        直接利用 Softmax 的性质平滑概率分布。

        Args:
            probs: 原始概率分布 (Tensor)
            temperature: 温度系数 (T > 1 平滑分布; T < 1 锐化分布; T = 1 原样)
                         建议设置在 2.0 - 5.0 之间以解决过度自信问题。
        """
        try:
            # 如果温度系数无效或为1，直接返回
            if temperature <= 0 or temperature == 1.0:
                return probs

            # 1. 反推 Logits (添加 epsilon 防止 log(0) 得到 -inf)
            eps = 1e-9
            # 注意：如果 probs 中有 0，log 后会变成负无穷，为了数值稳定性，限制最小值为 eps
            safe_probs = torch.clamp(probs, min=eps)
            logits = torch.log(safe_probs)

            # 2. 应用温度系数缩放
            # T 越大，logits 之间的差异越小
            scaled_logits = logits / temperature

            # 3. 重新计算 Softmax
            return torch.nn.functional.softmax(scaled_logits, dim=0)

        except Exception as e:
            logger.warning(f"温度缩放失败: {e}")
            return probs

    def detect_species(self, img_path: str, use_fp16: bool = False, iou: float = 0.3,
                       conf: float = 0.25, augment: bool = True,
                       agnostic_nms: bool = True, timeout: float = 20.0) -> Dict[str, Any]:
        """检测图像中的物种并应用翻译 (包含 LAB 预处理增强)"""
        use_fp16 = self._check_cuda(use_fp16)

        species_names = ""
        species_counts = ""
        detect_results = None
        min_confidence = None

        if not self.model:
            return {
                '物种名称': species_names, '物种数量': species_counts,
                'detect_results': detect_results, '最低置信度': min_confidence
            }

        def run_detection():
            nonlocal species_names, species_counts, detect_results, min_confidence
            try:
                # [修改] 1. 读取并预处理图像
                original_img_bgr = cv2.imread(img_path)
                if original_img_bgr is None:
                    raise FileNotFoundError(f"无法读取图像: {img_path}")

                # 应用增强 (LAB -> CLAHE -> BGR)
                processed_img = self._preprocess_image(original_img_bgr)

                # [修改] 2. 运行检测模型 (传入 numpy 数组而不是路径)
                results = self.model(
                    processed_img,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    imgsz=1024,
                    half=use_fp16,
                    iou=iou,
                    conf=conf,
                    max_det=20
                )

                # 3. 如果启用了分类模型，进行二次识别
                candidates_map = {}

                if self.cls_model:
                    try:
                        # [修改] 使用预处理后的图像进行裁切，确保分类器也能受益于增强
                        # 这是一个优化点：不需要再次读取 cv2.imread
                        original_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

                        for r in results:
                            if r.boxes is None: continue
                            for i, box in enumerate(r.boxes):
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                h, w, _ = original_img_rgb.shape
                                # 设定扩展比例，例如 0.2 表示每边向外扩展 20%
                                expand_ratio = 0.1

                                box_width = x2 - x1
                                box_height = y2 - y1

                                pad_w = int(box_width * expand_ratio)
                                pad_h = int(box_height * expand_ratio)

                                # 应用扩展并进行边界检查，防止越界
                                x1 = max(0, x1 - pad_w)
                                y1 = max(0, y1 - pad_h)
                                x2 = min(w, x2 + pad_w)
                                y2 = min(h, y2 + pad_h)

                                if x2 > x1 and y2 > y1:
                                    crop = original_img_rgb[y1:y2, x1:x2]

                                    ch, cw = crop.shape[:2]
                                    if ch != cw:
                                        max_dim = max(ch, cw)
                                        # Calculate padding sizes
                                        top = (max_dim - ch) // 2
                                        bottom = max_dim - ch - top
                                        left = (max_dim - cw) // 2
                                        right = max_dim - cw - left

                                        # Use gray padding (114, 114, 114) which is standard for YOLO
                                        crop = cv2.copyMakeBorder(
                                            crop,
                                            top, bottom, left, right,
                                            cv2.BORDER_CONSTANT,
                                            value=[114, 114, 114]
                                        )

                                    # 运行分类模型
                                    cls_res = self.cls_model(crop,
                                                             half=use_fp16
                                                             )

                                    # 获取原始概率张量 (Tensor)
                                    original_probs = cls_res[0].probs.data

                                    # 应用温度缩放 (temperature=3.0 可根据实际需求调整，越大越平缓)
                                    smoothed_probs = self._apply_temperature_scaling(original_probs, temperature=3.0)

                                    # 手动获取前 3 名 (TopK)
                                    # values: 置信度, indices: 类别索引
                                    topk_confs, topk_indices = torch.topk(smoothed_probs, 3)

                                    # 转为列表处理
                                    top3_indices = topk_indices.tolist()
                                    top3_confs = topk_confs.tolist()

                                    candidates = []
                                    for cls_idx, cls_conf in zip(top3_indices, top3_confs):
                                        # 注意：这里需要通过 names 字典获取原始名称
                                        raw_name = cls_res[0].names[int(cls_idx)]

                                        # 翻译名称
                                        trans_name = self.translation_dict.get(raw_name, raw_name)
                                        candidates.append({
                                            "name": trans_name,
                                            "conf": float(cls_conf)
                                        })

                                    # 存储候选信息，Key为box的索引或其他标识
                                    candidates_map[i] = candidates

                                    # [可选] 用分类模型的第一名替换检测模型的类别用于初步统计
                                    # 这里我们不直接修改box.cls，而是在统计时优先使用candidates[0]
                    except Exception as e:
                        logger.error(f"二次分类过程出错: {e}")

                detect_results = results

                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue

                    species_dict = r.names
                    confidences = r.boxes.conf.tolist()

                    if confidences:
                        current_min_confidence = min(confidences)
                        if min_confidence is None or current_min_confidence < min_confidence:
                            min_confidence = "%.3f" % current_min_confidence

                    detected_species_counts = {}

                    for i, box in enumerate(r.boxes):
                        final_name = ""

                        # 1. 优先检查是否存在分类模型的结果 (candidates_map)
                        # candidates_map[i] 是一个列表，第0个元素即为置信度最高的分类结果
                        if i in candidates_map and candidates_map[i]:
                            # 提取分类模型的第一名名称
                            final_name = candidates_map[i][0]['name']
                        else:
                            # 2. 如果没有分类结果，降级使用检测模型原本的类别
                            cls_id = int(box.cls.item())
                            raw_name = r.names[cls_id]
                            final_name = self.translation_dict.get(raw_name, raw_name)

                        # 3. 进行计数
                        detected_species_counts[final_name] = detected_species_counts.get(final_name, 0) + 1

                        # [重要] 将分类候选项注入到 results 对象中
                        # 这样后续保存 JSON (save_detection_info_json) 时也能读取到这些修正后的信息
                        if not hasattr(r, 'candidates_data'):
                            r.candidates_data = {}
                        if i in candidates_map:
                            r.candidates_data[i] = candidates_map[i]

                        # 生成统计字符串
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

    def _create_temp_enhanced_video(self, source_path: str, temp_path: str, stride: int) -> int:
        """
        读取源视频，应用跳帧和LAB增强，保存为临时MP4文件。
        返回生成的视频的总帧数。
        """
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise Exception(f"无法打开源视频: {source_path}")

        # 获取原始信息
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0: orig_fps = 25  # 默认值防止错误

        # 保持原始分辨率，不进行缩放
        new_w = orig_w
        new_h = orig_h

        # 初始化写入器 (使用 mp4v 编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_path, fourcc, orig_fps, (new_w, new_h))

        if not writer.isOpened():
            cap.release()
            raise Exception("无法创建临时视频写入器")

        idx = 0
        saved_count = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                # 1. 跳帧处理
                if idx % stride == 0:
                    if frame is not None and frame.size > 0:
                        # [修改] 移除 resize 步骤，直接对原尺寸图像进行 LAB 增强
                        enhanced_frame = self._preprocess_image(frame)

                        # 2. 写入临时视频
                        if enhanced_frame is not None:
                            writer.write(enhanced_frame)
                            saved_count += 1

                idx += 1
        finally:
            cap.release()
            writer.release()

        return saved_count

    def detect_video_species(self, video_source: str, output_dir: str,
                             use_fp16: bool = False, iou: float = 0.3,
                             conf: float = 0.25, augment: bool = True,
                             agnostic_nms: bool = True,
                             status_callback: Optional[Any] = None,
                             vid_stride: int = 1,
                             temp_video_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        对视频进行物种检测和追踪。
        策略：先生成跳帧+增强后的临时视频(保持原分辨率)，再进行追踪。
        """
        if hasattr(self, 'model_path') and self.model_path:
            try:
                self.model = self._load_model(self.model_path)
            except Exception as e:
                logger.warning(f"重置模型状态失败: {e}")

        if not self.model: return {'error': 'Model not loaded'}
        use_fp16 = self._check_cuda(use_fp16)

        # 准备路径
        output_dir = os.path.normpath(output_dir)
        video_name = os.path.splitext(os.path.basename(video_source))[0]
        if "http" in video_source: video_name = "stream_result"

        # 确定临时文件夹
        work_temp_dir = temp_video_dir if temp_video_dir else os.path.join(output_dir, "temp")
        os.makedirs(work_temp_dir, exist_ok=True)

        # 定义临时增强视频路径
        temp_enhanced_video_path = os.path.join(work_temp_dir, f"{video_name}_enhanced_temp.mp4")

        # YOLO 日志路径
        import tempfile
        temp_run_project = os.path.join(tempfile.gettempdir(), "neri_yolo_logs")

        # 追踪器配置
        tracker_config = resource_path(os.path.join("res", "model_cls", "tracker.yaml"))
        if not os.path.exists(tracker_config): tracker_config = "botsort.yaml"

        logger.info(f"开始预处理视频 (LAB增强, 保持原分辨率): {video_source}")

        try:
            # === 第一步：生成增强后的临时视频 ===
            # processed_frame_count 是实际生成的帧数（已包含跳帧逻辑）
            # 这将作为进度条的“总帧数”
            processed_frame_count = self._create_temp_enhanced_video(
                video_source, temp_enhanced_video_path, vid_stride
            )

            if processed_frame_count == 0:
                raise Exception("预处理后未生成有效帧")

            logger.info(f"预处理完成，生成临时视频: {temp_enhanced_video_path} (共 {processed_frame_count} 帧)")

            # === 第二步：运行 YOLO 追踪 ===
            # source 直接传入临时视频路径
            # imgsz=1024 仍保留作为推理尺寸，YOLO 会自动 resize 输入网络，不影响结果
            # vid_stride=1 必须为1，因为我们在第一步已经物理删除了不需要的帧
            results = self.model.track(
                source=temp_enhanced_video_path,
                tracker=tracker_config,
                augment=augment,
                agnostic_nms=agnostic_nms,
                imgsz=1024,
                half=use_fp16,
                iou=iou,
                conf=conf,
                persist=True,
                save=False,
                project=temp_run_project,
                name="track_log",
                exist_ok=True,
                stream=True,
                vid_stride=1  # 关键：不要让 YOLO 再次跳帧
            )

            tracks_data = defaultdict(list)
            current_track_frame = 0

            # === 第三步：处理结果并同步进度条 ===
            for r in results:
                current_track_frame += 1

                # 计算对应的原始视频帧索引 (用于数据记录)
                original_real_frame_idx = (current_track_frame - 1) * vid_stride

                # --- [修改核心] 状态回调更新 (用于 UI 进度条) ---
                if status_callback:
                    try:
                        # 1. 统计当前帧内的物种数量（用于实时显示）
                        frame_counts = Counter()
                        if r.boxes and r.boxes.cls is not None:
                            for cls_id in r.boxes.cls.int().tolist():
                                name = r.names[cls_id]
                                trans_name = self.translation_dict.get(name, name)
                                frame_counts[trans_name] += 1

                        # 2. 计算推理速度（用于显示 FPS 或延迟）
                        speed_ms = 0.0
                        if hasattr(r, 'speed') and isinstance(r.speed, dict):
                            speed_ms = sum(r.speed.values())

                        # 3. 获取当前帧的尺寸
                        h, w = r.orig_shape if hasattr(r, 'orig_shape') else (0, 0)

                        # 4. [关键] 调用回调函数
                        # current_track_frame: 当前处理到的帧数（分子）
                        # processed_frame_count: 临时视频的总帧数（分母，由 _create_temp_enhanced_video 返回）
                        status_callback(current_track_frame, processed_frame_count, w, h, frame_counts, speed_ms)

                    except Exception as e:
                        if "强制停止" in str(e): raise e
                        logger.error(f"视频状态回调出错: {e}")

                if r.boxes is None or r.boxes.id is None: continue

                ids = r.boxes.id.int().cpu().tolist()
                classes = r.boxes.cls.int().cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                boxes = r.boxes.xyxy.cpu().tolist()

                for track_id, cls_id, conf_val, box_val in zip(ids, classes, confs, boxes):
                    english_name = r.names[cls_id]
                    translated_name = self.translation_dict.get(english_name, english_name)

                    entry = {
                        "frame_index": original_real_frame_idx,  # 记录原始视频的时间点
                        "species": translated_name,
                        "original_species": english_name,
                        "confidence": float(conf_val),
                        "bbox": [float(x) for x in box_val]
                    }
                    tracks_data[track_id].append(entry)

            # === 第四步：保存 JSON 结果 ===
            target_json_dir = output_dir  # 默认输出到选择的目录
            if temp_video_dir: target_json_dir = temp_video_dir  # 如果指定了临时目录

            os.makedirs(target_json_dir, exist_ok=True)
            json_output_path = os.path.join(target_json_dir, f"{video_name}.json")

            final_json_data = {
                "video_source": video_source,
                "total_frames_processed": current_track_frame,
                "vid_stride": vid_stride,
                "tracker_config": tracker_config,
                "tracks": dict(tracks_data)
            }
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_data, f, ensure_ascii=False, indent=4)

            logger.info(f"视频处理完成，JSON已保存至: {json_output_path}")

            return {"json_path": json_output_path, "frame_count": current_track_frame, "status": "success"}

        except Exception as e:
            logger.error(f"视频追踪失败: {e}")
            return {"error": str(e), "status": "failed"}

        finally:
            # === 第五步：清理临时文件 ===
            if os.path.exists(temp_enhanced_video_path):
                try:
                    os.remove(temp_enhanced_video_path)
                    logger.info(f"已删除临时增强视频: {temp_enhanced_video_path}")
                except Exception as e:
                    logger.warning(f"删除临时视频失败: {e}")

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

                            if hasattr(r, 'candidates_data') and i in r.candidates_data:
                                box_info["候选项"] = r.candidates_data[i]
                                # 如果有分类结果，将"物种"字段更新为分类置信度最高的那一个
                                if r.candidates_data[i]:
                                    box_info["物种"] = r.candidates_data[i][0]['name']
                                    box_info["置信度"] = r.candidates_data[i][0]['conf']

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


