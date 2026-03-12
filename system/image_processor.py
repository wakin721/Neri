# system/image_processor.py

import os
import logging
import concurrent.futures
import gc
from typing import Dict, Any, Optional, List, Union, Tuple
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

    def _determine_device(self, use_fp16: bool) -> tuple[str, bool]:
        """检查 CUDA 或 XPU 可用性，并返回设备名称及是否使用 FP16"""
        device = 'cpu'
        fp16_enabled = False

        try:
            # 1. 首先检查 NVIDIA CUDA
            if torch.cuda.is_available():
                device = 'cuda'
                fp16_enabled = use_fp16
            else:
                # 2. 检查 Intel XPU
                try:
                    import intel_extension_for_pytorch as ipex
                    if hasattr(torch, 'xpu') and torch.xpu.is_available():
                        device = 'xpu'
                        fp16_enabled = use_fp16
                except ImportError:
                    logger.debug("未安装 intel_extension_for_pytorch，跳过 Intel GPU 检测")
        except Exception as e:
            logger.error(f"设备检测失败: {e}")

        return device, fp16_enabled

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

    def _process_single_image_task(self, args):
        """辅助方法：处理单张图片的线程任务"""
        idx, path = args
        try:
            # 这里的 self._preprocess_image 需要确保能被访问
            img = cv2.imread(path)
            if img is None:
                return None
            # 预处理 (LAB增强等)
            proc_img = self._preprocess_image(img)
            # 转换副本用于后续裁剪 (RGB)
            orig_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            return (idx, proc_img, orig_rgb)
        except Exception as e:
            logger.warning(f"处理图片 {path} 失败: {e}")
            return None

    def preload_batch_data(self, img_paths: List[str]) -> Optional[Tuple]:
        """
        预加载一批图片数据，返回 (valid_indices, processed_imgs, original_imgs_rgb)
        """
        try:
            processed_imgs = []
            valid_indices = []
            original_imgs_rgb = []

            max_workers = min(len(img_paths), 8)

            # 使用线程池并行读取和预处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._process_single_image_task, (idx, path))
                    for idx, path in enumerate(img_paths)
                ]

                for future in futures:
                    result = future.result()
                    if result is not None:
                        idx, proc_img, orig_rgb = result
                        valid_indices.append(idx)
                        processed_imgs.append(proc_img)
                        original_imgs_rgb.append(orig_rgb)

            if not processed_imgs:
                return None

            return (valid_indices, processed_imgs, original_imgs_rgb)
        except Exception as e:
            logger.error(f"预加载数据失败: {e}")
            return None

    def _crop_single_box(self, args):
        """辅助方法：处理单个检测框的裁剪与Padding任务"""
        r_idx, b_idx, box, orig_img_rgb = args
        try:
            h, w, _ = orig_img_rgb.shape
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 扩展逻辑
            expand_ratio = 0.1
            box_width = x2 - x1
            box_height = y2 - y1
            pad_w = int(box_width * expand_ratio)
            pad_h = int(box_height * expand_ratio)

            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            if x2 > x1 and y2 > y1:
                crop = orig_img_rgb[y1:y2, x1:x2]

                # Padding Square
                ch, cw = crop.shape[:2]
                if ch != cw:
                    max_dim = max(ch, cw)
                    top = (max_dim - ch) // 2
                    bottom = max_dim - ch - top
                    left = (max_dim - cw) // 2
                    right = max_dim - cw - left
                    crop = cv2.copyMakeBorder(
                        crop, top, bottom, left, right,
                        cv2.BORDER_CONSTANT, value=[114, 114, 114]
                    )
                return (r_idx, b_idx, crop)
        except Exception as e:
            pass
        return None

    def detect_batch_species(self, img_paths: List[str], use_fp16: bool = False, iou: float = 0.3,
                             conf: float = 0.25, augment: bool = True,
                             agnostic_nms: bool = True, timeout: float = 60.0,
                             preloaded_data: Optional[Tuple] = None,
                             classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        批量检测图像中的物种
        :param preloaded_data: (可选) 由 preload_batch_data 返回的预处理数据 (valid_indices, processed_imgs, original_imgs_rgb)
        """
        device_name, use_fp16 = self._determine_device(use_fp16)
        w_det = 0.4
        w_cls = 0.6
        batch_results_info = []

        if not self.model:
            for _ in img_paths:
                batch_results_info.append({
                    '物种名称': "", '物种数量': "",
                    'detect_results': None, '最低置信度': None
                })
            return batch_results_info

        def run_batch_process():
            nonlocal batch_results_info
            try:
                # 1. 优先使用预加载的数据，否则现场处理
                if preloaded_data:
                    valid_indices, processed_imgs, original_imgs_rgb = preloaded_data
                else:
                    # 如果没有预加载数据，则执行原有的加载逻辑 (调用新提取的 _process_single_image_task)
                    processed_imgs = []
                    valid_indices = []
                    original_imgs_rgb = []
                    max_workers = min(len(img_paths), 8)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(self._process_single_image_task, (idx, path))
                            for idx, path in enumerate(img_paths)
                        ]
                        for future in futures:
                            result = future.result()
                            if result is not None:
                                idx, proc_img, orig_rgb = result
                                valid_indices.append(idx)
                                processed_imgs.append(proc_img)
                                original_imgs_rgb.append(orig_rgb)

                if not processed_imgs:
                    return

                import tempfile
                temp_run_project = os.path.join(tempfile.gettempdir(), "yolo_logs")

                # 2. 批量运行检测模型
                det_results = self.model(
                    processed_imgs,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    imgsz=1920,
                    half=use_fp16,
                    device=device_name,
                    iou=iou,
                    conf=conf,
                    project=temp_run_project,
                    name="detect_log",
                    save=False
                )
                # 3. 准备分类裁剪 (Collection Phase)
                all_crops = []
                crop_map_info = []  # 映射: list index -> (result_index_in_batch, box_index)
                batch_candidates_maps = [{} for _ in det_results]

                if self.cls_model:
                    crop_tasks = []
                    # 收集所有需要裁剪的任务
                    for r_idx, r in enumerate(det_results):
                        if r.boxes is None: continue
                        orig_img_rgb = original_imgs_rgb[r_idx]
                        
                        for b_idx, box in enumerate(r.boxes):
                            crop_tasks.append((r_idx, b_idx, box, orig_img_rgb))

                    # 并行执行裁剪和Padding
                    if crop_tasks:
                        # 这里的 workers 数量不需要太多，主要是为了解耦内存操作
                        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                            results = executor.map(self._crop_single_box, crop_tasks)
                            
                            for res in results:
                                if res is not None:
                                    r_idx, b_idx, crop = res
                                    all_crops.append(crop)
                                    crop_map_info.append((r_idx, b_idx))

                    # 4. 批量运行分类模型 (Batch Inference)
                    if all_crops:
                        cls_results_list = self.cls_model(
                            all_crops,
                            half=use_fp16,
                            device=device_name,
                            save=False,
                            project=temp_run_project,
                            name="cls_log",
                            exist_ok=True
                        )

                        # 5. 映射回原结果 (Map Back)
                        for i, cls_res in enumerate(cls_results_list):
                            r_idx, b_idx = crop_map_info[i]

                            # 获取原始检测置信度
                            det_conf = float(det_results[r_idx].boxes[b_idx].conf.item())

                            # 温度缩放 & TopK
                            original_probs = cls_res.probs.data
                            smoothed_probs = self._apply_temperature_scaling(original_probs, temperature=3.0)
                            topk_confs, topk_indices = torch.topk(smoothed_probs, 3)

                            candidates = []
                            for c_idx, c_conf in zip(topk_indices.tolist(), topk_confs.tolist()):
                                raw_name = cls_res.names[int(c_idx)]
                                trans_name = self.translation_dict.get(raw_name, raw_name)
                                cls_conf_val = float(c_conf)

                                # 加权置信度
                                weighted_conf = (det_conf * w_det) + (cls_conf_val * w_cls)

                                candidates.append({
                                    "name": trans_name,
                                    "conf": weighted_conf,
                                    "raw_cls_conf": cls_conf_val,
                                    "raw_det_conf": det_conf
                                })

                            candidates.sort(key=lambda x: x["conf"], reverse=True)
                            batch_candidates_maps[r_idx][b_idx] = candidates

                # 6. 结果整合与统计
                # 此时 det_results 的长度等于 processed_imgs 的长度
                # 我们需要将其映射回原始 img_paths 的长度（处理读取失败的情况）

                det_iter = iter(det_results)
                cand_map_iter = iter(batch_candidates_maps)

                for idx in range(len(img_paths)):
                    if idx not in valid_indices:
                        # 读取失败的图片返回空
                        batch_results_info.append({
                            '物种名称': "", '物种数量': "",
                            'detect_results': None, '最低置信度': None
                        })
                        continue

                    r = next(det_iter)
                    candidates_map = next(cand_map_iter)

                    min_conf = None
                    detected_species_counts = {}

                    if r.boxes:
                        confs = r.boxes.conf.tolist()
                        if confs:
                            current_min = min(confs)
                            min_conf = "%.3f" % current_min

                        for i, box in enumerate(r.boxes):
                            final_name = ""
                            # 优先使用分类修正结果
                            if i in candidates_map and candidates_map[i]:
                                final_name = candidates_map[i][0]['name']
                            else:
                                cls_id = int(box.cls.item())
                                raw_name = r.names[cls_id]
                                final_name = self.translation_dict.get(raw_name, raw_name)

                            detected_species_counts[final_name] = detected_species_counts.get(final_name, 0) + 1

                            # 注入数据用于JSON保存
                            if not hasattr(r, 'candidates_data'):
                                r.candidates_data = {}
                            if i in candidates_map:
                                r.candidates_data[i] = candidates_map[i]

                    species_str = ",".join(list(detected_species_counts.keys()))
                    counts_str = ",".join(list(map(str, detected_species_counts.values())))

                    batch_results_info.append({
                        '物种名称': species_str if species_str else "空",
                        '物种数量': counts_str if counts_str else "空",
                        'detect_results': [r],  # 保持列表格式以便兼容 save_detection_info_json
                        '最低置信度': min_conf
                    })

                return True

            except Exception as e:
                logger.error(f"批量检测失败: {e}")
                return False

            finally:
                pass

        run_batch_process()

        try:
            # 1. 强制进行 Python 垃圾回收，断开未使用的 Tensor 引用
            gc.collect()

            # 2. 如果使用 GPU，强制清空 PyTorch 的显存缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 可选：整理碎片（PyTorch 某些版本支持）
                # torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"显存清理过程中发生错误 (不影响结果): {e}")

        return batch_results_info

    def _create_temp_enhanced_video(self, source_path: str, temp_path: str, stride: int) -> int:
        """
        [修改] 读取源视频，应用跳帧和LAB增强，保存为临时MP4文件。
        使用 Batch + ThreadPool 优化处理速度。
        """
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise Exception(f"无法打开源视频: {source_path}")

        # 获取原始信息
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0: orig_fps = 25

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_path, fourcc, orig_fps, (orig_w, orig_h))

        if not writer.isOpened():
            cap.release()
            raise Exception("无法创建临时视频写入器")

        idx = 0
        saved_count = 0
        batch_size = 32  # 批处理大小，控制内存占用
        
        try:
            while True:
                frames_batch = []
                # 1. 预读取一批符合 stride 的帧
                for _ in range(batch_size):
                    # 循环读取直到找到符合 stride 的帧或视频结束
                    while True:
                        success, frame = cap.read()
                        if not success:
                            break # 视频结束
                        
                        current_idx = idx
                        idx += 1
                        
                        if current_idx % stride == 0:
                            frames_batch.append(frame)
                            break # 找到一帧，加入批次
                    
                    if not success: # 如果视频读取完毕，跳出批次循环
                        break

                if not frames_batch:
                    break

                # 2. 并行预处理 (LAB增强)
                processed_batch = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(frames_batch), 8)) as executor:
                    # 使用 map 保证结果顺序与 frames_batch 一致
                    results = executor.map(self._preprocess_image, frames_batch)
                    processed_batch = list(results)

                # 3. 顺序写入视频
                for p_frame in processed_batch:
                    if p_frame is not None:
                        writer.write(p_frame)
                        saved_count += 1

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
                             temp_video_dir: Optional[str] = None,
                             classes: Optional[List[int]] = None) -> Dict[str, Any]:
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
        device_name, use_fp16 = self._determine_device(use_fp16)

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
            results = self.model.track(
                source=temp_enhanced_video_path,
                tracker=tracker_config,
                augment=augment,
                agnostic_nms=agnostic_nms,
                imgsz=1920,
                half=use_fp16,
                device=device_name,
                iou=iou,
                conf=conf,
                persist=True,
                save=False,
                project=temp_run_project,
                name="track_log",
                exist_ok=True,
                stream=True,
                vid_stride=1
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

            # 视频处理完成后，同步将结果更新到 SQLite 数据库中
            try:
                from system.detection_db import get_db_path, init_db, upsert_detection
                db_path = get_db_path(target_json_dir)
                if not os.path.exists(db_path):
                    init_db(db_path)

                # 获取带有后缀的完整视频文件名
                full_video_filename = os.path.basename(video_source)

                # 写入数据库 (base_name, 完整文件名, 包含检测结果的字典)
                upsert_detection(db_path, video_name, full_video_filename, final_json_data)
            except Exception as db_err:
                logger.warning(f"同步视频检测结果到 SQLite 失败（不影响正常流程）: {db_err}")

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

    def save_detection_info_json(self, results, image_name: str,
                                 species_info: dict, temp_photo_dir: str) -> str:
        """保存探测结果信息到指定的临时目录，并同步写入 SQLite"""
        if not results or not temp_photo_dir:
            return ""

        try:
            import json
            # ── 原有逻辑（保持不变，保留 JSON 文件作为兼容备份）──
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

            # ── 新增：同步写入 SQLite ──────────────────────────────────
            try:
                from system.detection_db import get_db_path, init_db, upsert_detection
                db_path = get_db_path(temp_photo_dir)
                if not os.path.exists(db_path):
                    init_db(db_path)
                upsert_detection(db_path, base_name, image_name, data_to_save)
            except Exception as db_err:
                logger.warning(f"写入 SQLite 失败（不影响正常流程）: {db_err}")
            # ─────────────────────────────────────────────────────────────

            return json_path
        except Exception as e:
            logger.error(f"保存检测结果JSON失败: {e}")
            return ""
