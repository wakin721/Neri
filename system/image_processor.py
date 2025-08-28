# system/image_processor.py

import os
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List
from collections import Counter
from ultralytics import YOLO
import json
from system.utils import resource_path

logger = logging.getLogger(__name__)

class ImageProcessor:
    """处理图像、检测物种的核心类"""

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

    def detect_species(self, img_path: str, use_fp16: bool = False, iou: float = 0.3,
                       conf: float = 0.25, augment: bool = True,
                       agnostic_nms: bool = True, timeout: float = 10.0) -> Dict[str, Any]:
        """检测图像中的物种并应用翻译"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                use_fp16 = False
        except ImportError:
            use_fp16 = False
        except Exception:
            use_fp16 = False

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
                    # 如果没有检测到任何物体，则跳过
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

                    # --- 翻译和合并逻辑 ---
                    detected_species_counts = {}
                    for element, count in counts.items():
                        # 获取检测到的原始英文名
                        english_name = species_dict.get(int(element), "unknown")
                        # 从翻译字典中查找中文名，如果找不到则使用原始英文名
                        translated_name = self.translation_dict.get(english_name, english_name)

                        # 按翻译后的中文名累加数量
                        if translated_name in detected_species_counts:
                            detected_species_counts[translated_name] += count
                        else:
                            detected_species_counts[translated_name] = count

                    # 将最终结果格式化为逗号分隔的字符串
                    species_list = list(detected_species_counts.keys())
                    counts_list = list(map(str, detected_species_counts.values()))

                    species_names = ",".join(species_list)
                    species_counts = ",".join(counts_list)
                    # --- 翻译逻辑结束 ---

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

    # V V V V V V V V V V V V V V V V V V V V
    # MODIFICATION: Accept dynamic temp_photo_dir
    # V V V V V V V V V V V V V V V V V V V V
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
                result_img.save(result_file, "JPEG", quality=95) # Directly save the image
                return result_file
        except Exception as e:
            logger.error(f"保存临时检测结果图片失败: {e}")
            return ""

    def save_detection_info_json(self, results, image_name: str, species_info: dict, temp_photo_dir: str) -> str:
        """保存探测结果信息到指定的临时目录"""
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