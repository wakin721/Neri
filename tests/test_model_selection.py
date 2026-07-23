import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from system.backend import services
from system.backend.models import DetectionItem, ProcessingOptions


class ProcessingOptionsModelSelectionTests(unittest.TestCase):
    def test_detection_requires_at_least_one_model(self) -> None:
        with self.assertRaisesRegex(ValidationError, "至少需要选择一个"):
            ProcessingOptions(enable_detection=True)

    def test_detection_accepts_any_selected_model(self) -> None:
        selections = [
            ("res/model/detector.pt", None),
            (None, "res/model_cls/classifier.pt"),
            ("res/model/detector.pt", "res/model_cls/classifier.pt"),
        ]
        for model_path, classification_model_path in selections:
            with self.subTest(
                model_path=model_path,
                classification_model_path=classification_model_path,
            ):
                options = ProcessingOptions(
                    enable_detection=True,
                    model_path=model_path,
                    classification_model_path=classification_model_path,
                )

                self.assertEqual(options.model_path, model_path)
                self.assertEqual(
                    options.classification_model_path,
                    classification_model_path,
                )

    def test_blank_model_paths_are_treated_as_unselected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "至少需要选择一个"):
            ProcessingOptions(
                enable_detection=True,
                model_path=" ",
                classification_model_path="",
            )

    def test_metadata_scan_does_not_require_a_model(self) -> None:
        options = ProcessingOptions(enable_detection=False)

        self.assertIsNone(options.model_path)
        self.assertIsNone(options.classification_model_path)


class ClassificationOnlyResultTests(unittest.TestCase):
    def test_classifier_names_are_used_without_a_detection_model(self) -> None:
        detector = SimpleNamespace(
            model=None,
            cls_model=SimpleNamespace(names={0: "empty", 1: "bird"}),
            translation_dict={"bird": "鸟"},
        )

        self.assertEqual(
            services._detector_model_names(detector),
            {0: "empty", 1: "bird"},
        )
        self.assertEqual(
            services._selected_species_class_ids(detector, ["鸟"]),
            [1],
        )

    def test_full_image_classification_is_serialized_without_a_box(self) -> None:
        detector = SimpleNamespace(translation_dict={})
        candidates = [
            {
                "name": "鸟",
                "conf": 0.91,
                "raw_cls_conf": 0.91,
                "class_id": 1,
            }
        ]

        detection_data = services._serialize_detector_output(
            detector,
            {
                "物种名称": "鸟",
                "物种数量": "1",
                "最低置信度": "0.910",
                "detect_results": None,
                "分类候选项": candidates,
            },
        )
        item = services._apply_detection_data(
            DetectionItem(filename="bird.jpg", path="bird.jpg", file_type="jpg"),
            detection_data,
        )

        self.assertEqual(detection_data["分类候选项"], candidates)
        self.assertEqual(item.species, ["鸟"])
        self.assertEqual(item.confidence, 0.91)
        self.assertEqual(item.detection_boxes, [])


if __name__ == "__main__":
    unittest.main()
