import json
import unittest
from unittest.mock import patch

from system.backend import maintenance


class PackageSourceResolutionTests(unittest.TestCase):
    def test_auto_uses_aliyun_for_mainland_china(self) -> None:
        with patch.object(maintenance, "_public_ip_country_code", return_value="CN"):
            source = maintenance.resolve_package_source("auto")

        self.assertEqual(source, ("aliyun", *maintenance.PIP_SOURCES["aliyun"]))

    def test_auto_uses_official_source_outside_mainland_china(self) -> None:
        with patch.object(maintenance, "_public_ip_country_code", return_value="US"):
            source = maintenance.resolve_package_source("auto")

        self.assertEqual(source, ("official", *maintenance.PIP_SOURCES["official"]))

    def test_auto_falls_back_to_official_source_when_detection_fails(self) -> None:
        with patch.object(maintenance, "_public_ip_country_code", return_value=None):
            source = maintenance.resolve_package_source("auto")

        self.assertEqual(source, ("official", *maintenance.PIP_SOURCES["official"]))

    def test_manual_source_does_not_detect_location(self) -> None:
        with patch.object(maintenance, "_public_ip_country_code") as detect_country:
            source = maintenance.resolve_package_source("tsinghua")

        detect_country.assert_not_called()
        self.assertEqual(source, ("tsinghua", *maintenance.PIP_SOURCES["tsinghua"]))


class PublicIpCountryTests(unittest.TestCase):
    def test_country_code_is_normalized(self) -> None:
        response = _FakeResponse({"ip": "203.0.113.1", "country": " cn "})
        with patch.object(maintenance.urllib.request, "urlopen", return_value=response):
            country = maintenance._public_ip_country_code()

        self.assertEqual(country, "CN")

    def test_invalid_response_is_treated_as_detection_failure(self) -> None:
        response = _FakeResponse({"country": "China"})
        with patch.object(maintenance.urllib.request, "urlopen", return_value=response):
            country = maintenance._public_ip_country_code()

        self.assertIsNone(country)


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return self._payload


if __name__ == "__main__":
    unittest.main()
