import unittest

from atdrps.config import Config, set_global_seed


class TestConfig(unittest.TestCase):
    def test_defaults_load(self):
        cfg = Config.load()
        self.assertEqual(cfg.window.size_s, 30.0)
        self.assertEqual(cfg.model.context, 16)
        self.assertIn("Exfiltration", cfg.mitre.stages)

    def test_attribute_and_mapping_access_agree(self):
        cfg = Config.load()
        self.assertEqual(cfg.train.loss_weights.stage, cfg["train"]["loss_weights"]["stage"])

    def test_overrides_are_typed(self):
        cfg = Config.load(
            overrides=[
                "model.backend=numpy",
                "window.size_s=12.5",
                "model.context=4",
                "ingest.close_on_teardown=false",
                "ingest.max_packets=null",
            ]
        )
        self.assertEqual(cfg.model.backend, "numpy")
        self.assertIsInstance(cfg.window.size_s, float)
        self.assertEqual(cfg.window.size_s, 12.5)
        self.assertIsInstance(cfg.model.context, int)
        self.assertIs(cfg.ingest.close_on_teardown, False)
        self.assertIsNone(cfg.ingest.max_packets)

    def test_nested_override_creates_path(self):
        cfg = Config.load(overrides=["brand.new.key=7"])
        self.assertEqual(cfg.get_path("brand.new.key"), 7)

    def test_bad_override_raises(self):
        with self.assertRaises(ValueError):
            Config.load(overrides=["no_equals_sign"])

    def test_unknown_attribute_raises(self):
        cfg = Config.load()
        with self.assertRaises(AttributeError):
            _ = cfg.definitely_not_a_key

    def test_roundtrip_dump(self):
        import tempfile, os, yaml

        cfg = Config.load(overrides=["seed=99"])
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cfg.yaml")
            cfg.dump(p)
            with open(p, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        self.assertEqual(data["seed"], 99)
        self.assertNotIn("Config", str(type(data["window"])))

    def test_seed_is_deterministic(self):
        import numpy as np

        set_global_seed(7)
        a = np.random.rand(5)
        set_global_seed(7)
        b = np.random.rand(5)
        self.assertTrue(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
