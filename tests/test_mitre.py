import unittest

from atdrps.data.mitre import (
    LABEL_PATTERNS, MITRE_TACTICS, OUT_OF_SCOPE, RULES, describe_stage,
    explain_rules, infiltration_flag, score_stages, stage_from_label,
)
from atdrps.data.schema import BENIGN_STAGE, STAGES


class TestLabelMapping(unittest.TestCase):
    def test_benign_spellings(self):
        for label in ("Benign", "BENIGN", "benign", "normal", "-", "", "  "):
            with self.subTest(label=label):
                self.assertEqual(stage_from_label(label), BENIGN_STAGE)

    def test_cicids2018_families(self):
        cases = {
            "SSH-Bruteforce": "InitialAccess",
            "FTP-BruteForce": "InitialAccess",
            "Brute Force -Web": "InitialAccess",
            "SQL Injection": "InitialAccess",
            "Infilteration": "InitialAccess",       # CIC's own spelling
            "Bot": "CommandAndControl",
            "PortScan": "Reconnaissance",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(stage_from_label(label), expected)

    def test_denial_of_service_maps_to_impact_not_to_the_chain(self):
        """DoS is Impact (TA0040), not a step toward infiltration. Calling it
        Reconnaissance -- both send a lot of SYNs -- would corrupt the
        transition dynamics the model exists to learn."""
        chain = {"Reconnaissance", "InitialAccess", "LateralMovement",
                 "CommandAndControl", "Exfiltration"}
        for label in ("DoS attacks-Hulk", "DDoS attacks-LOIC-HTTP", "DDOS",
                      "DoS Slowloris", "DoS GoldenEye", "SYN flood"):
            with self.subTest(label=label):
                stage = stage_from_label(label)
                self.assertEqual(stage, "Impact")
                self.assertNotIn(stage, chain)

    def test_new_attack_families_from_the_taxonomy(self):
        """Every family in the team's attack document resolves somewhere
        deliberate -- see docs/ATTACK_COVERAGE.md."""
        cases = {
            "WannaCry ransomware": "Impact",
            "Conficker worm": "LateralMovement",
            "EternalBlue SMB exploit": "LateralMovement",
            "DNS tunnel (iodine)": "Exfiltration",
            "dnscat2": "Exfiltration",
            "credential stuffing": "InitialAccess",
            "password spraying": "InitialAccess",
            "ARP spoofing": "Reconnaissance",
            "DNS cache poisoning": "Reconnaissance",
            "keylogger": "CommandAndControl",
            "remote access trojan": "CommandAndControl",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(stage_from_label(label), expected)

    def test_unknown_label_is_out_of_scope_not_a_guess(self):
        self.assertEqual(stage_from_label("some-new-2027-attack"), OUT_OF_SCOPE)

    def test_every_pattern_targets_a_real_stage(self):
        valid = set(STAGES) | {OUT_OF_SCOPE}
        for _pattern, stage in LABEL_PATTERNS:
            self.assertIn(stage, valid)

    def test_tactics_cover_every_stage(self):
        for stage in STAGES:
            self.assertIn(stage, MITRE_TACTICS)

    def test_descriptions_cite_attack(self):
        self.assertIn("TA0010", describe_stage("Exfiltration"))
        self.assertIn("T1041", describe_stage("Exfiltration"))
        self.assertIn("TA0040", describe_stage("Impact"))
        self.assertIn("T1486", describe_stage("Impact"))

    def test_out_of_scope_description_does_not_claim_a_tactic(self):
        text = describe_stage(OUT_OF_SCOPE)
        self.assertIn("Out of scope", text)
        self.assertNotIn("TA00", text)


class TestRuleEngine(unittest.TestCase):
    def test_rules_reference_only_documented_stages(self):
        for stage, *_ in RULES:
            self.assertIn(stage, STAGES)

    def test_scan_window_scores_reconnaissance_highest(self):
        window = {
            "distinct_dst_ports": 40, "mean_payload_zero_ratio": 0.95,
            "mean_flow_packets": 2, "rst_share": 0.5, "max_ports_per_src_dst": 38,
            "outbound_bytes_ratio": 0.1, "total_bytes": 5_000,
            "beacon_regularity": 0.1, "internal_flow_share": 0.0,
            "admin_service_share": 0.0, "external_flow_share": 0.0,
            "mean_flow_bytes": 100, "repeat_external_dst_count": 0,
            "max_flows_per_src_dst_port": 1, "auth_service_share": 0.0,
            "distinct_internal_dsts": 0, "mean_retransmission_ratio": 0.0,
            "mean_payload_mean": 0.0,
        }
        scores = score_stages(window)
        self.assertEqual(max(scores, key=scores.get), "Reconnaissance")
        self.assertGreater(scores["Reconnaissance"], 0.8)

    def test_exfiltration_window_scores_exfiltration_highest(self):
        window = {
            "distinct_dst_ports": 2, "mean_payload_zero_ratio": 0.05,
            "mean_flow_packets": 900, "rst_share": 0.0, "max_ports_per_src_dst": 1,
            "outbound_bytes_ratio": 0.97, "total_bytes": 9_000_000,
            "beacon_regularity": 0.3, "internal_flow_share": 0.1,
            "admin_service_share": 0.0, "external_flow_share": 0.8,
            "mean_flow_bytes": 900_000, "repeat_external_dst_count": 1,
            "max_flows_per_src_dst_port": 2, "auth_service_share": 0.0,
            "distinct_internal_dsts": 0, "mean_retransmission_ratio": 0.0,
            "mean_payload_mean": 1400,
        }
        scores = score_stages(window)
        self.assertEqual(max(scores, key=scores.get), "Exfiltration")

    def test_missing_features_do_not_fire_rules(self):
        self.assertEqual(max(score_stages({}).values()), 0.0)

    def test_scores_are_bounded(self):
        window = {name: 1e9 for _s, name, *_ in RULES}
        for value in score_stages(window).values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_explanations_name_the_feature_and_threshold(self):
        window = {"distinct_dst_ports": 40, "mean_payload_zero_ratio": 0.95}
        reasons = explain_rules(window, "Reconnaissance")
        self.assertTrue(any("distinct_dst_ports" in r for r in reasons))
        self.assertTrue(any("40" in r for r in reasons))

    def test_infiltration_flag(self):
        self.assertEqual(infiltration_flag("Exfiltration"), 1)
        self.assertEqual(infiltration_flag("LateralMovement"), 1)
        self.assertEqual(infiltration_flag("Reconnaissance"), 0)
        self.assertEqual(infiltration_flag(BENIGN_STAGE), 0)


if __name__ == "__main__":
    unittest.main()
