"""
Run with:  python3 -m unittest discover -s tests -v
(also collectible by pytest if installed)

These tests exercise the deterministic SOP core with LLM polishing disabled
(config.LLM_POLISH_ENABLED can be left on -- the client will simply report
itself unavailable if no Ollama server is reachable, and the orchestrator
falls back to the template reply automatically). This means the whole suite
runs identically with or without Ollama installed.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import SessionState, Phase
from app import orchestrator as orch


class TestSampleScenario(unittest.TestCase):
    def test_single_message_verifies_and_resolves(self):
        s = SessionState()
        msg = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. "
               "I'm calling about my denied healthcare claim from January. "
               "DOB is 1985-03-15, SSN last four is 4472.")
        out = orch.handle_turn(s, msg)
        self.assertTrue(out["verified"])
        self.assertEqual(out["matched_party_id"], "P9")
        self.assertEqual(out["resolved_case_id"], "CL-2048")
        self.assertEqual(out["resolved_intent"], "denial_question")
        self.assertEqual(out["phase"], "PROCESS_CASE")
        # must not have asked again for something already given
        self.assertNotIn("verify your identity", out["reply"].lower())
        # must disclose the denial reason now that verified+resolved
        self.assertIn("pathology report", out["reply"].lower())

    def test_no_disclosure_before_verification(self):
        s = SessionState()
        out = orch.handle_turn(s, "My name is Margaret Chen and I'm calling about a denied claim.")
        self.assertFalse(out["verified"])
        self.assertNotIn("CL-", out["reply"])
        self.assertNotIn("pathology", out["reply"].lower())


class TestPartialAndRefusal(unittest.TestCase):
    def test_partial_fields_then_completion_across_turns(self):
        s = SessionState()
        out = orch.handle_turn(s, "My name is Margaret Chen.")
        self.assertFalse(out["verified"])
        out = orch.handle_turn(s, "DOB is 1985-03-15")
        self.assertFalse(out["verified"])
        out = orch.handle_turn(s, "SSN last four is 4472")
        self.assertTrue(out["verified"])
        self.assertEqual(out["matched_party_id"], "P9")

    def test_refusal_gets_empathy_but_no_bypass(self):
        s = SessionState()
        orch.handle_turn(s, "My name is Margaret Chen.")
        out = orch.handle_turn(
            s, "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")
        self.assertFalse(out["verified"])
        self.assertNotIn("CL-", out["reply"])
        self.assertIn("frustrat", out["reply"].lower() + out["emotion"])

    def test_unknown_caller_never_confirms_or_denies_existence(self):
        s = SessionState()
        out = orch.handle_turn(s, "My name is Nobody Fake, DOB 2000-01-01, SSN last four 0000.")
        self.assertFalse(out["verified"])
        self.assertIsNone(out["matched_party_id"])


class TestOutOfScope(unittest.TestCase):
    def test_off_topic_question_declined_politely(self):
        s = SessionState()
        out = orch.handle_turn(s, "What is reinforcement learning?")
        self.assertIn("only help with", out["reply"].lower())

    def test_repeated_off_topic_offers_human(self):
        s = SessionState()
        orch.handle_turn(s, "What is reinforcement learning?")
        out = orch.handle_turn(s, "Tell me a joke instead.")
        self.assertIn("human representative", out["reply"].lower())


class TestRepresentativeConsent(unittest.TestCase):
    def test_registered_representative_default_scenario_approves(self):
        s = SessionState()
        orch.handle_turn(
            s, "Hi, I'm calling on behalf of my mother Margaret Chen about her claim.",
            consent_scenario="default")
        orch.handle_turn(s, "Sure, I will wait.")
        out = orch.handle_turn(s, "any update?")
        self.assertTrue(out["verified"])
        self.assertTrue(out["representative_mode"])
        self.assertEqual(out["matched_party_id"], "P9")

    def test_registered_representative_timeout_scenario_escalates(self):
        s = SessionState()
        orch.handle_turn(s, "I'm David Chen, calling on behalf of my mother.", consent_scenario="timeout")
        out = None
        for _ in range(5):
            out = orch.handle_turn(s, "any update?")
            if out["escalate_to_human"]:
                break
        self.assertTrue(out["escalate_to_human"])
        self.assertFalse(out["verified"])

    def test_unregistered_representative_declined_without_confirming_account(self):
        s = SessionState()
        out = orch.handle_turn(s, "I'm calling on behalf of my friend John Smith about his policy.")
        self.assertFalse(out["verified"])
        self.assertNotIn("no account", out["reply"].lower())
        self.assertIn("consent", out["reply"].lower())


class TestProcessCaseAndPostProcess(unittest.TestCase):
    def _verified_session(self):
        s = SessionState()
        orch.handle_turn(
            s, ("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472, "
                "calling about my denied healthcare claim from January."))
        return s

    def test_followup_document_question_is_grounded(self):
        s = self._verified_session()
        out = orch.handle_turn(s, "How do I submit the pathology report?")
        self.assertIn("portal", out["reply"].lower())
        self.assertIn("cl-2048", out["reply"].lower())

    def test_end_of_case_offers_email_and_respects_choice(self):
        s = self._verified_session()
        out = orch.handle_turn(s, "That's all, thanks.")
        self.assertEqual(out["phase"], "POST_PROCESS")
        self.assertIn("email", out["reply"].lower())
        out = orch.handle_turn(s, "No thanks.")
        self.assertEqual(out["email_decision"], "skipped")
        self.assertEqual(out["phase"], "ENDED")

    def test_email_yes_sends_and_ends_call(self):
        s = self._verified_session()
        orch.handle_turn(s, "That's all, thanks.")
        out = orch.handle_turn(s, "Yes please")
        self.assertEqual(out["email_decision"], "sent")
        self.assertEqual(out["phase"], "ENDED")
        self.assertIn("margaret@email.com", out["reply"])


class TestAmbiguousIntentResolution(unittest.TestCase):
    def test_multiple_candidates_prompts_disambiguation(self):
        s = SessionState()
        # Only a case-type hint (healthcare) still matches 2 claims (CL-2048, CL-2011)
        out = orch.handle_turn(
            s, "My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472, about a healthcare claim.")
        self.assertTrue(out["verified"])
        self.assertIn("CL-2048", out["reply"])
        self.assertIn("CL-2011", out["reply"])
        out = orch.handle_turn(s, "CL-2048 please")
        self.assertEqual(out["resolved_case_id"], "CL-2048")


if __name__ == "__main__":
    unittest.main()
