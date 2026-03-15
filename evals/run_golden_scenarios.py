"""Run all golden scenarios through ExplanationAgent → EvalAgent pipeline.

Usage:
    python -m evals.run_golden_scenarios [--output evals/EVAL_RESULTS.md]

Requires ANTHROPIC_API_KEY (or appropriate LLM key) in environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.agents.eval_agent import EvalAgent
from src.agents.explanation_agent import ExplanationAgent
from src.schemas.detection import FraudDetectionResult, SHAPFeature

SCENARIOS_PATH = Path("tests/golden/scenarios.json")
DEFAULT_OUTPUT = Path("evals/EVAL_RESULTS.md")

# Pre-built detection results for scenarios that need specific inputs
_SCENARIO_DETECTIONS: dict[str, FraudDetectionResult] = {
    "GS-001": FraudDetectionResult(
        transaction_id="TX_GS001",
        fraud_probability=0.87,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.45, feature_value=1299.99
            ),
            SHAPFeature(
                feature_name="DeviceInfo",
                shap_value=0.38,
                feature_value="iPhone 15 (new)",
            ),
            SHAPFeature(
                feature_name="P_emaildomain",
                shap_value=0.22,
                feature_value="protonmail.com",
            ),
            SHAPFeature(feature_name="card6", shap_value=0.15, feature_value="credit"),
            SHAPFeature(feature_name="addr1", shap_value=-0.08, feature_value=100),
        ],
        model_version="1.0.0",
        inference_latency_ms=4.1,
        confidence_tier="high",
    ),
    "GS-002": FraudDetectionResult(
        transaction_id="TX_GS002",
        fraud_probability=0.12,
        is_fraud_predicted=False,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=-0.15, feature_value=24.50
            ),
            SHAPFeature(feature_name="card6", shap_value=-0.10, feature_value="debit"),
            SHAPFeature(feature_name="addr1", shap_value=-0.08, feature_value=315),
        ],
        model_version="1.0.0",
        inference_latency_ms=3.5,
        confidence_tier="high",
    ),
    "GS-003": FraudDetectionResult(
        transaction_id="TX_GS003",
        fraud_probability=0.52,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.08, feature_value=150.00
            ),
            SHAPFeature(feature_name="card4", shap_value=0.06, feature_value="visa"),
        ],
        model_version="1.0.0",
        inference_latency_ms=3.9,
        confidence_tier="low",
    ),
    "GS-006": FraudDetectionResult(
        transaction_id="TX_GS006",
        fraud_probability=0.78,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.40, feature_value=899.99
            ),
            SHAPFeature(
                feature_name="DeviceInfo", shap_value=0.30, feature_value="MacBook Pro"
            ),
            SHAPFeature(
                feature_name="P_emaildomain", shap_value=0.20, feature_value="yahoo.com"
            ),
            SHAPFeature(
                feature_name="card4", shap_value=0.12, feature_value="mastercard"
            ),
            SHAPFeature(feature_name="addr1", shap_value=-0.05, feature_value=200),
        ],
        model_version="1.0.0",
        inference_latency_ms=4.0,
        confidence_tier="high",
    ),
    "GS-007": FraudDetectionResult(
        transaction_id="TX_GS007",
        fraud_probability=0.82,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.42, feature_value=599.00
            ),
            SHAPFeature(
                feature_name="DeviceInfo",
                shap_value=0.28,
                feature_value="Samsung Galaxy S24",
            ),
            SHAPFeature(
                feature_name="P_emaildomain",
                shap_value=0.18,
                feature_value="outlook.com",
            ),
            SHAPFeature(feature_name="card6", shap_value=0.10, feature_value="credit"),
        ],
        model_version="1.0.0",
        inference_latency_ms=3.7,
        confidence_tier="high",
    ),
    "GS-009": FraudDetectionResult(
        transaction_id="TX_GS009",
        fraud_probability=0.75,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.35, feature_value=450.00
            ),
            SHAPFeature(
                feature_name="DeviceInfo", shap_value=0.25, feature_value="Windows 10"
            ),
            SHAPFeature(
                feature_name="P_emaildomain", shap_value=0.15, feature_value="gmail.com"
            ),
        ],
        model_version="1.0.0",
        inference_latency_ms=4.3,
        confidence_tier="high",
    ),
    "GS-010": FraudDetectionResult(
        transaction_id="TX_GS010",
        fraud_probability=0.91,
        is_fraud_predicted=True,
        top_shap_features=[
            SHAPFeature(
                feature_name="TransactionAmt", shap_value=0.50, feature_value=2100.00
            ),
            SHAPFeature(
                feature_name="DeviceInfo",
                shap_value=0.35,
                feature_value="Unknown Device",
            ),
            SHAPFeature(
                feature_name="P_emaildomain",
                shap_value=0.25,
                feature_value="tempmail.org",
            ),
            SHAPFeature(feature_name="card4", shap_value=0.18, feature_value="visa"),
            SHAPFeature(feature_name="card6", shap_value=0.10, feature_value="credit"),
        ],
        model_version="1.0.0",
        inference_latency_ms=4.5,
        confidence_tier="high",
    ),
}


def _load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def _run_eval_scenario(
    scenario: dict,
    explain_agent: ExplanationAgent,
    eval_agent: EvalAgent,
) -> dict:
    """Run a single golden scenario through explain → eval pipeline.

    Returns a dict with scenario id, status, and result details.
    """
    sid = scenario["id"]
    result_entry: dict = {"id": sid, "description": scenario["description"]}

    # Skip scenarios not relevant to EvalAgent
    if sid in ("GS-004", "GS-005", "GS-008"):
        result_entry["status"] = "SKIPPED"
        result_entry["reason"] = (
            "Scenario tests injection/empty-SHAP/budget — not EvalAgent scope"
        )
        return result_entry

    detection = _SCENARIO_DETECTIONS.get(sid)
    if detection is None:
        result_entry["status"] = "SKIPPED"
        result_entry["reason"] = "No detection fixture defined"
        return result_entry

    # Determine target audience
    target_audience = scenario.get("target_audience", "analyst")

    try:
        # GS-009: inject a deliberately poor explanation
        if scenario.get("inject_poor_explanation"):
            from src.schemas.explanation import ExplanationResult

            poor_text = scenario["inject_poor_explanation"]
            explanation = ExplanationResult(
                transaction_id=detection.transaction_id,
                target_audience=target_audience,
                fraud_probability=detection.fraud_probability,
                is_fraud_predicted=detection.is_fraud_predicted,
                explanation_text=poor_text,
                cited_features=["TransactionAmt"],
                uncited_features=[
                    f.feature_name
                    for f in detection.top_shap_features
                    if f.feature_name != "TransactionAmt"
                ],
                hallucinated_features=[],
                uncertainty_flag=detection.confidence_tier == "low",
                uncertainty_disclosure=(
                    "Low confidence." if detection.confidence_tier == "low" else None
                ),
                explanation_generated=True,
                token_cost_usd=0.001,
                generation_latency_seconds=1.0,
            )
        else:
            explanation = explain_agent.explain(detection, target_audience)

        if not explanation.explanation_generated:
            result_entry["status"] = "DEGRADED"
            result_entry["reason"] = f"Explanation not generated: {explanation.warning}"
            return result_entry

        # Evaluate
        eval_result = eval_agent.evaluate(explanation, detection)

        result_entry["status"] = "PASS" if eval_result.passed else "FAIL"
        result_entry["overall_score"] = eval_result.overall_score
        result_entry["grounding_score"] = eval_result.grounding_score
        result_entry["clarity_score"] = eval_result.clarity_score
        result_entry["completeness_score"] = eval_result.completeness_score
        result_entry["audience_appropriateness_score"] = (
            eval_result.audience_appropriateness_score
        )
        result_entry["uncertainty_handling_score"] = (
            eval_result.uncertainty_handling_score
        )
        result_entry["passed"] = eval_result.passed
        result_entry["failure_reasons"] = eval_result.failure_reasons
        result_entry["eval_cost_usd"] = eval_result.token_cost_usd

        # Check expected outcomes from scenario
        if "expected_eval_passed" in scenario:
            expected = scenario["expected_eval_passed"]
            if eval_result.passed != expected:
                result_entry["expectation_mismatch"] = (
                    f"Expected passed={expected}, got passed={eval_result.passed}"
                )
        if "expected_overall_score_min" in scenario:
            min_score = scenario["expected_overall_score_min"]
            if eval_result.overall_score < min_score:
                result_entry["expectation_mismatch"] = (
                    f"Expected overall_score >= {min_score}, "
                    f"got {eval_result.overall_score}"
                )

    except Exception as e:
        result_entry["status"] = "ERROR"
        result_entry["error"] = f"{type(e).__name__}: {e}"

    return result_entry


def _generate_report(results: list[dict], elapsed: float) -> str:
    """Generate markdown report from scenario results."""
    lines = [
        "# Evaluation Results — Golden Scenarios",
        "",
        f"**Run date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Total elapsed**: {elapsed:.1f}s",
        "",
        "## Summary",
        "",
        "| Scenario | Description | Status | Overall Score |",
        "|----------|-------------|--------|---------------|",
    ]

    pass_count = 0
    fail_count = 0
    skip_count = 0
    error_count = 0

    for r in results:
        status = r["status"]
        score = r.get("overall_score", "—")
        if isinstance(score, float):
            score = f"{score:.2f}"

        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        elif status == "SKIPPED":
            skip_count += 1
        elif status in ("ERROR", "DEGRADED"):
            error_count += 1

        lines.append(f"| {r['id']} | {r['description'][:50]} | {status} | {score} |")

    lines.extend(
        [
            "",
            f"**Results**: {pass_count} passed, {fail_count} failed, "
            f"{skip_count} skipped, {error_count} errors",
            "",
        ]
    )

    # Detail sections
    lines.append("## Detailed Results")
    lines.append("")
    for r in results:
        lines.append(f"### {r['id']}: {r['description']}")
        lines.append(f"- **Status**: {r['status']}")
        if r.get("reason"):
            lines.append(f"- **Reason**: {r['reason']}")
        if r.get("overall_score") is not None:
            lines.append(f"- **Overall score**: {r['overall_score']:.4f}")
            lines.append(f"- **Grounding**: {r.get('grounding_score', '—')}")
            lines.append(f"- **Clarity**: {r.get('clarity_score', '—')}")
            lines.append(f"- **Completeness**: {r.get('completeness_score', '—')}")
            lines.append(
                f"- **Audience**: {r.get('audience_appropriateness_score', '—')}"
            )
            unc = r.get("uncertainty_handling_score")
            lines.append(f"- **Uncertainty**: {unc if unc is not None else 'N/A'}")
            lines.append(f"- **Passed**: {r.get('passed')}")
            if r.get("failure_reasons"):
                lines.append(f"- **Failure reasons**: {r['failure_reasons']}")
            if r.get("eval_cost_usd"):
                lines.append(f"- **Eval cost**: ${r['eval_cost_usd']:.6f}")
        if r.get("expectation_mismatch"):
            lines.append(f"- **EXPECTATION MISMATCH**: {r['expectation_mismatch']}")
        if r.get("error"):
            lines.append(f"- **Error**: {r['error']}")
        lines.append("")

    return "\n".join(lines)


def main(output_path: Path = DEFAULT_OUTPUT) -> int:
    scenarios = _load_scenarios()
    print(f"Loaded {len(scenarios)} golden scenarios from {SCENARIOS_PATH}")

    explain_agent = ExplanationAgent()
    eval_agent = EvalAgent()

    results = []
    start = time.monotonic()

    for scenario in scenarios:
        sid = scenario["id"]
        print(f"  Running {sid}: {scenario['description'][:60]}...")
        entry = _run_eval_scenario(scenario, explain_agent, eval_agent)
        results.append(entry)
        print(f"    → {entry['status']}")

    elapsed = time.monotonic() - start

    report = _generate_report(results, elapsed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport written to {output_path}")
    print(report)

    # Return non-zero if any expectation mismatches
    mismatches = [r for r in results if r.get("expectation_mismatch")]
    return 1 if mismatches else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run golden scenario evaluations")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output markdown file path",
    )
    args = parser.parse_args()
    sys.exit(main(args.output))
