#!/usr/bin/env python3
"""Apply the pre-registered lossless editorial fixes to eight appraisal bundles.

The v3 repair run produced 792 technically valid bundles and eight bundles with
presentation defects. This script does not generate text or change scores,
evidence citations, hypotheses, or tradeoffs. It removes persona-construction
wording, translates one two-character fragment, and trims schema-capped prose
back to its last complete model-generated sentence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.interviews import APPRAISAL_PROSE_LIMITS
from src.vllm_backend import VLLMOfflineBackend


EXPECTED_RAW_SHA256 = {
    "appraisal_end_of_shift_survey_bundle_1e0560a9cccdc5cb77ec":
        "39cdc140d4b902591cdcc7a3e673f2a6b2dc9a4562207e74bef1cb1a96825877",
    "appraisal_end_of_shift_interview_bundle_f89035af0714785f11b0":
        "eb751a463f3588552b316f13b8a32075acd95f983204fd6b299e23f35051dcf4",
    "appraisal_end_of_shift_survey_bundle_73977ca4ed21a23a6e5d":
        "404e3600bbae1a65bf6f817dcb0b82cb0c04d8619d8605691df9478889237f5b",
    "appraisal_end_of_shift_survey_bundle_32b3eb0e128f3b979016":
        "93ca0505b97b9d4d639e956880761b17ffad1db4dd23235e1868838a963f4dfc",
    "appraisal_end_of_shift_interview_bundle_277e18b177d16e9c5503":
        "91803b5a01b860f59fa6269545675c2459ed80c3e9f73aabb257778f5ffa1d16",
    "appraisal_end_of_shift_interview_bundle_6f087e7215f0b3860b2e":
        "f626ebcd5a3f469ac4a9dbe2875a618ccd72598469dd641892ce688835e6b2df",
    "appraisal_end_of_shift_interview_bundle_e3ddb6b184f0d3249e54":
        "639758f5ef6b203ccd4c61d109561bc1944b166c60e520075d2b322f80d536de",
    "appraisal_end_of_shift_survey_bundle_cb047580ce618bbb6eb5":
        "d468bf3a77ca2ee7caa93c9a28386ea1972555b34f5956dca0d378fcd6539577",
}

SURVEY_REPLACEMENTS = {
    "appraisal_end_of_shift_survey_bundle_1e0560a9cccdc5cb77ec": [
        ("the orientation's priorities", "my priorities"),
    ],
    "appraisal_end_of_shift_survey_bundle_73977ca4ed21a23a6e5d": [
        ("the orientation's priorities", "my priorities"),
    ],
    "appraisal_end_of_shift_survey_bundle_32b3eb0e128f3b979016": [
        (
            "This selective engagement aligns with the orientation to accept only "
            "high-value interruptions.",
            "This selective engagement supports accepting only high-value interruptions.",
        ),
        (
            "The orientation to defer non-urgent contacts fits well",
            "Deferring non-urgent contacts fits well",
        ),
    ],
    "appraisal_end_of_shift_survey_bundle_cb047580ce618bbb6eb5": [
        ("the orientation's priorities", "my priorities"),
    ],
}

INTERVIEW_REPLACEMENTS = {
    "appraisal_end_of_shift_interview_bundle_f89035af0714785f11b0": [
        ("potential延误", "potential delay"),
    ],
}

_COMPLETE_SENTENCE = re.compile(r"[.!?](?:[\"']|\))?(?=\s|$)")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trim_incomplete_prose(
    value: str,
    maximum_length: int,
    *,
    require_character_cap: bool = True,
) -> tuple[str, bool]:
    """Remove an unfinished tail while retaining complete generated sentences."""

    text = str(value).strip()
    if re.search(r"[.!?](?:[\"']|\))?$", text):
        return text, False
    if require_character_cap and len(text) < maximum_length:
        return text, False
    endings = list(_COMPLETE_SENTENCE.finditer(text))
    if not endings:
        raise ValueError("Capped prose has no complete generated sentence to retain")
    trimmed = text[: endings[-1].end()].strip()
    if not trimmed:
        raise ValueError("Capped prose trim produced an empty value")
    return trimmed, True


def _replace_exact(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one exact occurrence of {old!r}; observed {count}")
    return text.replace(old, new, 1)


def _survey_edits(
    prompt_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    edits = []
    for old, new in SURVEY_REPLACEMENTS[prompt_id]:
        matches = []
        for index, response in enumerate(payload.get("responses", [])):
            for field in ("short_rationale", "uncertainty_note"):
                if old in str(response.get(field, "")):
                    matches.append((index, field))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one survey field containing {old!r}; observed {matches}"
            )
        index, field = matches[0]
        response = payload["responses"][index]
        response[field] = _replace_exact(str(response[field]), old, new)
        edits.append(
            {
                "operation": "remove_persona_construction_wording",
                "question_or_dimension": response.get("dimension"),
                "field": field,
            }
        )
    return edits


def _interview_edits(
    prompt_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    edits = []
    for old, new in INTERVIEW_REPLACEMENTS.get(prompt_id, []):
        matches = []
        for index, answer in enumerate(payload.get("answers", [])):
            for field in APPRAISAL_PROSE_LIMITS:
                if old in str(answer.get(field, "")):
                    matches.append((index, field))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one interview field containing {old!r}; observed {matches}"
            )
        index, field = matches[0]
        answer = payload["answers"][index]
        answer[field] = _replace_exact(str(answer[field]), old, new)
        edits.append(
            {
                "operation": "translate_non_English_fragment",
                "question_or_dimension": answer.get("question_id"),
                "field": field,
            }
        )

    for answer in payload.get("answers", []):
        for field, maximum_length in APPRAISAL_PROSE_LIMITS.items():
            value = answer.get(field)
            if value is None:
                continue
            trimmed, changed = trim_incomplete_prose(
                str(value),
                maximum_length,
                require_character_cap=field != "answer",
            )
            if changed:
                answer[field] = trimmed
                edits.append(
                    {
                        "operation": "trim_incomplete_generated_tail",
                        "question_or_dimension": answer.get("question_id"),
                        "field": field,
                    }
                )
    return edits


def _read_packets(path: Path) -> dict[str, dict[str, Any]]:
    packets = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        packet = json.loads(line)
        prompt_id = str(packet.get("prompt_id", ""))
        if not prompt_id or prompt_id in packets:
            raise ValueError(f"Invalid prompt ID at {path}:{line_number}")
        packets[prompt_id] = packet
    return packets


def finalize(
    input_path: Path,
    packet_path: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    packets = _read_packets(packet_path)
    backend = VLLMOfflineBackend()
    output_lines = []
    observed_ids = set()
    record_audits = []
    unmodified_count = 0

    for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        prompt_id = str(record.get("prompt_id", ""))
        if prompt_id not in EXPECTED_RAW_SHA256:
            output_lines.append(line)
            unmodified_count += 1
            continue
        if prompt_id in observed_ids:
            raise ValueError(f"Duplicate target response: {prompt_id}")
        observed_ids.add(prompt_id)
        raw_before = str(record.get("raw_response", ""))
        observed_sha = _sha256(raw_before)
        expected_sha = EXPECTED_RAW_SHA256[prompt_id]
        if observed_sha != expected_sha:
            raise ValueError(
                f"Raw response lineage mismatch for {prompt_id}: {observed_sha}"
            )
        packet = packets.get(prompt_id)
        if packet is None:
            raise ValueError(f"Missing immutable packet for {prompt_id}")
        payload = json.loads(raw_before)
        packet_type = str(record.get("packet_type", ""))
        if packet_type == "end_of_shift_survey_bundle":
            edits = _survey_edits(prompt_id, payload)
        elif packet_type == "end_of_shift_interview_bundle":
            edits = _interview_edits(prompt_id, payload)
        else:
            raise ValueError(f"Unexpected target packet type: {packet_type}")
        try:
            normalized = backend.normalize_packet_response(packet, payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Finalized response failed for {prompt_id}: {error}") from error
        raw_after = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        updated = dict(record)
        updated["raw_response_before_deterministic_postprocessing"] = raw_before
        updated["raw_response"] = raw_after
        updated["response"] = normalized
        updated["deterministic_postprocessing"] = {
            "policy": "lossless_appraisal_editorial_finalization_v1",
            "edits": edits,
            "scores_or_evidence_changed": False,
        }
        updated.pop("generation_error", None)
        output_lines.append(json.dumps(updated, sort_keys=True, ensure_ascii=False))
        record_audits.append(
            {
                "prompt_id": prompt_id,
                "packet_type": packet_type,
                "raw_response_sha256_before": observed_sha,
                "raw_response_sha256_after": _sha256(raw_after),
                "edits": edits,
            }
        )

    missing = sorted(set(EXPECTED_RAW_SHA256) - observed_ids)
    if missing:
        raise ValueError(f"Missing expected repair records: {missing}")
    if unmodified_count != 792 or len(record_audits) != 8:
        raise ValueError(
            f"Unexpected finalization scope: unmodified={unmodified_count}, "
            f"edited={len(record_audits)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text("\n".join(output_lines) + "\n")
    temporary_path.replace(output_path)

    metadata_input = input_path.with_suffix(input_path.suffix + ".metadata.json")
    metadata = json.loads(metadata_input.read_text()) if metadata_input.is_file() else {}
    metadata.update(
        {
            "deterministic_postprocessing_policy":
                "lossless_appraisal_editorial_finalization_v1",
            "deterministic_postprocessing_record_count": len(record_audits),
            "unmodified_response_record_count": unmodified_count,
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    audit = {
        "finalization_pass": True,
        "input_response_count": unmodified_count + len(record_audits),
        "unmodified_response_record_count": unmodified_count,
        "edited_response_record_count": len(record_audits),
        "scores_or_evidence_changed": False,
        "policy": "lossless_appraisal_editorial_finalization_v1",
        "record_audits": record_audits,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize(
        Path(args.input).expanduser().resolve(),
        Path(args.packets).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        Path(args.audit_json).expanduser().resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
