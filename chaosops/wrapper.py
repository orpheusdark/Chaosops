from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


class ChaosOpsWrapper:
    """Converts env observations to prompts and model text back to tool calls."""

    TOOLS = [
        "query_system(payload={})",
        "get_schema(payload={})",
        "request_access(payload={\"justification\": \"...\"})",
        "fix_service(payload={\"config\": {...}, \"token\": \"...\"})",
    ]

    def observation_to_prompt(self, observation: Dict[str, Any]) -> str:
        tools = "\n".join(f"- {t}" for t in self.TOOLS)
        return (
            "You are an autonomous SRE agent in ChaosOps.\n"
            "Choose the next action to recover auth_service.\n"
            "Return STRICT JSON only with this shape:\n"
            "{\"action\":\"query_system|get_schema|request_access|fix_service\",\"payload\":{...}}\n\n"
            f"Observation:\n{json.dumps(observation, ensure_ascii=True)}\n\n"
            f"Available tools:\n{tools}\n"
        )

    def _extract_json_block(self, text: str) -> Optional[str]:
        text = text.strip()
        if not text:
            return None

        if text.startswith("{") and text.endswith("}"):
            return text

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)

        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace:
            return brace.group(0)

        return None

    def parse_model_output(self, text: str) -> Dict[str, Any]:
        raw_json = self._extract_json_block(text)
        if not raw_json:
            return {"action": "query_system", "payload": {}}

        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            return {"action": "query_system", "payload": {}}

        if not isinstance(obj, dict):
            return {"action": "query_system", "payload": {}}

        action = obj.get("action")
        payload = obj.get("payload", {})

        if action not in {"query_system", "get_schema", "request_access", "fix_service"}:
            action = "query_system"
        if not isinstance(payload, dict):
            payload = {}

        return {"action": action, "payload": payload}
