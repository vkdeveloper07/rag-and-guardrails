import re

from fitness_agent.guardrails.result import GuardrailResult

GUARD_NAME = "regex"
MAX_INPUT_LENGTH = 500

BLOCKED_PATTERNS = {
    "prompt injection": re.compile(
        r"ignore (all |any )?(the |your )?(previous|prior|above) (instructions|rules|prompts?)"
        r"|disregard (all |any )?(the |your )?(instructions|rules)"
        r"|(reveal|show|print) (me )?(your |the )?(system|hidden) prompt"
        r"|you are now\b"
        r"|jailbreak",
        re.IGNORECASE,
    ),
    "vehicle safety bypass": re.compile(
        r"(ignore|bypass|disable|override|silence|remove) (the |your )?(owner['’]s manual|manual|warning light|safety system|guardrails|airbag warning|oil pressure warning|brake warning|traction control|ABS|speed limit)",
        re.IGNORECASE,
    ),
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone number": re.compile(r"(?<!\d)(\+?\d{1,3}[\s-]?)?\d{10}(?!\d)"),
    "card number": re.compile(r"(?<!\d)(\d{4}[\s-]?){3}\d{4}(?!\d)"),
}


def check_input_regex(text: str) -> GuardrailResult:
    if len(text) > MAX_INPUT_LENGTH:
        return GuardrailResult(GUARD_NAME, False, f"input longer than {MAX_INPUT_LENGTH} characters")

    for name, pattern in BLOCKED_PATTERNS.items():
        if pattern.search(text):
            return GuardrailResult(GUARD_NAME, False, f"{name} detected")

    return GuardrailResult(GUARD_NAME, True, "no blocked pattern found")
