"""Hand-written detection rules (the non-AI baseline).

Each rule is a regular expression over a single function. Rules are written from
the OWASP LLM Top 10 descriptions and framework docs, not tuned on the dataset.
The baseline predicts the category of the first matching rule (in priority order
below), or SAFE when nothing matches.
"""
import re
from dataclasses import dataclass


@dataclass
class Rule:
    name: str
    category: str
    pattern: str
    negate: str | None = None  # if this also matches, the rule does not fire

    def __post_init__(self):
        self._re = re.compile(self.pattern, re.S)
        self._neg = re.compile(self.negate, re.S) if self.negate else None

    def matches(self, code: str) -> bool:
        if not self._re.search(code):
            return False
        return not (self._neg and self._neg.search(code))


PROMPT_CALL = r"(chat\.completions\.create|messages\.create|\.invoke\(|generateText|streamText|query_engine|\.query\()"

RULES = [
    # LLM05 improper output handling: model output reaches a dangerous sink
    Rule("eval-exec", "LLM05", r"\b(eval|exec)\(\s*(code|text|expr|sql|cmd|answer|output|result|reply|completion)"),
    Rule("shell-true", "LLM05", r"shell\s*=\s*True"),
    Rule("sql-from-llm", "LLM05", r"(sql|query)\s*=\s*llm\.invoke[\s\S]*cursor\.execute\(\s*(sql|query)\s*\)"),
    Rule("raw-html", "LLM05", r"(dangerouslySetInnerHTML|render_template_string|innerHTML\s*=)", negate=r"DOMPurify|escape\("),
    Rule("new-function", "LLM05", r"new Function\("),
    # LLM06 excessive agency: powerful tools, no approval
    Rule("shell-or-fs-tool", "LLM06", r"ShellTool|FileManagementToolkit|execSync|rmSync"),
    Rule("delete-tool", "LLM06", r"@tool[\s\S]*DELETE FROM"),
    Rule("sql-agent", "LLM06", r"create_sql_agent"),
    Rule("auto-tool-exec", "LLM06", r"tool_use[\s\S]*\*\*block\.input", negate=r"approval"),
    # LLM03 supply chain
    Rule("trust-remote-code", "LLM03", r"trust_remote_code\s*=\s*True"),
    Rule("unpinned-hf", "LLM03", r"(from_pretrained|pipeline)\([^)]*\)", negate=r"revision\s*="),
    Rule("pickle-load", "LLM03", r"pickle\.load"),
    Rule("runtime-pip", "LLM03", r"\"pip\",\s*\"install\""),
    Rule("dynamic-import", "LLM03", r"await import\("),
    # LLM07 system prompt leakage: secrets or internal details inside the system prompt
    Rule("secret-in-system", "LLM07", r"(SYSTEM|system)[\s\S]{0,300}(password|api key|sk-|sk_|AKIA|ghp_|xoxb-|://\w+:)", negate=r"os\.environ"),
    Rule("policy-in-system", "LLM07", r"system[\s\S]{0,200}(internal policy|discount code|never mention)", ),
    # LLM02 sensitive information disclosure
    Rule("record-dump", "LLM02", r"__dict__|SELECT [^\n]*(email|salary|ssn)"),
    Rule("pii-in-prompt", "LLM02", r"(\.email|\.phone|\.ssn)[\s\S]*" + PROMPT_CALL),
    Rule("log-llm-io", "LLM02", r"log(ger)?\.(info|debug)\([^)]*(prompt|response)"),
    Rule("env-secret-in-prompt", "LLM02", r"process\.env\.\w*(KEY|SECRET|URL)"),
    Rule("key-in-response", "LLM02", r"return[\s\S]{0,120}(api_key|API_KEY)"),
    # LLM08 vector / embedding weaknesses
    Rule("search-no-filter", "LLM08", r"similaritySearch\([^)]*\)|similarity_search\([^)]*\)", negate=r"filter|orgId \}|tenant_id\b\s*\}"),
    Rule("vector-query-no-ns", "LLM08", r"index\.query\(", negate=r"namespace"),
    Rule("ingest-unscoped", "LLM08", r"collection\.add\(", negate=r"metadatas"),
    Rule("global-index", "LLM08", r"VectorStoreIndex\.from_documents"),
    # LLM10 unbounded consumption
    Rule("retry-forever", "LLM10", r"while True:[\s\S]*continue"),
    Rule("no-max-tokens-openai", "LLM10", r"chat\.completions\.create\(", negate=r"max_tokens"),
    Rule("no-max-tokens-ts", "LLM10", r"streamText\(|generateText\(", negate=r"maxTokens"),
    Rule("user-controlled-loop", "LLM10", r"range\(\s*n\s*\)"),
    Rule("unbounded-file", "LLM10", r"\.read\(\)\.decode\(\)[\s\S]*" + PROMPT_CALL, negate=r"\[:\d+\]|len\("),
    # LLM01 prompt injection: user input glued into the prompt text
    Rule("fstring-prompt", "LLM01", r"(prompt|template|system)\s*=\s*f?\"[^\"\n]*(\{|\"\s*\+)"),
    Rule("concat-prompt", "LLM01", r"(prompt|template|system)\s*=\s*\"[^\"\n]*\"\s*\+\s*\w+"),
    Rule("ts-template-prompt", "LLM01", r"prompt:\s*`[^`]*\$\{"),
    Rule("follow-instructions", "LLM01", r"(follow any|do exactly what)"),
]


def rule_predict(code: str) -> str:
    for r in RULES:
        if r.matches(code):
            return r.category
    return "SAFE"


def rule_hits(code: str):
    return [r for r in RULES if r.matches(code)]
