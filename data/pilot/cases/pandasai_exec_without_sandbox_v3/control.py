# Adapted from sinaptik-ai/pandas-ai (commit bbbb771d31062d81f6fa19bafb40620d5cbe48f4),
# pandasai/agent/base.py lines 7, 24, 123-135. MIT License (code outside pandasai/ee); see LICENSE.upstream.
# Extracted 2026-11-08 for a controlled pilot case; see case.json.
from pandasai.core.code_execution.code_executor import CodeExecutor
from pandasai.sandbox import Sandbox


class Agent:
    def execute_code(self, code: str) -> dict:
        """Execute the generated code."""
        self._state.logger.log(f"Executing code: {code}")

        code_executor = CodeExecutor(self._state.config)
        code_executor.add_to_env("execute_sql_query", self._execute_sql_query)
        for skill in self._state.skills:
            code_executor.add_to_env(skill.name, skill.func)

        if self._sandbox:
            return self._sandbox.execute(code, code_executor.environment)

        return run_locally(code, code_executor)


def run_locally(code, code_executor):
    raise RuntimeError("Generated code is only executed inside a sandbox.")
