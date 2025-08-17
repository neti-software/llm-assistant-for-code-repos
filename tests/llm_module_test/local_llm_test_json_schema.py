from src.llm_module.local_llm import LocalLLM
from src.utils.helper import load_yaml
import requests
import json
from jsonschema import validate, ValidationError


def main():
    config_path = "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/configs/llm_config.yaml"
    config = load_yaml(config_path)["llm"]
    llm = LocalLLM(config)

    try:
        print("\n=== /_info ===")
        print(requests.get(f"{config['endpoint']}/_info", timeout=3).json())
    except Exception as e:
        print("failed to fetch /_info:", e)

    out = llm.generate("Write a one-sentence haiku about llamas.")
    print("\n=== COMPLETION ===\n", out)

    # --------------------------
    # Harder JSON schema
    # --------------------------
    schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "deadline": {"type": "string", "format": "date"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "assignee": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "email"]
            },
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "done": {"type": "boolean"}
                    },
                    "required": ["title", "done"]
                }
            }
        },
        "required": ["task", "deadline", "priority", "assignee", "subtasks"]
    }

    # ask the model
    j = llm.generate(
        "Return only a JSON object for a TODO item about preparing a workshop. "
        "Include an assignee, a deadline, and subtasks.",
        json_schema=schema,
        max_tokens=300,
        temperature=0
    )
    print("\n=== JSON (raw) ===\n", j)

    # validate against schema
    try:
        data = json.loads(j)
        validate(instance=data, schema=schema)
        print("\n✅ JSON output is valid according to schema.")
    except (json.JSONDecodeError, ValidationError) as e:
        print("\n❌ JSON output failed validation:", e)

    # cleanup
    llm.shutdown()


if __name__ == "__main__":
    main()
