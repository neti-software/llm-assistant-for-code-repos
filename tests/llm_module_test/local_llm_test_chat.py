from src.llm_module.local_llm import LocalLLM
from src.utils.helper import load_yaml


def main():
    config_path = "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/configs/llm_config.yaml"
    config = load_yaml(config_path)["llm"]
    llm = LocalLLM(config)

    print(f"🔗 Connected to MCP server at {config['endpoint']}")
    print("Type your message (or 'exit' to quit)\n")

    try:
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                print("👋 Bye!")
                break

            try:
                answer = llm.generate(prompt, max_tokens=200, temperature=0.7)
                print(f"LLM: {answer.strip()}\n")
            except Exception as e:
                print(f"⚠️ Error: {e}")

    except KeyboardInterrupt:
        print("\n👋 Bye!")

    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
