from capy.chat_engine import CapyChatEngine


def test_chat_engine():
    chat_engine = CapyChatEngine()
    chat_engine.load_llm()
    chat_engine.run_interactive_cli()
    chat_engine.offload_llm()


if __name__ == "__main__": 
    test_chat_engine()