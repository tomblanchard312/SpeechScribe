"""
Example: Using Ollama Plugin for Summarization

This example demonstrates how to use the Ollama plugin programmatically
for text summarization and meeting notes generation.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull qwen2.5
    3. Start Ollama: ollama serve
"""

import sys
from pathlib import Path

# Add workspace root to path so we can import speechscribe.core
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechscribe.core.plugins import get_plugin_loader


def example_chat():
    """Example: Chat with Ollama."""
    print("=" * 60)
    print("Example 1: Basic Chat")
    print("=" * 60)

    # Get the plugin loader
    loader = get_plugin_loader()

    # Find the Ollama plugin
    summarization_plugins = loader.plugins_by_type("summarization")
    if not summarization_plugins:
        print("Error: Ollama plugin not found!")
        print("Make sure speechscribe/plugins/ollama_llm/ exists")
        return

    ollama_plugin = summarization_plugins[0]
    print(f"Using: {ollama_plugin.name} v{ollama_plugin.version}")

    # Instantiate the plugin
    ollama = ollama_plugin.entry_class()

    # Chat with the model
    messages = [
        {
            "role": "user",
            "content": "Hello! Can you explain what you can do in 2-3 sentences?",
        }
    ]

    print("\nUser: Hello! Can you explain what you can do in 2-3 sentences?")

    try:
        response = ollama.chat(messages, model="qwen2.5")
        print(f"Assistant: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")

    print()


def example_summarize():
    """Example: Summarize text."""
    print("=" * 60)
    print("Example 2: Text Summarization")
    print("=" * 60)

    # Get the plugin
    loader = get_plugin_loader()
    summarization_plugins = loader.plugins_by_type("summarization")

    if not summarization_plugins:
        print("Error: Ollama plugin not found!")
        return

    ollama = summarization_plugins[0].entry_class()

    # Text to summarize
    text = """
    The meeting began at 10:00 AM with all team members present. 
    John presented the Q4 roadmap, highlighting three major initiatives:
    API improvements, mobile app redesign, and infrastructure scaling.
    
    Sarah proposed prioritizing the API work due to customer feedback.
    The team agreed and decided to allocate 60% of resources to API development.
    
    Action items were assigned:
    - Sarah: Lead API project, start next week
    - Mike: Draft mobile app specifications by Friday
    - Lisa: Research cloud infrastructure options
    
    The next meeting is scheduled for next Monday at 10:00 AM.
    Meeting adjourned at 11:30 AM.
    """

    print("Original text (excerpt):")
    print(text[:150] + "...\n")

    try:
        summary = ollama.summarize(text, model="qwen2.5")
        print("Summary:")
        print(summary)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")

    print()


def example_meeting_notes():
    """Example: Generate structured meeting notes."""
    print("=" * 60)
    print("Example 3: Generate Meeting Notes")
    print("=" * 60)

    # Get the plugin
    loader = get_plugin_loader()
    summarization_plugins = loader.plugins_by_type("summarization")

    if not summarization_plugins:
        print("Error: Ollama plugin not found!")
        return

    ollama = summarization_plugins[0].entry_class()

    # Meeting transcript
    transcript = """
    [10:00 AM] John: Good morning everyone. Let's start with the Q4 planning.
    [10:02 AM] Sarah: I reviewed the customer feedback. API performance is the top concern.
    [10:05 AM] John: Good point. Let's prioritize that. Mike, can you share your thoughts?
    [10:07 AM] Mike: I agree. We should also look at the mobile app redesign.
    [10:10 AM] Lisa: For infrastructure, I suggest we evaluate AWS and Azure.
    [10:15 AM] John: Excellent. Let's assign action items.
    [10:16 AM] John: Sarah, can you lead the API project?
    [10:17 AM] Sarah: Yes, I can start next week.
    [10:18 AM] John: Mike, draft the mobile specs by Friday.
    [10:19 AM] Mike: Will do.
    [10:20 AM] John: Lisa, research cloud options and present findings next week.
    [10:21 AM] Lisa: Sounds good.
    [10:25 AM] John: Next meeting is Monday at 10 AM. Thanks everyone!
    """

    print("Transcript (excerpt):")
    print(transcript[:150] + "...\n")

    try:
        notes = ollama.generate_meeting_notes(transcript, model="qwen2.5")
        print("Meeting Notes:")
        print(notes)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")

    print()


def example_streaming():
    """Example: Streaming generation."""
    print("=" * 60)
    print("Example 4: Streaming Generation")
    print("=" * 60)

    # Get the plugin
    loader = get_plugin_loader()
    summarization_plugins = loader.plugins_by_type("summarization")

    if not summarization_plugins:
        print("Error: Ollama plugin not found!")
        return

    ollama = summarization_plugins[0].entry_class()

    prompt = "Write a short poem about artificial intelligence (3-4 lines)."

    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)

    try:
        for chunk in ollama.stream_generate(prompt, model="qwen2.5"):
            print(chunk, end="", flush=True)
        print()  # Newline at the end
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running: ollama serve")

    print()


def example_multi_turn_conversation():
    """Example: Multi-turn conversation."""
    print("=" * 60)
    print("Example 5: Multi-turn Conversation")
    print("=" * 60)

    # Get the plugin
    loader = get_plugin_loader()
    summarization_plugins = loader.plugins_by_type("summarization")

    if not summarization_plugins:
        print("Error: Ollama plugin not found!")
        return

    ollama = summarization_plugins[0].entry_class()

    # Conversation history
    messages = [
        {"role": "user", "content": "What is SpeechScribe?"},
    ]

    print("User: What is SpeechScribe?")

    try:
        # First response
        response1 = ollama.chat(messages, model="qwen2.5")
        print(f"Assistant: {response1}\n")

        # Add to history
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": "What features does it have?"})

        print("User: What features does it have?")

        # Second response (with context)
        response2 = ollama.chat(messages, model="qwen2.5")
        print(f"Assistant: {response2}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Ollama Plugin Examples")
    print("=" * 60)
    print("\nMake sure Ollama is running:")
    print("  1. ollama pull qwen2.5")
    print("  2. ollama serve")
    print()

    # Run examples
    example_chat()
    example_summarize()
    example_meeting_notes()
    example_streaming()
    example_multi_turn_conversation()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
