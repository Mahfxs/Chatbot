"""
Unit Tests for Nova Chatbot
Run with: pytest test_chatbot.py -v
"""

import pytest
from chatbot import Chatbot, IntentRecognizer, DialogueManager, ResponseGenerator, tokenize, compute_similarity


# ─────────────────────────────────────────────
#  NLP Utility Tests
# ─────────────────────────────────────────────

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]

def test_tokenize_punctuation():
    tokens = tokenize("Hi! How are you?")
    assert "hi" in tokens
    assert "how" in tokens
    assert "are" in tokens

def test_compute_similarity_full_match():
    assert compute_similarity(["hello", "world"], ["hello", "world"]) == 1.0

def test_compute_similarity_partial():
    score = compute_similarity(["hello"], ["hello", "world"])
    assert score == 0.5

def test_compute_similarity_no_match():
    assert compute_similarity(["foo"], ["hello", "world"]) == 0.0

def test_compute_similarity_empty_keywords():
    assert compute_similarity(["hello"], []) == 0.0


# ─────────────────────────────────────────────
#  Intent Recognizer Tests
# ─────────────────────────────────────────────

@pytest.fixture
def recognizer():
    return IntentRecognizer("intents.json")

def test_recognizer_greeting(recognizer):
    intent, score = recognizer.predict("hello")
    assert intent == "greeting"
    assert score > 0

def test_recognizer_farewell(recognizer):
    intent, score = recognizer.predict("goodbye")
    assert intent == "farewell"

def test_recognizer_thanks(recognizer):
    intent, score = recognizer.predict("thank you so much")
    assert intent == "thanks"

def test_recognizer_fallback_on_gibberish(recognizer):
    intent, score = recognizer.predict("xyzzy florp blarg")
    # Should either fallback or return low confidence
    assert score < 0.5


# ─────────────────────────────────────────────
#  Dialogue Manager Tests
# ─────────────────────────────────────────────

@pytest.fixture
def dialogue():
    return DialogueManager()

def test_dialogue_initial_state(dialogue):
    assert dialogue.turn_count == 0
    assert dialogue.history == []

def test_dialogue_update(dialogue):
    dialogue.update("hello", "greeting", "Hi there!")
    assert dialogue.turn_count == 1
    assert len(dialogue.history) == 1
    assert dialogue.get_last_intent() == "greeting"

def test_dialogue_reset(dialogue):
    dialogue.update("hello", "greeting", "Hi!")
    dialogue.reset()
    assert dialogue.turn_count == 0
    assert dialogue.history == []

def test_dialogue_multiple_turns(dialogue):
    dialogue.update("hi", "greeting", "Hello!")
    dialogue.update("thanks", "thanks", "You're welcome!")
    assert dialogue.turn_count == 2
    assert dialogue.get_last_intent() == "thanks"


# ─────────────────────────────────────────────
#  Response Generator Tests
# ─────────────────────────────────────────────

@pytest.fixture
def generator():
    return ResponseGenerator("intents.json")

def test_generator_greeting(generator):
    response = generator.generate("greeting")
    assert isinstance(response, str)
    assert len(response) > 0

def test_generator_fallback(generator):
    response = generator.generate("unknown_intent_xyz")
    assert isinstance(response, str)

def test_generator_time_placeholder(generator):
    response = generator.generate("time")
    assert "{time}" not in response  # Placeholder should be replaced

def test_generator_date_placeholder(generator):
    response = generator.generate("date")
    assert "{date}" not in response


# ─────────────────────────────────────────────
#  Full Chatbot Integration Tests
# ─────────────────────────────────────────────

@pytest.fixture
def bot():
    return Chatbot(name="TestBot")

def test_bot_responds_to_greeting(bot):
    response = bot.respond("hello")
    assert isinstance(response, str)
    assert len(response) > 0

def test_bot_handles_empty_input(bot):
    response = bot.respond("")
    assert "something" in response.lower()

def test_bot_logs_history(bot):
    bot.respond("hi")
    bot.respond("thanks")
    assert len(bot.dialogue.history) == 2

def test_bot_low_confidence_fallback(bot):
    response = bot.respond("zzzzz qqqqq aaaaa")
    assert isinstance(response, str)
