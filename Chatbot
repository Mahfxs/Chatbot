"""
AI Chatbot with NLP - Intent Recognition & Dialogue Management
Author: Your Name
Description: A Python-based chatbot using NLP for intent recognition,
             dialogue management, and response generation.
"""

import re
import json
import random
from datetime import datetime


# ─────────────────────────────────────────────
#  NLP Utilities
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r'\b\w+\b', text.lower())


def compute_similarity(tokens: list[str], keywords: list[str]) -> float:
    """Return ratio of matched keywords to total keywords."""
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw in tokens)
    return matches / len(keywords)


# ─────────────────────────────────────────────
#  Intent Engine
# ─────────────────────────────────────────────

class IntentRecognizer:
    """
    Lightweight rule-based intent recognizer.
    Replace the `intents` dict with a trained ML model for production use.
    """

    def __init__(self, intents_path: str = "intents.json"):
        with open(intents_path, "r") as f:
            data = json.load(f)
        self.intents = data["intents"]

    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent_tag, confidence_score) for the given text."""
        tokens = tokenize(text)
        best_intent = "fallback"
        best_score = 0.0

        for intent in self.intents:
            for pattern in intent["patterns"]:
                pattern_tokens = tokenize(pattern)
                score = compute_similarity(tokens, pattern_tokens)
                if score > best_score:
                    best_score = score
                    best_intent = intent["tag"]

        return best_intent, round(best_score, 2)


# ─────────────────────────────────────────────
#  Dialogue Manager
# ─────────────────────────────────────────────

class DialogueManager:
    """
    Manages conversation state and context across turns.
    Tracks history and handles multi-turn dialogue flows.
    """

    def __init__(self):
        self.history: list[dict] = []
        self.context: dict = {}
        self.turn_count: int = 0

    def update(self, user_input: str, intent: str, response: str):
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "intent": intent,
            "bot": response,
        })
        self.context["last_intent"] = intent

    def get_last_intent(self) -> str:
        return self.context.get("last_intent", "")

    def reset(self):
        self.history.clear()
        self.context.clear()
        self.turn_count = 0

    def export_history(self, path: str = "conversation_log.json"):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[✔] Conversation saved to {path}")


# ─────────────────────────────────────────────
#  Response Generator
# ─────────────────────────────────────────────

class ResponseGenerator:
    """
    Generates responses based on detected intent.
    Supports templated, random, and context-aware replies.
    """

    def __init__(self, intents_path: str = "intents.json"):
        with open(intents_path, "r") as f:
            data = json.load(f)
        self.responses = {
            intent["tag"]: intent["responses"]
            for intent in data["intents"]
        }

    def generate(self, intent: str, context: dict = None) -> str:
        responses = self.responses.get(intent, self.responses.get("fallback", ["I'm not sure how to respond to that."]))
        response = random.choice(responses)

        # Dynamic placeholders
        response = response.replace("{time}", datetime.now().strftime("%H:%M"))
        response = response.replace("{date}", datetime.now().strftime("%B %d, %Y"))

        return response


# ─────────────────────────────────────────────
#  Main Chatbot Class
# ─────────────────────────────────────────────

class Chatbot:
    """
    Core chatbot orchestrator combining:
    - Intent Recognition
    - Dialogue Management
    - Response Generation
    """

    CONFIDENCE_THRESHOLD = 0.2

    def __init__(self, name: str = "Nova", intents_path: str = "intents.json"):
        self.name = name
        self.recognizer = IntentRecognizer(intents_path)
        self.dialogue = DialogueManager()
        self.generator = ResponseGenerator(intents_path)
        print(f"\n{'='*50}")
        print(f"  {self.name} Chatbot — Ready")
        print(f"  Type 'quit' to exit | 'history' to view log")
        print(f"{'='*50}\n")

    def respond(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please say something!"

        intent, confidence = self.recognizer.predict(user_input)

        # Fall back if confidence is too low
        if confidence < self.CONFIDENCE_THRESHOLD:
            intent = "fallback"

        response = self.generator.generate(intent, self.dialogue.context)
        self.dialogue.update(user_input, intent, response)

        return response

    def run(self):
        """Start the interactive CLI chat loop."""
        print(f"{self.name}: Hello! I'm {self.name}, your AI assistant. How can I help you today?\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{self.name}: Goodbye! Have a great day! 👋")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "bye"):
                print(f"{self.name}: Goodbye! Have a great day! 👋")
                self.dialogue.export_history()
                break

            if user_input.lower() == "history":
                print(json.dumps(self.dialogue.history, indent=2))
                continue

            if user_input.lower() == "reset":
                self.dialogue.reset()
                print(f"{self.name}: Conversation reset. Starting fresh!\n")
                continue

            response = self.respond(user_input)
            print(f"{self.name}: {response}\n")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    bot = Chatbot(name="Nova")
    bot.run()
