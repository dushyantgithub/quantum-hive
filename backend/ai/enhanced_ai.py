"""
Enhanced AI Engine for Quantum Hive
Provides sophisticated responses using advanced pattern matching and NLP techniques
"""
import logging
import re
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedAIEngine:
    """Enhanced AI engine with sophisticated response generation"""
    
    def __init__(self):
        """Initialize the enhanced AI engine"""
        self.conversation_history = []
        self.user_preferences = {}
        self.context = {}
        
        # Load response patterns and templates
        self._load_response_patterns()
        self._load_conversation_templates()
        
        logger.info("Enhanced AI engine initialized")
    
    def _load_response_patterns(self):
        """Load sophisticated response patterns"""
        self.patterns = {
            # Greetings and introductions
            "greetings": {
                "patterns": [
                    r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
                    r"\b(how are you|how's it going|how do you do)\b"
                ],
                "responses": [
                    "Hello! I'm Quantum Hive, your AI assistant. How can I help you today?",
                    "Hi there! I'm here to assist you with whatever you need.",
                    "Hello! I'm Quantum Hive, ready to help you with tasks, questions, or just conversation.",
                    "Greetings! I'm your AI assistant. What would you like to work on today?"
                ]
            },
            
            # Capabilities and help
            "capabilities": {
                "patterns": [
                    r"\b(what can you do|what are your capabilities|help|what do you do)\b",
                    r"\b(can you|are you able to|do you know how to)\b"
                ],
                "responses": [
                    "I'm Quantum Hive, your AI assistant! I can help with:\n• Answering questions and providing information\n• Having natural conversations\n• Setting reminders and managing tasks\n• Controlling smart home devices (coming soon)\n• Playing music and media\n• Providing weather updates\n• Helping with calculations and conversions\n\nWhat would you like to do?",
                    "I'm here to assist you with various tasks! I can answer questions, have conversations, help with productivity tasks, and much more. Just let me know what you need!",
                    "I'm your AI assistant with capabilities for conversation, information retrieval, task management, and smart home control. How can I help you today?"
                ]
            },
            
            # Time and date
            "time_date": {
                "patterns": [
                    r"\b(time|what time|current time|clock)\b",
                    r"\b(date|what date|today|what day)\b",
                    r"\b(day of week|what day is it)\b"
                ],
                "responses": [
                    "The current time is {time}.",
                    "It's {time} right now.",
                    "The time is {time}.",
                    "Today is {date}, and it's currently {time}."
                ]
            },
            
            # Weather (placeholder for future API integration)
            "weather": {
                "patterns": [
                    r"\b(weather|temperature|forecast|is it going to rain)\b",
                    r"\b(hot|cold|warm|cool)\b"
                ],
                "responses": [
                    "I don't have access to real-time weather data yet, but that's a feature I'm working on! I'll be able to provide weather updates soon.",
                    "Weather information isn't available right now, but I'm learning to access weather APIs for you.",
                    "I can't check the weather yet, but that capability is coming in a future update!"
                ]
            },
            
            # Math and calculations
            "math": {
                "patterns": [
                    r"\b(calculate|what is|how much is|math|arithmetic)\b",
                    r"\b(\d+\s*[\+\-\*\/]\s*\d+)",
                    r"\b(plus|minus|times|divided by|multiply|add|subtract)\b"
                ],
                "responses": [
                    "Let me calculate that for you: {result}",
                    "The answer is {result}.",
                    "That equals {result}."
                ]
            },
            
            # Entertainment and fun
            "entertainment": {
                "patterns": [
                    r"\b(joke|funny|humor|entertain me|tell me something)\b",
                    r"\b(play music|song|music)\b",
                    r"\b(game|play a game|fun)\b"
                ],
                "responses": [
                    "Here's a joke for you: Why don't scientists trust atoms? Because they make up everything! 😄",
                    "Want to hear something interesting? The shortest war in history was between Britain and Zanzibar in 1896. It lasted only 38 minutes!",
                    "I'd love to play music for you, but I need to connect to your media system first. That feature is coming soon!",
                    "I can tell you jokes, share interesting facts, or help you find entertainment. What would you prefer?"
                ]
            },
            
            # System control
            "system": {
                "patterns": [
                    r"\b(volume|turn up|turn down|mute|unmute)\b",
                    r"\b(brightness|dim|brighten|lights)\b",
                    r"\b(restart|reboot|shutdown|power off)\b"
                ],
                "responses": [
                    "I can help you control system settings. What would you like to adjust?",
                    "System control features are being developed. I'll be able to help with that soon!",
                    "I'm learning to control your system settings. That capability is coming in the next update."
                ]
            },
            
            # Personal assistance
            "personal": {
                "patterns": [
                    r"\b(reminder|remind me|set a reminder|schedule)\b",
                    r"\b(note|write down|remember)\b",
                    r"\b(todo|task|to do|checklist)\b"
                ],
                "responses": [
                    "I can help you set reminders and manage tasks. What would you like me to remind you about?",
                    "Task management is a great feature! I can help you organize your day. What do you need to remember?",
                    "I'm here to help with productivity! I can set reminders, take notes, and help manage your tasks."
                ]
            },
            
            # Exit and goodbye
            "exit": {
                "patterns": [
                    r"\b(goodbye|bye|exit|quit|stop|end)\b",
                    r"\b(see you|talk to you later|that's all)\b"
                ],
                "responses": [
                    "Goodbye! It was great talking with you. Have a wonderful day!",
                    "See you later! Don't hesitate to call me if you need anything else.",
                    "Goodbye! I'm here whenever you need assistance. Take care!",
                    "Farewell! It's been a pleasure helping you today."
                ]
            },
            
            # Gratitude
            "gratitude": {
                "patterns": [
                    r"\b(thank you|thanks|appreciate it|grateful)\b",
                    r"\b(awesome|great|excellent|perfect)\b"
                ],
                "responses": [
                    "You're very welcome! I'm happy to help.",
                    "My pleasure! I'm here to assist you anytime.",
                    "You're welcome! It's what I'm here for.",
                    "Glad I could help! Don't hesitate to ask if you need anything else."
                ]
            }
        }
    
    def _load_conversation_templates(self):
        """Load conversation templates for more natural responses"""
        self.templates = {
            "follow_up": [
                "Is there anything else you'd like to know about that?",
                "Would you like me to elaborate on that?",
                "Do you have any other questions?",
                "Is there anything specific you'd like to explore further?"
            ],
            "clarification": [
                "Could you please clarify what you mean?",
                "I want to make sure I understand correctly. Could you rephrase that?",
                "I'm not quite sure what you're asking. Could you explain a bit more?",
                "To help you better, could you provide more details?"
            ],
            "encouragement": [
                "That's a great question!",
                "I'm glad you asked about that.",
                "That's an interesting topic to explore.",
                "Let me help you with that."
            ]
        }
    
    def _extract_math_expression(self, text: str) -> Optional[str]:
        """Extract mathematical expressions from text"""
        # Simple pattern for basic arithmetic
        math_pattern = r'(\d+\s*[\+\-\*\/]\s*\d+)'
        match = re.search(math_pattern, text)
        if match:
            return match.group(1)
        return None
    
    def _evaluate_math(self, expression: str) -> Optional[str]:
        """Safely evaluate mathematical expressions"""
        try:
            # Clean the expression
            clean_expr = re.sub(r'[^\d\+\-\*\/\(\)\.]', '', expression)
            result = eval(clean_expr)
            return str(result)
        except:
            return None
    
    def _get_current_time(self) -> Tuple[str, str]:
        """Get current time and date"""
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%B %d, %Y")
        return time_str, date_str
    
    def _match_patterns(self, text: str) -> Tuple[str, float]:
        """Match text against patterns and return best match with confidence"""
        text_lower = text.lower()
        best_match = None
        best_confidence = 0.0
        
        for category, pattern_data in self.patterns.items():
            for pattern in pattern_data["patterns"]:
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Calculate confidence based on match length and frequency
                    confidence = len(matches) * len(matches[0]) / len(text_lower)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = category
        
        return best_match, best_confidence
    
    def _generate_response(self, category: str, text: str) -> str:
        """Generate a response based on the matched category"""
        if category not in self.patterns:
            return self._generate_fallback_response(text)
        
        responses = self.patterns[category]["responses"]
        
        # Handle special cases
        if category == "time_date":
            time_str, date_str = self._get_current_time()
            response = random.choice(responses)
            response = response.replace("{time}", time_str).replace("{date}", date_str)
            return response
        
        elif category == "math":
            expression = self._extract_math_expression(text)
            if expression:
                result = self._evaluate_math(expression)
                if result:
                    response = random.choice(responses)
                    return response.replace("{result}", f"{expression} = {result}")
        
        # Return random response from category
        return random.choice(responses)
    
    def _generate_fallback_response(self, text: str) -> str:
        """Generate a fallback response when no pattern matches"""
        fallback_responses = [
            "That's an interesting point! Could you tell me more about what you're looking for?",
            "I'm not quite sure how to respond to that. Could you rephrase or ask something else?",
            "I'm still learning and that's a bit outside my current capabilities. Is there something else I can help you with?",
            "That's a great question, but I might not have the right information for that. What else can I assist you with?",
            "I want to make sure I understand correctly. Could you clarify what you're asking about?"
        ]
        return random.choice(fallback_responses)
    
    def _add_context(self, text: str, response: str):
        """Add conversation context for better future responses"""
        self.conversation_history.append({
            "timestamp": time.time(),
            "user_input": text,
            "ai_response": response,
            "category": "conversation"
        })
        
        # Keep only recent history (last 10 exchanges)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate an enhanced response to user input
        
        Args:
            prompt (str): User input text
            system_prompt (str, optional): System prompt for context
            
        Returns:
            str: Generated response
        """
        try:
            # Clean and normalize input
            prompt = prompt.strip()
            if not prompt:
                return "I didn't catch that. Could you please repeat?"
            
            # Match patterns and get confidence
            category, confidence = self._match_patterns(prompt)
            
            # Generate response
            if confidence > 0.1:  # Threshold for pattern matching
                response = self._generate_response(category, prompt)
            else:
                response = self._generate_fallback_response(prompt)
            
            # Add context for future responses
            self._add_context(prompt, response)
            
            # Add follow-up question occasionally
            if random.random() < 0.3 and category not in ["exit", "gratitude"]:
                follow_up = random.choice(self.templates["follow_up"])
                response += f" {follow_up}"
            
            logger.info(f"Enhanced AI response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}")
            return "I encountered an error processing your request. Could you try again?"
    
    def chat_with_context(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response with conversation context
        
        Args:
            messages (List[Dict]): List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: Generated response
        """
        try:
            if not messages:
                return "Hello! How can I help you today?"
            
            # Get the last user message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_message = msg.get("content", "")
                    break
            
            if not last_message:
                return "I didn't catch that. Could you please repeat?"
            
            # Generate response using the enhanced engine
            return self.generate_response(last_message)
            
        except Exception as e:
            logger.error(f"Error in chat with context: {e}")
            return "I encountered an error. Could you try again?"
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the AI model"""
        return {
            "name": "Enhanced AI Engine",
            "version": "2.0",
            "type": "Pattern-based with NLP",
            "capabilities": [
                "Natural language understanding",
                "Context-aware responses",
                "Mathematical calculations",
                "Time and date information",
                "Conversation management",
                "Task assistance"
            ],
            "description": "Advanced pattern-matching AI with sophisticated response generation"
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.conversation_history.clear()
            self.context.clear()
            logger.info("Enhanced AI engine cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Test the enhanced AI engine
    logging.basicConfig(level=logging.INFO)
    print("Testing Enhanced AI Engine...")
    
    ai_engine = EnhancedAIEngine()
    
    test_queries = [
        "Hello, how are you?",
        "What can you do?",
        "What time is it?",
        "Tell me a joke",
        "Calculate 15 + 27",
        "Thank you for your help",
        "Goodbye"
    ]
    
    try:
        for query in test_queries:
            print(f"\nUser: {query}")
            response = ai_engine.generate_response(query)
            print(f"AI: {response}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ai_engine.cleanup() 