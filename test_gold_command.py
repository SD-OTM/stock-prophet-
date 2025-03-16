#!/usr/bin/env python3

"""
Test script for simulating the /gold command
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.append('.')
import main

# Mock the Telegram Update and Context objects
class MockUpdate:
    class MockMessage:
        class MockUser:
            def __init__(self, user_id):
                self.id = user_id
                
        def __init__(self, user_id):
            self.from_user = self.MockUser(user_id)
            self.message_id = 123
            
        def reply_text(self, text, parse_mode=None):
            print("\n=== TELEGRAM BOT RESPONSE ===\n")
            print(text)
            print("\n=========================\n")
            
    def __init__(self, user_id):
        self.message = self.MockMessage(user_id)
        self.effective_user = self.MockMessage.MockUser(user_id)

class MockContext:
    class MockBot:
        def send_message(self, chat_id, text, parse_mode=None):
            print(f"\n=== BOT SENDING MESSAGE TO {chat_id} ===\n")
            print(text)
            print("\n=========================\n")
    
    def __init__(self, args=None):
        self.args = args or []
        self.bot = self.MockBot()
        
# Test the /gold command
print("\n=== TESTING /GOLD COMMAND ===\n")

# Create mock update and context objects
update = MockUpdate("test_user")
context = MockContext()

# Call the check_gold_price function directly
main.check_gold_price(update, context)

print("\n=== TEST COMPLETE ===\n")