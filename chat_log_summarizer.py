#Generalized draft for the AI Chat Log Summarizer

"""
AI Chat Log Summarizer

This script analyzes chat logs between a user and an AI assistant,
providing statistics and keyword analysis.
"""

import re
import os
from collections import Counter
import string
from typing import Dict, List, Tuple

# Stop words - common words we want to exclude from keyword analysis
STOP_WORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'to', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'and', 'but', 'or', 'as', 'if', 'at', 'by',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which',
    'me', 'him', 'whom', 'could', 'would'
}

def parse_chat_log(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Parse the chat log file and separate messages by speaker.
    
    Args:
        file_path: Path to the chat log file
        
    Returns:
        Tuple of lists containing user messages and AI messages
    """
    user_messages = []
    ai_messages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Use regex to find user and AI messages
            user_pattern = re.compile(r'User:(.*?)(?=AI:|$)', re.DOTALL)
            ai_pattern = re.compile(r'AI:(.*?)(?=User:|$)', re.DOTALL)
            
            user_matches = user_pattern.findall(content)
            ai_matches = ai_pattern.findall(content)
            
            # Clean up the extracted messages
            user_messages = [msg.strip() for msg in user_matches]
            ai_messages = [msg.strip() for msg in ai_matches]
            
        return user_messages, ai_messages
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

def extract_keywords(messages: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Extract the most frequently used words from the messages.
    
    Args:
        messages: List of messages to analyze
        top_n: Number of top keywords to return
        
    Returns:
        List of tuples containing (keyword, count)
    """
    # Combine all messages into one text
    all_text = ' '.join(messages).lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    all_text = all_text.translate(translator)
    
    # Split into words
    words = all_text.split()
    
    # Filter out stop words and single characters
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get the top N most common words
    return word_counts.most_common(top_n)

def generate_summary(user_messages: List[str], ai_messages: List[str]) -> str:
    """
    Generate a summary of the chat log.
    
    Args:
        user_messages: List of user messages
        ai_messages: List of AI messages
        
    Returns:
        A formatted summary string
    """
    # Count messages
    user_count = len(user_messages)
    ai_count = len(ai_messages)
    total_exchanges = min(user_count, ai_count)  # A complete exchange requires both user and AI
    
    # Extract keywords from user questions to understand the nature of conversation
    user_keywords = extract_keywords(user_messages)
    all_keywords = extract_keywords(user_messages + ai_messages)
    
    # Format the keywords as strings
    user_keywords_str = ', '.join([f"{word}" for word, _ in user_keywords])
    all_keywords_str = ', '.join([f"{word}" for word, _ in all_keywords])
    
    # Determine the nature of the conversation based on keywords
    conversation_nature = "General conversation"
    if user_keywords:
        main_topic = user_keywords[0][0]
        conversation_nature = f"Discussion focused on {main_topic}"
    
    # Generate summary
    summary = f"""Summary:
    - The conversation had {total_exchanges} exchanges.
    - User sent {user_count} messages, AI responded with {ai_count} messages.
    - The nature of the conversation: {conversation_nature}.
    - Most common keywords from user: {user_keywords_str}.
    - Most common keywords overall: {all_keywords_str}.
    """
    return summary

def analyze_chat_log(file_path: str) -> str:
    """
    Main function to analyze a chat log file.
    
    Args:
        file_path: Path to the chat log file
        
    Returns:
        A summary of the chat log
    """
    # Parse the chat log
    user_messages, ai_messages = parse_chat_log(file_path)
    
    if not user_messages and not ai_messages:
        return "No messages found or error reading file."
    
    # Generate and return the summary
    return generate_summary(user_messages, ai_messages)

def main():
    """
    Main entry point for the script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze AI chat logs.')
    parser.add_argument('file_path', help='Path to the chat log file')
    
    args = parser.parse_args()
    
    summary = analyze_chat_log(args.file_path)
    print(summary)

if __name__ == "__main__":
    main()