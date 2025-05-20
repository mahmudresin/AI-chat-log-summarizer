# AI Chat Log Summarizer

A Python-based tool that reads chat logs between a user and an AI, parses the conversation, and produces a simple summary including message counts and frequently used keywords.

## Features

- **Chat Log Parsing**: Separates messages by speaker (User and AI)
- **Message Statistics**: Counts total messages and messages per participant
- **Keyword Analysis**: Extracts the top 5 most frequently used words (excluding common stop words)
- **Advanced Keyword Extraction**: Optional TF-IDF approach for better keyword identification
- **Batch Processing**: Ability to process multiple chat logs from a directory

## Requirements

- Python 3.6 or higher
- No external libraries required (uses only standard library)

## Installation

No installation needed. Simply download the script and run it.

```bash
# Clone or download this repository
git clone https://github.com/yourusername/ai-chat-log-summarizer.git
cd ai-chat-log-summarizer

# Make the script executable (Linux/Mac)
chmod +x chat_log_summarizer.py
```

## Usage

### Basic Usage

Process a single chat log file:

```bash
python chat_log_summarizer.py path/to/chat_log.txt
```

### Advanced Options

Use TF-IDF for keyword extraction:

```bash
python chat_log_summarizer.py path/to/chat_log.txt --tfidf
```

Process all .txt files in a directory:

```bash
python chat_log_summarizer.py path/to/chat_logs_directory/
```

Save the summary to a file:

```bash
python chat_log_summarizer.py path/to/chat_log.txt -o summary.txt
```

Combine all options:

```bash
python chat_log_summarizer.py path/to/chat_logs_directory/ --tfidf -o summary.txt
```

## Example Output

```
Summary:
- The conversation had 6 exchanges.
- User sent 6 messages, AI responded with 6 messages.
- The nature of the conversation: Discussion focused on machine.
- Most common keywords from user: machine, learning, resources, applications, started.
- Most common keywords overall: machine, learning, resources, applications, started.
```

## Chat Log Format

The script expects chat logs to be in the following format:

```
User: Message from the user
AI: Response from the AI assistant
User: Another message from the user
AI: Another response from the AI assistant
```

## License

Apache License 2.0
