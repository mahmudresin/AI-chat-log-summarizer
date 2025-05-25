# Enhanced AI Chat Log Analyzer

A comprehensive Python-based tool that analyzes chat logs between users and AI assistants, featuring advanced NLP capabilities, sentiment analysis, interactive GUI, and data visualizations.

## ✨ Features

- **Advanced NLP Processing**: Powered by NLTK with lemmatization, POS tagging, and named entity recognition
- **Sentiment Analysis**: VADER sentiment analysis for emotional tone detection
- **Interactive GUI**: User-friendly graphical interface with multiple tabs and visualizations
- **Data Visualizations**: Charts and graphs showing conversation patterns, sentiment distribution, and keyword analysis
- **Batch Processing**: Analyze multiple chat logs from a directory
- **Export Capabilities**: Save summaries and visualizations
- **Command-line Interface**: Alternative CLI for automation and scripting

## 🔧 Requirements

- Python 3.7 or higher
- Required packages (install via `pip install -r requirements.txt`):
  - nltk>=3.8
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - pandas>=1.3.0
  - numpy>=1.21.0

## 📦 Installation

1. Clone or download this repository:
```bash
git clone https://github.com/mahmudresin/enhanced-chat-analyzer.git
cd enhanced-chat-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The first run will automatically download required NLTK data.

## 🚀 Usage

### GUI Application (Recommended)

Launch the interactive GUI:
```bash
python enhanced_chat_analyzer.py
```

Features of the GUI:
- **Load Chat Log**: Analyze a single chat file
- **Load Directory**: Batch process multiple files
- **Generate Visualizations**: Create charts and graphs
- **Export Summary**: Save analysis results

### Command-line Interface

For automation and scripting:

```bash
# Analyze a single file
python cli_analyzer.py chat_log.txt

# Analyze multiple files in a directory
python cli_analyzer.py /path/to/chat_logs/

# Generate visualizations
python cli_analyzer.py chat_log.txt --visualize

# Save summary and visualizations
python cli_analyzer.py chat_log.txt -o summary.txt -v --viz-output charts.png
```

## 📊 Analysis Features

### 1. **Comprehensive Statistics**
- Message counts for each participant
- Average message lengths
- Question detection
- Conversation flow analysis

### 2. **Advanced Keyword Extraction**
- NLTK-powered token# AI Chat Log Summarizer

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
git clone https://github.com/mahmudresin/ai-chat-log-summarizer.git
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

MIT License