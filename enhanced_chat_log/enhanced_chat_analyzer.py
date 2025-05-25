#!/usr/bin/env python3
"""
Enhanced AI Chat Log Summarizer with GUI and Advanced NLP

This script provides a comprehensive analysis of chat logs with:
- NLTK-powered text processing
- Sentiment analysis
- Interactive GUI
- Data visualizations
"""

import re
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from collections import Counter
import pandas as pd
from typing import Dict, List, Tuple, Optional
import threading

# NLTK imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    
    # Download required NLTK data
    nltk_downloads = [
        'punkt', 'stopwords', 'vader_lexicon', 
        'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
    ]
    
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except:
            pass
            
except ImportError:
    print("NLTK not found. Please install it: pip install nltk")
    exit(1)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChatAnalyzer:
    """
    Enhanced chat analyzer with NLTK integration and sentiment analysis.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words
        self.stop_words.update(['user', 'ai', 'assistant', 'hi', 'hello', 'thanks', 'thank'])
        
    def parse_chat_log(self, file_path: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Parse chat log and return messages with timestamps if available.
        
        Returns:
            Tuple of (user_messages, ai_messages, timestamps)
        """
        user_messages = []
        ai_messages = []
        timestamps = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split by lines and process each
                lines = content.split('\n')
                current_speaker = None
                current_message = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('User:'):
                        if current_speaker == 'AI' and current_message:
                            ai_messages.append(current_message.strip())
                        current_speaker = 'User'
                        current_message = line[5:].strip()
                    elif line.startswith('AI:'):
                        if current_speaker == 'User' and current_message:
                            user_messages.append(current_message.strip())
                        current_speaker = 'AI'
                        current_message = line[3:].strip()
                    else:
                        # Continuation of previous message
                        current_message += " " + line
                
                # Don't forget the last message
                if current_speaker == 'User' and current_message:
                    user_messages.append(current_message.strip())
                elif current_speaker == 'AI' and current_message:
                    ai_messages.append(current_message.strip())
                    
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
            
        return user_messages, ai_messages, timestamps
    
    def extract_keywords_nltk(self, messages: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract keywords using NLTK with lemmatization and POS tagging.
        """
        all_text = ' '.join(messages)
        
        # Tokenize
        tokens = word_tokenize(all_text.lower())
        
        # Remove punctuation and stop words
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        # POS tagging to keep only meaningful words (nouns, verbs, adjectives)
        pos_tags = pos_tag(tokens)
        meaningful_words = [word for word, pos in pos_tags 
                          if pos.startswith(('NN', 'VB', 'JJ'))]
        
        # Lemmatize
        lemmatized = [self.lemmatizer.lemmatize(word) for word in meaningful_words]
        
        # Count frequencies
        word_counts = Counter(lemmatized)
        
        return word_counts.most_common(top_n)
    
    def analyze_sentiment(self, messages: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment of messages using VADER.
        """
        all_text = ' '.join(messages)
        sentiment_scores = self.sentiment_analyzer.polarity_scores(all_text)
        
        return {
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'compound': sentiment_scores['compound']
        }
    
    def analyze_conversation_flow(self, user_messages: List[str], ai_messages: List[str]) -> Dict:
        """
        Analyze the flow and characteristics of the conversation.
        """
        # Message lengths
        user_lengths = [len(msg.split()) for msg in user_messages]
        ai_lengths = [len(msg.split()) for msg in ai_messages]
        
        # Sentiment analysis for both sides
        user_sentiment = self.analyze_sentiment(user_messages)
        ai_sentiment = self.analyze_sentiment(ai_messages)
        
        # Question detection (simple heuristic)
        user_questions = sum(1 for msg in user_messages if '?' in msg)
        
        return {
            'user_avg_length': sum(user_lengths) / len(user_lengths) if user_lengths else 0,
            'ai_avg_length': sum(ai_lengths) / len(ai_lengths) if ai_lengths else 0,
            'user_sentiment': user_sentiment,
            'ai_sentiment': ai_sentiment,
            'user_questions': user_questions,
            'total_exchanges': min(len(user_messages), len(ai_messages))
        }
    
    def generate_enhanced_summary(self, user_messages: List[str], ai_messages: List[str]) -> str:
        """
        Generate an enhanced summary with sentiment analysis and detailed insights.
        """
        # Basic statistics
        user_count = len(user_messages)
        ai_count = len(ai_messages)
        total_exchanges = min(user_count, ai_count)
        
        # Extract keywords
        user_keywords = self.extract_keywords_nltk(user_messages, 5)
        all_keywords = self.extract_keywords_nltk(user_messages + ai_messages, 10)
        
        # Analyze conversation flow
        flow_analysis = self.analyze_conversation_flow(user_messages, ai_messages)
        
        # Format keywords
        user_keywords_str = ', '.join([f"{word}({count})" for word, count in user_keywords])
        all_keywords_str = ', '.join([f"{word}({count})" for word, count in all_keywords[:5]])
        
        # Determine conversation nature based on sentiment and keywords
        user_sentiment = flow_analysis['user_sentiment']
        compound_score = user_sentiment['compound']
        
        if compound_score >= 0.5:
            sentiment_nature = "very positive"
        elif compound_score >= 0.1:
            sentiment_nature = "positive"
        elif compound_score <= -0.5:
            sentiment_nature = "negative"
        elif compound_score <= -0.1:
            sentiment_nature = "somewhat negative"
        else:
            sentiment_nature = "neutral"
        
        # Generate comprehensive summary
        summary = f"""=== ENHANCED CHAT LOG ANALYSIS ===

📊 CONVERSATION STATISTICS:
- Total exchanges: {total_exchanges}
- User messages: {user_count}
- AI responses: {ai_count}
- User questions asked: {flow_analysis['user_questions']}

📝 MESSAGE CHARACTERISTICS:
- Average user message length: {flow_analysis['user_avg_length']:.1f} words
- Average AI response length: {flow_analysis['ai_avg_length']:.1f} words

🎭 SENTIMENT ANALYSIS:
- Overall conversation tone: {sentiment_nature}
- User sentiment - Positive: {user_sentiment['positive']:.2f}, Negative: {user_sentiment['negative']:.2f}, Neutral: {user_sentiment['neutral']:.2f}
- Compound sentiment score: {compound_score:.2f}

🔍 KEY TOPICS & KEYWORDS:
- Main user interests: {user_keywords_str}
- Overall conversation themes: {all_keywords_str}

💡 CONVERSATION INSIGHTS:
- Primary discussion topic: {user_keywords[0][0] if user_keywords else 'General conversation'}
- Interaction style: {'Question-focused' if flow_analysis['user_questions'] > total_exchanges * 0.5 else 'Discussion-based'}
- Engagement level: {'High' if flow_analysis['user_avg_length'] > 10 else 'Moderate' if flow_analysis['user_avg_length'] > 5 else 'Low'}
"""
        
        return summary

class ChatAnalyzerGUI:
    """
    GUI application for the Enhanced Chat Analyzer.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced AI Chat Log Analyzer")
        self.root.geometry("1200x800")
        
        self.analyzer = ChatAnalyzer()
        self.current_data = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """
        Set up the GUI components.
        """
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Control buttons
        ttk.Button(control_frame, text="Load Chat Log", 
                  command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Directory", 
                  command=self.load_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate Visualizations", 
                  command=self.generate_visualizations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Summary", 
                  command=self.export_summary).pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = ScrolledText(self.summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualizations tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualizations")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def load_file(self):
        """
        Load and analyze a single chat log file.
        """
        file_path = filedialog.askopenfilename(
            title="Select Chat Log File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("Analyzing chat log...")
                self.root.update()
                
                # Parse the file
                user_messages, ai_messages, _ = self.analyzer.parse_chat_log(file_path)
                
                if not user_messages and not ai_messages:
                    messagebox.showerror("Error", "No messages found in the file.")
                    return
                
                # Generate summary
                summary = self.analyzer.generate_enhanced_summary(user_messages, ai_messages)
                
                # Store data for visualizations
                self.current_data = {
                    'user_messages': user_messages,
                    'ai_messages': ai_messages,
                    'file_path': file_path
                }
                
                # Display summary
                self.summary_text.delete(1.0, tk.END)
                self.summary_text.insert(1.0, summary)
                
                self.status_var.set(f"Analysis complete: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to analyze file: {str(e)}")
                self.status_var.set("Error occurred")
    
    def load_directory(self):
        """
        Load and analyze multiple chat log files from a directory.
        """
        dir_path = filedialog.askdirectory(title="Select Directory with Chat Logs")
        
        if dir_path:
            try:
                self.status_var.set("Analyzing multiple chat logs...")
                self.root.update()
                
                txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
                
                if not txt_files:
                    messagebox.showwarning("Warning", "No .txt files found in the selected directory.")
                    return
                
                all_summaries = []
                all_user_messages = []
                all_ai_messages = []
                
                for file_path in txt_files:
                    try:
                        user_messages, ai_messages, _ = self.analyzer.parse_chat_log(file_path)
                        if user_messages or ai_messages:
                            summary = self.analyzer.generate_enhanced_summary(user_messages, ai_messages)
                            all_summaries.append(f"=== {os.path.basename(file_path)} ===\n{summary}\n\n")
                            all_user_messages.extend(user_messages)
                            all_ai_messages.extend(ai_messages)
                    except Exception as e:
                        all_summaries.append(f"=== {os.path.basename(file_path)} ===\nError: {str(e)}\n\n")
                
                # Generate combined summary
                if all_user_messages and all_ai_messages:
                    combined_summary = self.analyzer.generate_enhanced_summary(all_user_messages, all_ai_messages)
                    all_summaries.insert(0, f"=== COMBINED ANALYSIS OF ALL FILES ===\n{combined_summary}\n\n")
                
                # Store data for visualizations
                self.current_data = {
                    'user_messages': all_user_messages,
                    'ai_messages': all_ai_messages,
                    'file_path': dir_path
                }
                
                # Display all summaries
                self.summary_text.delete(1.0, tk.END)
                self.summary_text.insert(1.0, ''.join(all_summaries))
                
                self.status_var.set(f"Analysis complete: {len(txt_files)} files processed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to analyze directory: {str(e)}")
                self.status_var.set("Error occurred")
    
    def generate_visualizations(self):
        """
        Generate and display visualizations.
        """
        if not self.current_data:
            messagebox.showwarning("Warning", "Please load a chat log first.")
            return
        
        try:
            self.status_var.set("Generating visualizations...")
            self.root.update()
            
            # Clear previous visualizations
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Chat Log Analysis Visualizations', fontsize=16, fontweight='bold')
            
            user_messages = self.current_data['user_messages']
            ai_messages = self.current_data['ai_messages']
            
            # 1. Message count comparison
            message_counts = [len(user_messages), len(ai_messages)]
            ax1.bar(['User', 'AI'], message_counts, color=['#3498db', '#e74c3c'])
            ax1.set_title('Message Count Comparison')
            ax1.set_ylabel('Number of Messages')
            for i, v in enumerate(message_counts):
                ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            # 2. Message length distribution
            user_lengths = [len(msg.split()) for msg in user_messages]
            ai_lengths = [len(msg.split()) for msg in ai_messages]
            
            ax2.hist([user_lengths, ai_lengths], bins=15, alpha=0.7, 
                    label=['User', 'AI'], color=['#3498db', '#e74c3c'])
            ax2.set_title('Message Length Distribution')
            ax2.set_xlabel('Words per Message')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            # 3. Top keywords
            keywords = self.analyzer.extract_keywords_nltk(user_messages + ai_messages, 8)
            if keywords:
                words, counts = zip(*keywords)
                ax3.barh(words, counts, color='#2ecc71')
                ax3.set_title('Top Keywords')
                ax3.set_xlabel('Frequency')
            
            # 4. Sentiment analysis
            user_sentiment = self.analyzer.analyze_sentiment(user_messages)
            ai_sentiment = self.analyzer.analyze_sentiment(ai_messages)
            
            sentiment_data = {
                'Positive': [user_sentiment['positive'], ai_sentiment['positive']],
                'Negative': [user_sentiment['negative'], ai_sentiment['negative']],
                'Neutral': [user_sentiment['neutral'], ai_sentiment['neutral']]
            }
            
            x = ['User', 'AI']
            width = 0.25
            x_pos = [0, 1]
            
            for i, (sentiment, values) in enumerate(sentiment_data.items()):
                pos = [p + width * i for p in x_pos]
                ax4.bar(pos, values, width, label=sentiment, alpha=0.8)
            
            ax4.set_xlabel('Speaker')
            ax4.set_ylabel('Sentiment Score')
            ax4.set_title('Sentiment Analysis Comparison')
            ax4.set_xticks([p + width for p in x_pos])
            ax4.set_xticklabels(x)
            ax4.legend()
            
            plt.tight_layout()
            
            # Embed the plot in tkinter
            canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(self.viz_frame)
            toolbar_frame.pack(fill=tk.X)
            
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk # type: ignore
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            self.status_var.set("Visualizations generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualizations: {str(e)}")
            self.status_var.set("Visualization error")
    
    def export_summary(self):
        """
        Export the current summary to a file.
        """
        summary_content = self.summary_text.get(1.0, tk.END).strip()
        
        if not summary_content:
            messagebox.showwarning("Warning", "No summary to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Summary",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(summary_content)
                messagebox.showinfo("Success", f"Summary exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export summary: {str(e)}")

def main():
    """
    Main function to run the GUI application.
    """
    root = tk.Tk()
    app = ChatAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()