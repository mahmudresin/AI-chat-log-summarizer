#!/usr/bin/env python3
"""
Command-line version of the Enhanced Chat Analyzer
"""

import argparse
import os
import sys
from enhanced_chat_analyzer import ChatAnalyzer
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Enhanced AI Chat Log Analyzer (CLI)')
    parser.add_argument('path', help='Path to chat log file or directory')
    parser.add_argument('--output', '-o', help='Output file for summary')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Generate and save visualizations')
    parser.add_argument('--viz-output', default='chat_analysis.png',
                       help='Output file for visualizations (default: chat_analysis.png)')
    
    args = parser.parse_args()
    
    analyzer = ChatAnalyzer()
    
    try:
        if os.path.isfile(args.path):
            # Single file
            user_messages, ai_messages, _ = analyzer.parse_chat_log(args.path)
            summary = analyzer.generate_enhanced_summary(user_messages, ai_messages)
            print(summary)
            
        elif os.path.isdir(args.path):
            # Directory
            import glob
            txt_files = glob.glob(os.path.join(args.path, "*.txt"))
            
            all_user_messages = []
            all_ai_messages = []
            
            for file_path in txt_files:
                try:
                    user_messages, ai_messages, _ = analyzer.parse_chat_log(file_path)
                    if user_messages or ai_messages:
                        print(f"\n=== {os.path.basename(file_path)} ===")
                        summary = analyzer.generate_enhanced_summary(user_messages, ai_messages)
                        print(summary)
                        
                        all_user_messages.extend(user_messages)
                        all_ai_messages.extend(ai_messages)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            if all_user_messages and all_ai_messages:
                print("\n=== COMBINED ANALYSIS ===")
                combined_summary = analyzer.generate_enhanced_summary(all_user_messages, all_ai_messages)
                print(combined_summary)
        else:
            print(f"Error: {args.path} is not a valid file or directory")
            sys.exit(1)
            
        # Generate visualizations if requested
        if args.visualize:
            print(f"\nGenerating visualizations to {args.viz_output}...")
            
            # Create visualizations (simplified for CLI)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Chat Log Analysis', fontsize=16, fontweight='bold')
            
            # Use the last analyzed data
            if 'all_user_messages' in locals():
                user_msgs = all_user_messages
                ai_msgs = all_ai_messages
            else:
                user_msgs = user_messages
                ai_msgs = ai_messages
            
            # Message counts
            message_counts = [len(user_msgs), len(ai_msgs)]
            ax1.bar(['User', 'AI'], message_counts, color=['#3498db', '#e74c3c'])
            ax1.set_title('Message Count')
            
            # Message lengths
            user_lengths = [len(msg.split()) for msg in user_msgs]
            ai_lengths = [len(msg.split()) for msg in ai_msgs]
            ax2.hist([user_lengths, ai_lengths], bins=10, alpha=0.7, 
                    label=['User', 'AI'], color=['#3498db', '#e74c3c'])
            ax2.set_title('Message Length Distribution')
            ax2.legend()
            
            # Keywords
            keywords = analyzer.extract_keywords_nltk(user_msgs + ai_msgs, 8)
            if keywords:
                words, counts = zip(*keywords)
                ax3.barh(words, counts, color='#2ecc71')
                ax3.set_title('Top Keywords')
            
            # Sentiment
            user_sentiment = analyzer.analyze_sentiment(user_msgs)
            sentiment_scores = [user_sentiment['positive'], user_sentiment['negative'], user_sentiment['neutral']]
            ax4.pie(sentiment_scores, labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%')
            ax4.set_title('Sentiment Distribution')
            
            plt.tight_layout()
            plt.savefig(args.viz_output, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {args.viz_output}")
        
        # Save summary if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary if 'summary' in locals() else combined_summary)
            print(f"\nSummary saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()