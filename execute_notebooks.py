import nbformat
from nbclient import NotebookClient
import sys

sys.stdout.reconfigure(encoding='utf-8')

def run_notebook(nb_path):
    print(f"[*] Executing notebook: {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    client = NotebookClient(nb, timeout=300, kernel_name='python3')
    client.execute()
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"[OK] Successfully executed and saved outputs to: {nb_path}")

if __name__ == "__main__":
    try:
        run_notebook("task1_web_scraping_sentiment_analysis.ipynb")
    except Exception as e:
        print("[!] Error in Task 1 execution:", e)
        
    try:
        run_notebook("task2_advanced_eda_text_mining.ipynb")
    except Exception as e:
        print("[!] Error in Task 2 execution:", e)
