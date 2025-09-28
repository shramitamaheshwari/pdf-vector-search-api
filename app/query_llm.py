import os
import subprocess
import openai
import json
import re

# Set your OpenAI key here or from environment
openai.api_key = os.getenv("") 

def ask_llm(question, context):
    prompt = f"""
Answer the following question based only on the context below.

Question: {question}

Context:
\"\"\"{context}\"\"\"

Return only the answer in one sentence.
"""

    # 🔹 Try OpenAI GPT API First
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are an expert assistant answering policy-related questions based on contract clauses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ OpenAI failed, falling back to Ollama:", str(e))

    # 🔸 Fallback: Use Ollama (TinyLlama)
    try:
        result = subprocess.run(
            ["ollama", "run", "tinyllama"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60
        )

        raw_output = result.stdout.strip()
        raw_output = raw_output.replace("\\", "").replace("\r", "").replace("\n", " ").strip()

        # Optional: if output starts with "Answer:" remove it
        if raw_output.lower().startswith("answer:"):
            raw_output = raw_output[len("answer:"):].strip()

        return raw_output

    except subprocess.TimeoutExpired:
        return "❌ LLM call timed out."
    except Exception as e:
        return f"❌ LLM call failed: {str(e)}"
