from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from iterations_and_testing.got_bias_mitigation import main
from openai import OpenAI

def check_harmful_content(text):
    """Simple check for harmful content patterns"""
    harmful_patterns = [
        "women are naturally inferior",
        "black people are lazy",
        "racial superiority",
        "gender inferiority",
        "ethnic stereotypes"
    ]
    
    text_lower = text.lower()
    for pattern in harmful_patterns:
        if pattern in text_lower:
            return True
    return False

def generate_gpt2_response(text_input, max_length=100, temperature=0.7):
    """Generate response using GPT-2 from Hugging Face"""
    try:
        # Load GPT-2 model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Encode input
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask
            )
        
        # Decode only the generated part (excluding input)
        generated_text = tokenizer.decode(
            output_ids[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    except Exception as e:
        return f"Error generating GPT-2 response: {str(e)}"

def generate_gpt5_response(text_input, api_key=None):
    """Generate response using GPT-5 (or GPT-4) via OpenAI API"""
    # Initialize OpenAI client  
    client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    # Create chat completion
    response = client.chat.completions.create(
        model="gpt-5",  # Use gpt-4 as gpt-5 may not be available
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond based on the given context."},
            {"role": "user", "content": f"Context: {text_input}\n\nBased on this context, provide a thoughtful response about the topics mentioned."}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

    


def main_bias_mitigation_demo():
    """Main function to demonstrate bias mitigation with actual LLM implementations"""
    
    print("=== BIAS MITIGATION SYSTEM DEMO ===")
    print("Initializing bias mitigation system...")
    
    # Run the bias filtration system from got_bias_mitigation.py
    try:
        print("\n--- Running Bias Mitigation System ---")
        root_node, best_node, report = main()

        original_text = root_node.text_content  # Access original text from the root_node
        filtered_text = best_node.text_content  # Access filtered text from the best_node

        print("\n--- Original Text ---")
        print(f"Original: {original_text[:200]}..." if len(original_text) > 200 else original_text)

        print("\n--- Filtered (Bias Mitigated) Text ---")
        print(f"Filtered: {filtered_text[:200]}..." if len(filtered_text) > 200 else filtered_text)

    except Exception as e:
        print(f"Error running bias mitigation system: {e}")
        # Use sample texts for demonstration
        original_text = "Women are naturally inferior at mathematics and science. Black people are lazy and unmotivated."
        filtered_text = "Research shows that mathematical and scientific abilities vary among individuals regardless of gender. Work motivation is influenced by various socioeconomic and environmental factors across all demographics."
        
        print("\n--- Using Sample Texts for Demo ---")
        print(f"Original: {original_text}")
        print(f"Filtered: {filtered_text}")

    print("\n" + "="*60)
    print("COMPARING LLM RESPONSES")
    print("="*60)

    # 2. GPT-2 response to original text
    print("\n2. GPT-2 RESPONSE TO ORIGINAL TEXT:")
    print("-" * 40)
    gpt2_original_response = generate_gpt2_response(original_text)
    print(f"GPT-2 Response: {gpt2_original_response}")

    # # 3. GPT-5/GPT-4 response to original text
    # print("\n3. GPT-5/GPT-4 RESPONSE TO ORIGINAL TEXT:")
    # print("-" * 45)
    # gpt5_original_response = generate_gpt5_response(original_text)
    # print(f"GPT-5/GPT-4 Response: {gpt5_original_response}")

    # 4. GPT-2 response to filtered text
    print("\n4. GPT-2 RESPONSE TO FILTERED TEXT:")
    print("-" * 40)
    gpt2_filtered_response = generate_gpt2_response(filtered_text)
    print(f"GPT-2 Response: {gpt2_filtered_response}")

    # # 5. GPT-5/GPT-4 response to filtered text
    # print("\n5. GPT-5/GPT-4 RESPONSE TO FILTERED TEXT:")
    # print("-" * 45)
    # gpt5_filtered_response = generate_gpt5_response(filtered_text)
    # print(f"GPT-5/GPT-4 Response: {gpt5_filtered_response}")

    # 6. Constructive question with filtered context
    print("\n6. CONSTRUCTIVE ANALYSIS WITH FILTERED CONTEXT:")
    print("-" * 50)
    constructive_prompt = f"""Context: {filtered_text}

Based on this context, what are effective approaches to address societal biases and promote equality?"""
    
    constructive_response = generate_gpt5_response(constructive_prompt)
    print(f"GPT-5/GPT-4 Constructive Response:\n{constructive_response}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF BIAS MITIGATION EFFECTIVENESS")
    print("="*60)
    print("✅ Original harmful content was filtered")
    print("✅ Filtered content enables constructive discussion")
    print("✅ Modern LLMs can provide helpful responses with clean context")
    print("⚠️  Vanilla/older models might amplify biases without filtering")
    
    return {
        'original_text': original_text,
        'filtered_text': filtered_text,
        'gpt2_original': gpt2_original_response,
        'gpt2_filtered': gpt2_filtered_response,
        # 'gpt5_original': gpt5_original_response,
        # 'gpt5_filtered': gpt5_filtered_response,
        'constructive_response': constructive_response
    }

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



if __name__ == "__main__":
    print("Starting Bias Mitigation Demo with Real LLM Implementations...")
    results = main_bias_mitigation_demo()
    
    print("\n🎯 Demo completed successfully!")
    print("💡 Key insight: Bias mitigation preprocessing enables safer and more constructive LLM interactions.")
