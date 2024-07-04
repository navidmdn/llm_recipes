import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, TextIteratorStreamer
from fire import Fire
from typing import List, Tuple


def load_test_llama3_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_size = 256
    config.intermediate_size = 16
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    model = LlamaForCausalLM(config=config)
    return model


system_prompt = "You are a helpful and friendly chatbot"

#todo: add stremer to generation
def run(model_name_or_path='Qwen/Qwen1.5-0.5B-Chat', local_test=False, max_new_tokens=512,
        temperature=0.1, top_p=0.9, num_return_sequences=1, do_sample=True, num_beams=1):

    # Load model and tokenizer
    if local_test:
        model = load_test_llama3_model(model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def build_input_from_chat_history(chat_history: List[Tuple], msg: str):
        messages = [{'role': 'system', 'content': system_prompt}]
        for user_msg, ai_msg in chat_history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        messages.append({'role': 'user', 'content': msg})
        return messages

    # Define the chat function
    def chat(message, chat_history):

        messages = build_input_from_chat_history(chat_history, message)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True
        )

        output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                                do_sample=do_sample, temperature=temperature, top_p=top_p,
                                num_return_sequences=num_return_sequences, num_beams=num_beams)

        response_token_ids = output[0][input_ids.shape[1]:]
        response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
        chat_history.append((message, response))

        yield "", chat_history


    # Create the Gradio interface
    css = """
    #custom-chatbot .user { background-color: #A0E7E5; color: #000000; }
    #custom-chatbot .assistant { background-color: #B4F8C8; color: #000000; }
    """
    iface = gr.Blocks()#css=css)

    with iface:
        gr.Markdown("# Let's Chat!")
        chatbox = gr.Chatbot([], layout="bubble", bubble_full_width=False) #elem_id="custom-chatbot")
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbox], [msg, chatbox])
        clear.click(lambda: None, None, chatbox, queue=False)

    # Launch the app
    iface.launch()


if __name__ == '__main__':
    Fire(run)