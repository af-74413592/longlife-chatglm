import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
import gradio as gr
import mdtex2html
import os
from datetime import datetime
import json
import multiprocessing
from finetune import Lora_finetune


parser = ArgumentParser()

parser.add_argument('--init',type=str,default="THUDM/chatglm2-6b")
parser.add_argument('--last',type=str,default="chatglm2-update")
parser.add_argument('--max_lifes',type=int,default=3)
args = parser.parse_args()

lifes = 0
dialogs = []
flag = 0
tokenizer = AutoTokenizer.from_pretrained(args.init, trust_remote_code=True)
model = AutoModel.from_pretrained(args.init, trust_remote_code=True).cuda()

model = model.eval()

#some code copy from https://github.com/THUDM/ChatGLM-6B

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def save(input,response):
    global lifes
    global dialogs
    global model
    global tokenizer
    global flag
    lifes += 1
    dialogs.append((input,response))
    if lifes == args.max_lifes + 1:
        dialogs = dialogs[:-1]
        time = make_dataset(dialogs)
        if not flag:
            lora = Lora_finetune(args.init,time)
        else:
            lora = Lora_finetune(args.last,time)
        flag += 1
        del model
        sub_process = multiprocessing.Process(target=lora.train())
        sub_process.start()
        sub_process.join()
        lifes = 0
        dialogs = []
        tokenizer = AutoTokenizer.from_pretrained(args.last, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.last, trust_remote_code=True).cuda()
        model.eval()
        

def get_time():
    current_datetime = datetime.now()

    date_format = "%Y-%m-%d-%H-%M-%S"  # 指定日期格式
    formatted_datetime = current_datetime.strftime(date_format)

    return formatted_datetime

def make_dataset(dialogs):
    time = get_time()
    with open(os.path.join('datas',time+'.json'),'w',encoding='utf-8') as f:
        data_list = []
        new_history = []
        for i,(input,response) in enumerate(dialogs):
            data_dict = {}
            data_dict['key'] = input
            data_dict['value'] = response
            data_dict['history'] = new_history.copy()
            new_history.append((data_dict['key'],data_dict['value']))
            data_list.append(data_dict)
        print(data_list)
        f.write(json.dumps(data_list,ensure_ascii=False))
        return time

def make_dataset_SWAP(dialogs):
    time = get_time()
    with open(os.path.join('datas',time+'.json'),'w',encoding='utf-8') as f:
        data_list = []
        new_history = []
        for i,(input,response) in enumerate(dialogs):
            data_dict = {}
            if i == 0:
                data_dict['key'] = "AI正在等待用户提问。"
                data_dict['value'] = "用户："+ input
                data_dict['history'] = []
                new_history.append((data_dict['key'],data_dict['value']))
                data_list.append(data_dict)
                temp_response = response
            elif i != len(dialogs)-1:
                data_dict['key'] = "AI："+temp_response
                data_dict['value'] = "用户："+input
                data_dict['history'] = new_history.copy()
                new_history.append((data_dict['key'],data_dict['value']))
                data_list.append(data_dict)
                temp_response = response
            else:
                data_dict['key'] = "AI："+temp_response
                data_dict['value'] = "用户："+input
                data_dict['history'] = new_history.copy()
                new_history.append((data_dict['key'],data_dict['value']))
                data_list.append(data_dict)
                data_dict_final = {}
                data_dict_final['key'] = "AI："+response
                data_dict_final['value'] = "对话已终止。"
                data_dict_final['history'] = new_history.copy()
                data_list.append(data_dict_final)
        print(data_list)
        f.write(json.dumps(data_list,ensure_ascii=False))
        return time

def predict(input, chatbot, max_length, top_p, temperature, history):
    global lifes
    global model
    global tokenizer
    chatbot.append((parse_text(input), ""))
    if lifes >= args.max_lifes:
        chatbot[-1] = (parse_text(input), parse_text('我困了，想要休息一会。'))
        history = []
        yield chatbot, history
        save(input,'我困了，想要休息一会。')
    else:
        print(model)
        for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                                temperature=temperature):
            chatbot[-1] = (parse_text(input), parse_text(response))       
            yield chatbot, history
        save(input,response)
    
def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">LongLife ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
