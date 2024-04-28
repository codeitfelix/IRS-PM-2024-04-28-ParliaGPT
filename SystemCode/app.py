import time
import gradio as gr

from app.retriever import FusedRetriever
from app.llm_generate import LLMInference

llm = LLMInference()

def user_inference(query:str):
    retrieved = FusedRetriever(query, 8)
    extracts = llm.get_extracts(query, retrieved.ce_list)
    images = llm.get_response_images(extracts)
    response = llm.generate_response(query, extracts)
    
    return images, response

def text_streamer(response):
    for i in range(len(response)):
        time.sleep(0.005)
        yield response[:i+1]


print('Loading UI...')
with gr.Blocks() as demo:
    gr.Markdown('# ParliaGPT')
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label='User Query', placeholder='*Type in your query here*', lines=3, show_copy_button=True)
            with gr.Row():
                gen = gr.Button('Get Response')
                clear = gr.Button('Clear')
            # out_text = gr.Textbox(visible=False)
            out_stream = gr.Textbox(label='Bot Response', placeholder='', lines=8, show_copy_button=True)
            
        out_img = gr.Gallery(label='Retrieved Contexts', preview=True, columns=2)
    
    gen.click(fn=user_inference, inputs=inp, outputs=[out_img, out_stream]).success(fn=text_streamer, inputs=out_stream, outputs=out_stream)
    # gen.click(fn=user_inference, inputs=inp, outputs=[out_img, out_text]).success(fn=text_streamer, inputs=out_text, outputs=out_stream)
    clear.click(lambda: [None, None], outputs=[out_img, out_stream])

demo.launch(server_port=7868)
