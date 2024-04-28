import os
from tqdm import tqdm
from threading import Thread

import torch
import duckdb
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig

from constants.embed import LLM_MODEL
from constants.directories import RELATIONAL_DIR, IMG_DIR


class LLMInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.__initialize_model()
    
    
    def __initialize_model(self):    
        model_name_or_path = LLM_MODEL
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     torch_dtype=torch_dtype,
                                                     device_map=device,
                                                     quantization_config=gptq_config,
                                                     trust_remote_code=False, 
                                                     revision="main"
                                                     )
        
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    
    
    def get_extracts(self, query, hyb_list:list[tuple[str, float]]):
        connection = duckdb.connect(os.path.join(RELATIONAL_DIR, 'data.db'))
        extracts = []
        for (id, _) in tqdm(hyb_list):
            text = connection.sql(f"SELECT text FROM parliament_debate WHERE id = '{id}'").fetchone()[0].replace('\n\n( )\n\n', ' ')
            system_message = 'You are a helpful and intelligent assistant that extracts relevant information from passages.'

            prompt = f'''Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question.
            If none of the context is relevant, strictly return NO_OUTPUT.
            Remember, *DO NOT* edit the extracted parts of the context.

            Here are some examples for reference:

            ##########

            question = "What are the benefits of eating fruits?"

            context = "Apples contain key nutrients, including fiber and antioxidants.
            They may offer health benefits, including lowering blood sugar levels and benefitting heart health.
            Apples are among the world's most popular fruits.
            They grow on the apple tree (Malus domestica), originally from Central Asia.
            Apples are high in fiber, vitamin C, and various antioxidants. They are also very filling, considering their low calorie count.
            Studies show that eating apples can have multiple benefits for your health.
            Usually eaten raw, apples can also be used in various recipes, juices, and drinks. Various types abound, with a variety of colors and sizes."

            answer = "Apples contain key nutrients, including fiber and antioxidants.
            They may offer health benefits, including lowering blood sugar levels and benefitting heart health.
            Studies show that eating apples can have multiple benefits for your health."

            ##########

            question = "Why is it important to eat vegetables?"

            context = "A diet rich in vegetables and fruits can lower blood pressure, reduce the risk of heart disease and stroke, prevent some types of cancer, lower risk of eye and digestive problems, and have a positive effect upon blood sugar, which can help keep appetite in check.
            Eating non-starchy vegetables and fruits like apples, pears, and green leafy vegetables may even promote weight loss.
            Their low glycemic loads prevent blood sugar spikes that can increase hunger.
            At least nine different families of fruits and vegetables exist, each with potentially hundreds of different plant compounds that are beneficial to health.
            Eat a variety of types and colors of produce in order to give your body the mix of nutrients it needs.
            This not only ensures a greater diversity of beneficial plant chemicals but also creates eye-appealing meals."

            answer = "A diet rich in vegetables and fruits can lower blood pressure, reduce the risk of heart disease and stroke, prevent some types of cancer, lower risk of eye and digestive problems, and have a positive effect upon blood sugar, which can help keep appetite in check.
            Eating non-starchy vegetables and fruits like apples, pears, and green leafy vegetables may even promote weight loss.
            Their low glycemic loads prevent blood sugar spikes that can increase hunger.
            At least nine different families of fruits and vegetables exist, each with potentially hundreds of different plant compounds that are beneficial to health."

            #########

            question = "What are the health benefits of exercise?"

            context = "A diet rich in vegetables and fruits can lower blood pressure, reduce the risk of heart disease and stroke, prevent some types of cancer, lower risk of eye and digestive problems, and have a positive effect upon blood sugar, which can help keep appetite in check.
            Eating non-starchy vegetables and fruits like apples, pears, and green leafy vegetables may even promote weight loss.
            Their low glycemic loads prevent blood sugar spikes that can increase hunger.
            At least nine different families of fruits and vegetables exist, each with potentially hundreds of different plant compounds that are beneficial to health.
            Eat a variety of types and colors of produce in order to give your body the mix of nutrients it needs.
            This not only ensures a greater diversity of beneficial plant chemicals but also creates eye-appealing meals."

            answer = "NO_OUTPUT"

            #########

            Answer the following:

            question = {query}

            contexts = {text}

            answer = '''

            prompt_template = f'''<|im_start|>system
            {system_message}<|im_end|>

            <|im_start|>user
            {prompt}<|im_end|>

            <|im_start|>assistant
            '''
        
            tokens = self.tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

            generation_params = {"do_sample": True,
                                 "temperature": 0.7,
                                 "top_p": 0.95,
                                 "top_k": 40,
                                 "max_new_tokens": 2048,
                                 "repetition_penalty": 1.1
                                 }

            # Generation without a streamer, which will include the prompt in the output
            generation_output = self.model.generate(
                tokens,
                # streamer=streamer,
                **generation_params
            )

            # Get the tokens from the output, decode them, print them
            token_output = generation_output[0]
            text_output = self.tokenizer.decode(token_output[tokens.shape[1]:]).partition('<|im_end|>')[0]

            if not 'NO_OUTPUT' in text_output.upper():
                extracts.append((id, text_output.strip()))

            del generation_output
            torch.cuda.empty_cache()    

        connection.close()
        
        return extracts
    
    
    def generate_response(self, query, extracts:list[tuple[str, str]]):
        summary = [text for (_, text) in extracts]
        
        system_message = 'You are an intelligent and helpful assistant that provides detailed answers to questions in complete sentences.'
        
        prompt = f'''Given the following question and contexts, use only information found within the contexts to provide a detailed answer to the question.
        The answer provided should be supported by information within the contexts.
        Remember, *DO NOT* apply your own external knowledge to answer the question.

        question = {query}

        contexts = {summary}

        answer = '''

        prompt_template = f'''<|im_start|>system
        {system_message}<|im_end|>

        <|im_start|>user
        {prompt}<|im_end|>

        <|im_start|>assistant
        '''
        
        tokens = self.tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        
        # Normal
        generation_params = {"do_sample": True,
                             "temperature": 0.7,
                             "top_p": 0.95,
                             "top_k": 40,
                             "max_new_tokens": 2048,
                             "repetition_penalty": 1.1
                             }
        
        generation_output = self.model.generate(tokens,
                                        #    streamer=streamer,
                                           **generation_params
                                           )
    
        token_output = generation_output[0]
        text_output = self.tokenizer.decode(token_output[tokens.shape[1]:]).partition('<|im_end|>')[0]

        del generation_output
        torch.cuda.empty_cache()
        
        return text_output
    
    
    def get_response_images(self, extracts:list[tuple[str, str]]):
        connection = duckdb.connect(os.path.join(RELATIONAL_DIR, 'data.db'))
        ids = tuple(id for (id, _) in extracts)
        ref_data = connection.sql(f"SELECT filename, page_number FROM parliament_debate WHERE id in {ids}").fetchall()

        filepaths = []
        for (filename, pg_num) in ref_data:
            folder_name = os.path.splitext(filename)[0]

            for pg in pg_num.split(', '):
                pagepath = os.path.join(IMG_DIR, f'{folder_name}', f'{pg}.jpg')
                filepaths.append((pagepath, f'Document:{filename}, Page:{pg}'))
                
        connection.close()
        
        return filepaths