from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from config import openai_keys

import os
os.environ['OPENAI_API_KEY'] = openai_keys

llm = OpenAI(temperature = 0.7)

def generate_restaurent_name_and_items(cuisine):


    # Chain 1 : Restaurent name
    prompt_template_name = PromptTemplate(

        input_variables = ['cuisine'],
        template = "I want to open a restaurant for {cuisine} food, Suggest a fancy name for this."
    )

    name_chain = LLMChain(llm = llm, prompt = prompt_template_name, output_key = "restaurent_name")

    # Chain 2 : Menu Items
    prompt_template_items = PromptTemplate(

        input_variables = ['restaurent_name'],
        template = """Suggest some menu items for {restaurent_name}. Return it as a comma seperated list."""
    )

    food_items_chain = LLMChain(llm = llm, prompt = prompt_template_items, output_key = "menu_items")

    from langchain.chains import SequentialChain

    chain = SequentialChain(
    
        chains = [name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurent_name', 'menu_items']
    )

    response = chain({'cuisine' : cuisine})

    return response

if __name__ == "__main__":

    print(generate_restaurent_name_and_items("Italian"))
