import numpy as np
import streamlit as st
import openai
import pandas as pd
from scipy.spatial.distance import cosine

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

st.write("Hello World -- from Ishaan")

openai.api_key = st.secrets["OPENAI_API_KEY"]

def call_chatgpt(prompt: str) -> str:
    """
    Uses the OpenAI API to generate an AI response to a prompt.

    Args:
        prompt: A string representing the prompt to send to the OpenAI API.

    Returns:
        A string representing the AI's generated response.

    """

    # Use the OpenAI API to generate a response based on the input prompt.
    # What is the difference between this invocation (openai.Completion.create) and the function below (initialize_agent(...).run)?
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans

SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]

def call_langchain(prompt: str) -> str:
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True)
    output = agent.run(prompt)

    return output

def openai_text_embedding(prompt: str) -> str:
    return openai.Embedding.create(input=prompt, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]

def calculate_sts_openai_score(sentence1: str, sentence2: str) -> float:
    # Compute sentence embeddings
    embedding1 = openai_text_embedding(sentence1)  # Flatten the embedding array
    embedding2 = openai_text_embedding(sentence2)  # Flatten the embedding array

    # Convert to array
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score

def add_dist_score_column(
    dataframe: pd.DataFrame, sentence: str
) -> pd.DataFrame:
    
    dataframe["stsopenai"] = dataframe["questions"].apply(
            lambda x: calculate_sts_openai_score(x, sentence)
    )

    sorted_dataframe = dataframe.sort_values(by="stsopenai", ascending=False)

    return sorted_dataframe.iloc[:5, :]

df = pd.read_csv("mckinsey-covid-report.csv")            # Read the CSV

# Check if "messages" is not stored in session state and set it to an empty list
if "messages" not in st.session_state:
    st.session_state.messages = []

# Iterate over each message in the session state messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from chat_input and store it in the prompt variable using the walrus operator ":="
if prompt := st.chat_input("What is up?"):
    # Add user message to session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    df_screened_by_dist_score = add_dist_score_column(
        df, prompt
    )                                                        # Compute the distance score between the prompt and each item in the CVS?
    ref_from_internet = call_langchain(prompt)             # Appears to get the default answer to the prompt from ChatGPT?
    ref_from_covid_data = df_screened_by_dist_score.answers  # Not exactly sure what this does?
    engineered_prompt = f"""
        Basedon the context: {ref_from_internet},
        and based on more context: {ref_from_covid_data}
        answer the user prompt {prompt}
    """                                                      # Create an engineered prompt
    response = call_chatgpt(engineered_prompt)               # Can be changed depending on model


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
