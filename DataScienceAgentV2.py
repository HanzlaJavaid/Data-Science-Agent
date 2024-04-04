
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from langchain.schema import ChatMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.utilities import PythonREPL
import streamlit as st
from custom_python_env import execute_code
from context_library import context_v5
import os
from agent_tool_beta import *
from llama_index.core import set_global_handler

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Phoenix can display in real time the traces automatically
# collected from your LlamaIndex application.
# import phoenix.session.session as px

# Look for a URL in the output to open the App in a browser.
# session = px.launch_app(port=9000)
# The App is initially empty, but as you proceed with the steps below,
# traces will appear automatically as your LlamaIndex application runs.

import llama_index.core

# llama_index.core.set_global_handler("arize_phoenix")

st.title(':blue[The Data Science Agent]')
def python_repl(
    code: str
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""

    print('CODE:', code)
    try:
        code = code.strip('Python')
        result = execute_code(code,globals())
        print('RESULT: ', result)
        return result
    except BaseException as e:
        print(f"Failed to execute. Code: {code} Error: {repr(e)}")
        return f"Failed to execute. Code: {code} Error: {repr(e)}"
    print(f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}")
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

python_repl = FunctionTool.from_defaults(
    fn=python_repl,
    name="python_repl",
    description="""Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.""",
)

# Choose the LLM to use


context = context_v5
llm_openai = OpenAI(model="gpt-3.5-turbo-0125",temperature = 0.5,api_key=openai_api_key)
llm = llm_openai
agent = ReActAgent.from_tools(
            [python_repl],
            llm=llm,
            verbose=True,
            context=context,
            max_iterations=20,
        )
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

if "memory" not in st.session_state:
    st.session_state["memory"] = []


for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt_message := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt_message))
    st.chat_message("user").write(prompt_message)
    with st.chat_message("assistant"):
        #stream_handler = StreamHandler(st.empty())
        user_question = f"memory = {st.session_state['memory']} \n\n Question: {prompt_message}"
        print(user_question)
        st.session_state['memory'].append({"role":"user","content":prompt_message})
        response = agent.chat(user_question)
        agent_response = response.response
        st.session_state['memory'].append({"role": "assistant", "content": agent_response})
        print(response)
        st.session_state["messages"] = [ChatMessage(role="assistant", content=agent_response)]
    st.chat_message("assistant").write(agent_response)
