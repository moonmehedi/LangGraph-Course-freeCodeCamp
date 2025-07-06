# %%
# %%
from typing import TypedDict, List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

load_dotenv()

# %%
class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

# %%
llm = ChatOpenAI(model='gpt-4o-mini')


def process(state:AgentState)->AgentState:
    '''this node will solve the input you give '''

    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f'\nAI:{response.content}')

    print("State So Far: ",state['messages'])
    return state



# %%
graph = StateGraph(AgentState)

graph.add_node('process',process)

#start and end edges

graph.add_edge(START,'process')
graph.add_edge('process',END)


agent = graph.compile()


# %%
from IPython.display import display ,Image

display(Image(agent.get_graph().draw_mermaid_png()))

# %%
conversation_history = []
user_input = input("Enter: ")

while user_input != 'exit':
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({'messages':conversation_history})
    conversation_history = result['messages']
    user_input = input('Enter: ')



with open('/workspaces/LangGraph-Course-freeCodeCamp/practice/exercies/AgentsPractice/memory.txt','w') as file:
    file.write('Your conversation Log : \n')
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f'you:{message.content}\n')
        elif isinstance(message,AIMessage):
            file.write(f'AI: {message.content}\n\n')
    file.write('End Of Conversation')

print("conversation saved to lagging.txt")

    


