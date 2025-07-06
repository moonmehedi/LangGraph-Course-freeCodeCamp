# %%
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage,HumanMessage,AIMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode

load_dotenv()

# %%
 #this is the global variable to store document content

document_content = ''

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

# %%
@tool
def update(content:str)->str:
    """ Update the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is :\n{document_content}"


@tool
def save(filename:str)->str:
    """ 
    Save the current document to a text file and finish the process.

    Args:
    filename: Name for the text file  
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open('/workspaces/LangGraph-Course-freeCodeCamp/practice/exercies/AgentsPractice/'+filename,'w') as file:
            file.write(document_content)
        print(f'\n Document has been saved to :',{filename})
        return f'Document has been saved successfully to {filename}.'
    except Exception as e:
        return f"Error saving document :{str(e)}"
    

tools = [update,save]

model = ChatOpenAI(model = 'gpt-4o-mini').bind_tools(tools)


def our_agent(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    if not state['messages']:
        user_input = input("I'm ready to help you update a document. What would you like to create ?\n")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input('\n what would y like to do with the document ? ')
        print("\n USER:{user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages'])+ [user_message]

    response = model.invoke(all_messages)

    print(f'\n AI: {response.content}')

    #if tool call happen print it

    if hasattr(response,'tool_calls') and response.tool_calls:
        print(f'Using Tools:{[tc['name'] for tc in response.tool_calls]}')

    return {'messages':list(state['messages']) + [user_message,response]}


def should_continue(state:AgentState) -> str:
    """ Determine if we should continue or end the conversationf"""

    messages  = state['messages']

    if not messages:
        return 'continue'
    
    for message in reversed(messages):
        if(isinstance(message, ToolMessage)) and 'saved' in message.content.lower() and 'document' in message.content.lower():
            return 'end_node'
    return 'continue'

# print it nicely

def print_message(messages):
    """Function I made to print the messages in a more reliable formate """
    if not messages:
        return 
    
    for message in messages[-3:]:
        if isinstance(message,ToolMessage):
            print(f'\n Tool Result :{message.content}')


graph = StateGraph(AgentState)

graph.add_node('agent',our_agent)
graph.add_node('tools',ToolNode(tools))

graph.set_entry_point('agent')
graph.add_edge('agent','tools')
graph.add_conditional_edges(
    'tools',
    should_continue,
    {
        'end_node':END,
        'continue':'agent'
    }
)


app = graph.compile()

def run_document_agent():
    print('\n ======= DRAFTER ========')

    state = {'messages':[]}

    for step in app.stream(state, stream_mode='values'):
        if 'messages' in step:
            print_message(step['messages'])

    print("\n ======= Drafter FINISHED =======")


if __name__ == "__main__":
    run_document_agent()


