from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# 1. Load your existing local model and FAISS index
llm = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.1, num_ctx=2048)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local("faiss_pm_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

# 2. Turn the retriever into a Tool the agent can choose to call
retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description=(
        "Search the local document index for information. "
        "Use this whenever the user asks a question that might be "
        "answered by the ingested document (e.g. Transformer architecture, "
        "product management topics)."
    ),
)

tools = [retriever_tool]

# 3. A ReAct-style prompt (the model must "think" before acting)
react_prompt = PromptTemplate.from_template("""
Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format EXACTLY. Action and Action Input must be on SEPARATE lines.
Do NOT write the action as a function call like tool(query='x').

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as plain text only, no parentheses or quotes
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here is a correct example:

Question: What is the Kano model?
Thought: I should search the document index for this term.
Action: document_search
Action Input: Kano model
Observation: [results would appear here]
Thought: I now know the final answer
Final Answer: [the answer based on the observation]

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# 4. Build the agent and its executor
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # shows the Thought/Action/Observation trace in your terminal
    handle_parsing_errors=True,  # small models sometimes format badly; this prevents a crash
)

# 5. Run it
if __name__ == "__main__":
    print("\n=== Local ReAct Agent Initialized (Type 'exit' to quit) ===")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke({"input": user_input})
        print(f"\nAgent: {result['output']}")
