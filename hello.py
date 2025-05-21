import chainlit as cl
import os
from dotenv import load_dotenv,find_dotenv
from agents import Agent,RunConfig,AsyncOpenAI,OpenAIChatCompletionsModel,Runner


load_dotenv(find_dotenv())

# step1 provider

gemini_api_key= os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
  api_key=gemini_api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# step2 model
model = OpenAIChatCompletionsModel(
  model="gemini-2.0-flash",
  openai_client=provider,
)
# 
# step 03 Config:Define at run level

config = RunConfig(
  model=model,
  model_provider=provider,
  tracing_disabled=True
)

Replier_agent : Agent = Agent(
    name=" Replier Assistant",
    instructions="Read the email and its summary, then write a polite and professional reply that matches the email's content. For example, if the email is about a meeting, confirm the meeting or suggest a time.",
    model=model,
    handoffs=[]
)
Summarizer_agent : Agent = Agent(
    name=" Summrizer Assistant",
    instructions="Create a short, clear summary of important emails. Include the main point of the email, like ,This email is about a meeting, or ,This email asks a question., If a reply is needed, send the summary and email to the Replier Agent.",
    model=model,
    handoffs=[Replier_agent]
)

triage_agent : Agent = Agent(
    name="General Assistant",
    instructions="Read the email and classify it as spam, important, or a specific category (e.g., meeting, question). If the email is important, send it to the Summarizer Agent. If it,s spam, ignore or delete it",
    model=model,
    handoffs=[Summarizer_agent]
)

@cl.on_chat_start
async def handle_chat_start():
  cl.user_session.set("history", [])
  await cl.Message(content=f""" # Email Handling Assistant """).send()
  await cl.Message(content="Hello! I am Email Support Assistant. How Can i help you today").send()


@cl.on_message
async def handle_message(message: cl.Message):
  history = cl.user_session.get("history")
  msg = cl.Message(content="")
  await msg.send()
  history.append({"role": "user", "content": message.content})
  
  
  
  result = Runner.run_sync(
    triage_agent,
    input=history,
    run_config=config
    
  )
 

  history.append({"role": "assistant", "content": result.final_output}) 
  cl.user_session.set("history", history)  
  await cl.Message(content=result.final_output).send()