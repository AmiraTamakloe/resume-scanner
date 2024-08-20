import os
from crewai import Crew, Process
from dotenv import load_dotenv, find_dotenv 
from langchain_groq import ChatGroq 
from tools.utils import *
from tools.agents import agents
from tools.tasks import tasks
load_dotenv(find_dotenv())

# Configuration of dotenv (api keys)
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load the llm
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Provided the inputs
resume = read_all_pdf_pages("Student_CV_example.pdf")
job_desire = input("Enter Desiring Job: ")

# Creating agents and tasks
job_requirements_researcher, resume_swot_analyser = agents(llm)

research, resume_swot_analysis = tasks(llm, job_desire, resume)

# Building crew and kicking it off
crew = Crew(
    agents=[job_requirements_researcher, resume_swot_analyser],
    tasks=[research, resume_swot_analysis],
    verbose=1,
    process=Process.sequential
)

result = crew.kickoff()
print(result)