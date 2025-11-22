import asyncio
from pathlib import Path

from dotenv import load_dotenv

from app.agent import LLMSolverAgent
from app.browser import scrape_quiz_page
from app.llm import AIPipeClient, parse_task

DEMO_URL = "https://tds-llm-analysis.s-anand.net/demo"
WORKSPACE = Path("demo_workspace")

load_dotenv()


async def main():
    print(f"Scraping {DEMO_URL} …")
    scrape_result = await scrape_quiz_page(DEMO_URL)

    print("Parsing quiz instructions via LLM …")
    task = await parse_task(scrape_result.html)

    print("Launching solver agent …")
    client = AIPipeClient()  # reuses your env-configured key/model
    agent = LLMSolverAgent(client=client, max_turns=8)
    result = await agent.solve(task, workspace=WORKSPACE)

    print("\nAgent answer:", result.answer)
    print("Reasoning:", result.reasoning)
    print("\nSteps:")
    for step in result.steps:
        print(f"- [{step.action}] {step.thought}")
        if step.observation:
            print(f"    observation: {str(step.observation)[:200]}…")


asyncio.run(main())
