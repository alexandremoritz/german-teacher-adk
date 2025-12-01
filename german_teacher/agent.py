from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from typing import Dict, Any
from google.adk.tools import ToolContext

MODEL = "gemini-2.0-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

recent_news_agent = LlmAgent(
    name="RecentNewsAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Searches for recent news in Germany.",
    instruction="""You are a specialized news agent. Your goal is to find one interesting and recent piece of news from Germany.
    
    Steps:
    1. Use the `google_search` tool to find news in German.
    2. Select a news item that is suitable for a language learner (interesting, clear context).
    3. Return the content of the news item.

    Output only the story text, with no introduction or explanation.
    """,
    tools=[google_search],
    output_key="recent_news",
)

writer_agent = LlmAgent(
    name="WriterAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Writes a text based on the news.",
    instruction="""You are a German language teacher writing a text for students.
    
    Input:
    - Recent News: {recent_news}
    
    Task:
    - Write a text in German based on the provided news.
    - The text should consist of 5 paragraphs, with approximately 100 words per paragraph.
    - Ensure the language level is appropriate for intermediate learners (B1/B2).
    - The text should be engaging and educational.
    - Don't include intro or outro.
    """,
    output_key="base_text",
)

memo_agent = LlmAgent(
    name="VokabelnAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Extracts difficult words and provides translations.",
    instruction="""You are a vocabulary expert.
    
    Input:
    - Text: {base_text}
    
    Task:
    - Identify 5 difficult or interesting words from the text.
    - For each word, provide:
        - The word itself (with article if it's a noun).
        - The German definition or a synonym.
        - The English translation.
    - Format the output clearly.
    - Don't include intro or outro.
    """,
    output_key="vokabeln",
)

understand_agent = LlmAgent(
    name="VerstandenAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Generates comprehension questions.",
    instruction="""You are a reading comprehension expert.
    
    Input:
    - Text: {base_text}
    
    Task:
    - Create 3 comprehension questions based on the text.
    - The questions should test the student's understanding of the main points and details.
    - Write the questions in German.
    - Don't include intro or outro.
    """,
    output_key="understanding_questions",
)

grammar_agent = LlmAgent(
    name="GrammatikAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Explains a grammar rule found in the text.",
    instruction="""You are a German grammar teacher.
    
    Input:
    - Text: {base_text}
    
    Task:
    - Identify one specific grammar rule or concept used in the text (e.g., passive voice, specific preposition usage, adjective endings).
    - Explain the rule clearly in German (or English if better for explanation, but preferably German).
    - Cite the sentence from the text where this rule is applied.
    - Don't include intro or outro.
    """,
    output_key="grammer_rule",
)

writing_assignment_agent = LlmAgent(
    name="SchreibaufgabeAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Creates a writing assignment for the student.",
    instruction="""You are a creative writing teacher.
    
    Input:
    - Text: {base_text}
    
    Task:
    - Create a writing assignment for the student based on the text.
    - The assignment should ask the student to write about 100 words.
    - The topic should be related to the news but allow for personal opinion or creative expression.
    - Write the assignment prompt in German.
    - Don't include intro or outro.
    """,
    output_key="writing_assignment",
)

parallel_exercises = ParallelAgent(
    name="ParallelExercisesAgent",
    sub_agents=[memo_agent, understand_agent, grammar_agent, writing_assignment_agent],
)

def aggregate_exercises(tool_context: ToolContext) -> str:
    """Aggregates the generated exercises into a Markdown format.
    
    Returns:
        The aggregated Markdown string.
    """
    state = tool_context.state
    
    base_text = state.get("base_text", "")
    vokabeln = state.get("vokabeln", "")
    verstanden = state.get("verstanden", "")
    grammatik = state.get("grammatik", "")
    schreibaufgabe = state.get("schreibaufgabe", "")
    
    aggregated_text = f"""# German Learning Exercises

## Text basierend auf aktuellen Nachrichten
{base_text}

## Vokabeln
{vokabeln}

## Verständnisfragen
{verstanden}

## Grammatik
{grammatik}

## Schreibaufgabe
{schreibaufgabe}
"""
    return aggregated_text

aggregation_agent = LlmAgent(
    name="AggregationAgent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Aggregates all generated content.",
    instruction="""Call the `aggregate_exercises` tool to aggregate the content.
    Output the result of the tool directly.

    Keep exact format as the tool output. Better to not change anything.

    Just repeat the tool output.
    """,
    tools=[aggregate_exercises],
)

generate_exercises = SequentialAgent(
    name="GenerateExercisesAgent",
    sub_agents=[recent_news_agent, writer_agent, parallel_exercises, aggregation_agent],
)

def save_to_markdown(filename: str, content: str):
    """Saves content to a markdown file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {filename}"

root_agent = LlmAgent(
    name="root_agent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    description="Generate complete german exercises based on recent news.",
    instruction="""You are a German Teacher Assistant.
    
    Your main capability is to generate full lesson plans and exercises based on current news.
    
    Rules:
    - Always use the `generate_exercises` tool when the user asks for exercises or a lesson.
    - After generating the exercises, use the `save_to_markdown` tool to save the result to a file named 'xxx.md' with xxx the name of the news.
    - You do not need to explain the process, just deliver the result and confirm the file has been saved.
    """,
    tools=[AgentTool(generate_exercises), save_to_markdown],
)