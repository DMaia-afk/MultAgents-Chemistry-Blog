# Integração CrewAI + Ollama local (via LangChain)
import os

# Ensure OpenAI client uses local Ollama endpoint — set before importing OpenAI/CrewAI
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")

import warnings
from openai import OpenAI
from crewai import Agent, Task, Crew, LLM
warnings.filterwarnings('ignore')

# NOTE: You can still set real environment variables in your shell instead
# of relying on these defaults. In PowerShell:
#   set OPENAI_API_BASE=http://localhost:11434/v1
#   set OPENAI_API_KEY=ollama

# Create a CrewAI LLM object that points to the local Ollama instance.
# Use the 'ollama/' prefix so CrewAI uses the local provider instead of OpenAI.
model_content = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)
model_writer = LLM(
    model="ollama/phi3.5:latest",
    base_url="http://localhost:11434"
)
model_tool = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434" 
)

topic = ["The Role of pH in Food Processing", "The Crucial Role of pH in Biological Systems"]

strategy_agent = Agent(
    role="Scientific Strategy Agent for Chemistry Blog",
    goal="Define high-value, relevant chemistry topics aligned with the blog's educational goals and Structure chemistry content around search intent without altering scientific accuracy",
    backstory="You are responsible for the scientific content strategy of a chemistry blog."
              "You decide which topics should be published based on educational value, audience level"
              "and long-term content pillars. You do not write the full content; your focus is purely on structure and keywords."
              "Your decisions guide all downstream agents."
              "You specialize in SEO for scientific and educational content."
              "You analyze how students and readers search for chemistry topics like topic."
              "You define primary and secondary keywords, search intent and an optimal article structure."
              "You never repeat a {topic} that has already been covered in the blog.",
    allow_delegation=False,
	verbose=True,
    memory = False,
    llm=model_tool,
    temperature=0.6
)

chemistry_expert = Agent(
    role="Chemistry Subject Matter Expert and Scientific Chemistry Reviewer",
    goal="Provide scientifically accurate definitions, laws, formulas and conceptual boundaries for topic and Review the article for scientific accuracy, clarity and conceptual consistency",
    backstory="You are a chemistry expert responsible for ensuring scientific rigor. "
              "You define correct concepts, equations, mechanisms and known exceptions related to topic. "
              "You identify common misconceptions and define what can and cannot be simplified. "
              "Your output is authoritative and technically precise."
              "You are a rigorous scientific reviewer with a strong background in chemistry education. "
              "You verify terminology, units, equations and conceptual coherence. "
              "You identify inaccuracies, ambiguities and dangerous oversimplifications. "
              "You suggest precise corrections without rewriting the entire article.",
    allow_delegation=False,
    verbose=True,
    llm=model_writer,
    temperature=0.2
)

didactic_agent = Agent(
    role="Didactic Chemistry Educator and Chemistry Applications Specialist",
    goal="Translate rigorous chemistry concepts into accessible explanations for the target audience and Connect chemistry concepts to real-world, industrial or laboratory applications",
    backstory="You are an experienced chemistry educator. "
              "Your task is to translate complex scientific explanations into clear and pedagogically sound language. "
              "You may use safe analogies when explicitly allowed, but you never introduce new scientific information "
              "or contradict the chemistry expert."
              "You focus on practical applications of chemistry concepts. "
              "You relate theory to real-world examples such as industrial processes, materials, "
              "laboratory techniques or everyday phenomena. "
              "All examples must be factual and scientifically accurate.",
    allow_delegation=False,
    verbose=True,
    llm=model_content,
    temperature=0.5
)

final_editor = Agent(
    role="Technical Editor and SEO Integrity Guardian",
    goal="Finalize the chemistry article by ensuring 100% scientific accuracy and perfect SEO optimization.",
    backstory=(
        "You are the final gatekeeper of the blog. Your mission is twofold: "
        "1. Technical Integrity: You meticulously audit the text to ensure no chemistry terms, "
        "formulas or nomenclatures were distorted during writing. "
        "2. SEO Finalization: You polish titles, meta-descriptions, and H1-H3 tags for maximum search "
        "visibility without ever using clickbait. "
        "You receive a draft and deliver a publishing-ready masterpiece. "
        "You never compromise scientific truth for the sake of an SEO keyword."
    ),
    allow_delegation=False,
    verbose=True,
    memory=False,
    llm=model_tool
)



strategy_task = Task(
    description=(
        "1. Define a high-value chemistry topic aligned with our educational pillars.\n"
        "2. Conduct SEO research for this topic: identify primary/secondary keywords and search intent.\n"
        "3. Create a detailed article outline (H1-H3) that balances pedagogy with SEO structure."
    ),
    expected_output=(
        "A comprehensive content blueprint including: Selected Topic, Target Audience, "
        "Primary/Secondary Keywords, Search Intent, and a detailed H1-H3 Outline."
    ),
    agent=strategy_agent
)

expert_science_task = Task(
    description=(
        "Based on the blueprint, provide a rigorous scientific explanation. "
        "Define laws, IUPAC names, formulas (using LaTeX), and reaction mechanisms. "
        "Explicitly list common misconceptions and set 'red lines' for what cannot be simplified "
        "to ensure the content remains factually bulletproof."
    ),
    expected_output=(
        "A technical dossier containing: Exact formulas, formal definitions, "
        "detailed scientific principles, and a list of forbidden oversimplifications."
    ),
    agent=chemistry_expert
)

didactic_writing_task = Task(
    description=(
        "Draft the full article. You must translate the technical dossier into accessible language "
        "without losing accuracy. Integrate real-world applications (industrial/laboratory) "
        "throughout the text to maintain engagement. Follow the SEO outline strictly."
    ),
    expected_output=(
        "A complete, fluid chemistry article draft that connects theory to practice, "
        "written in a pedagogical tone and formatted in Markdown."
    ),
    agent=didactic_agent
)

final_optimization_task = Task(
    description=(
        "Final audit of the article. Check every formula, unit, and chemical term for accuracy. "
        "Then, optimize the On-page SEO: refine the title tag, meta description, "
        "and ensure keywords are naturally integrated. Fix any Markdown formatting issues."
    ),
    expected_output=(
        "The final publishing-ready article in Markdown, including SEO Metadata "
        "(Title, Description, Keywords) and a 'Final Integrity' confirmation."
    ),
    agent=final_editor
)

crew = Crew(
    agents=[strategy_agent, chemistry_expert, didactic_agent, final_editor],
    tasks=[strategy_task, expert_science_task, didactic_writing_task, final_optimization_task],
    verbose=True,
    memory = False,
)

result = crew.kickoff()

# Salva o resultado em um arquivo Markdown automaticamente
with open("resultado.md", "w", encoding="utf-8") as f:
    f.write(f"# Resultado do Workflow de Química\n\n{result}\n")
# Save the result to a Markdown file automatically
with open("result.md", "w", encoding="utf-8") as f:
    f.write(f"# Chemistry Workflow Result\n\n{result}\n")