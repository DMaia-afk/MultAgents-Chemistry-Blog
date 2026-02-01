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
my_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434" 
)

strategy_agent = Agent(
    role="Scientific Content Strategist",
    goal="Define high-value, relevant chemistry topics aligned with the blog's educational goals",
    backstory="You are responsible for the scientific content strategy of a chemistry blog. "
              "You decide which topics should be published based on educational value, audience level "
              "and long-term content pillars. You do not write content or explain chemistry concepts. "
              "Your decisions guide all downstream agents.",
    allow_delegation=False,
	verbose=True,
    memory = False,
    llm=my_llm
)

chemistry_expert = Agent(
    role="Chemistry Subject Matter Expert",
    goal="Provide scientifically accurate definitions, laws, formulas and conceptual boundaries for topic",
    backstory="You are a chemistry expert responsible for ensuring scientific rigor. "
              "You define correct concepts, equations, mechanisms and known exceptions related to topic. "
              "You identify common misconceptions and define what can and cannot be simplified. "
              "Your output is authoritative and technically precise.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

seo_researcher = Agent(
    role="Scientific SEO Researcher",
    goal="Structure chemistry content around search intent without altering scientific accuracy",
    backstory="You specialize in SEO for scientific and educational content. "
              "You analyze how students and readers search for chemistry topics like topic. "
              "You define primary and secondary keywords, search intent and an optimal article structure. "
              "You never modify or reinterpret chemistry concepts.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

didactic_agent = Agent(
    role="Didactic Chemistry Educator",
    goal="Translate rigorous chemistry concepts into accessible explanations for the target audience",
    backstory="You are an experienced chemistry educator. "
              "Your task is to translate complex scientific explanations into clear and pedagogically sound language. "
              "You may use safe analogies when explicitly allowed, but you never introduce new scientific information "
              "or contradict the chemistry expert.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

technical_writer = Agent(
    role="Technical Chemistry Writer",
    goal="Write a clear, structured and accurate chemistry article based on provided inputs",
    backstory="You are a technical writer specialized in chemistry content. "
              "You write the full article using the structure defined by the SEO researcher "
              "and the explanations validated by the chemistry expert and didactic agent. "
              "You do not introduce new concepts, analogies or interpretations.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

scientific_reviewer = Agent(
    role="Scientific Chemistry Reviewer",
    goal="Review the article for scientific accuracy, clarity and conceptual consistency",
    backstory="You are a rigorous scientific reviewer with a strong background in chemistry education. "
              "You verify terminology, units, equations and conceptual coherence. "
              "You identify inaccuracies, ambiguities and dangerous oversimplifications. "
              "You suggest precise corrections without rewriting the entire article.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

seo_onpage_agent = Agent(
    role="SEO On-page Optimizer",
    goal="Optimize chemistry content for search engines without altering scientific meaning",
    backstory="You specialize in on-page SEO for scientific content. "
              "You refine titles, headings, meta descriptions and internal links "
              "while preserving the exact scientific meaning of the article. "
              "You avoid clickbait and misleading simplifications.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

applications_agent = Agent(
    role="Chemistry Applications Specialist",
    goal="Connect chemistry concepts to real-world, industrial or laboratory applications",
    backstory="You focus on practical applications of chemistry concepts. "
              "You relate theory to real-world examples such as industrial processes, materials, "
              "laboratory techniques or everyday phenomena. "
              "All examples must be factual and scientifically accurate.",
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

# Create Tasks and Crew
strategy_task = Task(
    description=(
        "Define the most relevant chemistry topic to be published. "
        "The topic must align with the blog's educational goals, target audience level "
        "and long-term content pillars."
    ),
    expected_output=(
        "A structured topic plan including:\n"
        "- Selected topic\n"
        "- Educational objective\n"
        "- Target audience level\n"
        "- Content pillar"
    ),
    agent=strategy_agent
)

chemistry_expert_task = Task(
    description=(
        "Provide a scientifically rigorous explanation of the selected topic. "
        "Define all relevant concepts, laws, formulas, equations and known exceptions. "
        "Identify common misconceptions and define acceptable simplification boundaries."
    ),
    expected_output=(
        "A technical chemistry brief including:\n"
        "- Formal definitions\n"
        "- Equations and formulas (when applicable)\n"
        "- Key scientific principles\n"
        "- Common misconceptions\n"
        "- Simplification limits"
    ),
    agent=chemistry_expert
)

seo_research_task = Task(
    description=(
        "Analyze search intent for the selected chemistry topic and define an optimal article structure. "
        "Identify primary and secondary keywords without altering scientific meaning."
    ),
    expected_output=(
        "An SEO research report including:\n"
        "- Primary keyword\n"
        "- Secondary keywords\n"
        "- Search intent\n"
        "- Recommended article outline (H1–H3)"
    ),
    agent=seo_researcher
)

didactic_task = Task(
    description=(
        "Translate the rigorous chemistry explanation into an accessible and pedagogically sound version "
        "appropriate for the defined audience level. "
        "Use safe analogies only when appropriate and never introduce new scientific information."
    ),
    expected_output=(
        "A didactic explanation including:\n"
        "- Clear, audience-appropriate explanations\n"
        "- Approved analogies (if applicable)\n"
        "- Pedagogical notes or warnings"
    ),
    agent=didactic_agent
)

writing_task = Task(
    description=(
        "Write the full chemistry article using the approved structure, scientific content "
        "and didactic explanations. "
        "Ensure clarity, logical flow and technical accuracy."
    ),
    expected_output=(
        "A complete draft of the chemistry article, ready for scientific review."
    ),
    agent=technical_writer
)

review_task = Task(
    description=(
        "Review the article for scientific accuracy, conceptual consistency, correct terminology "
        "and unit usage. Identify errors, ambiguities or dangerous oversimplifications."
    ),
    expected_output=(
        "A review report including:\n"
        "- Identified issues\n"
        "- Required corrections\n"
        "- Suggested precise adjustments"
    ),
    agent=scientific_reviewer
)

seo_onpage_task = Task(
    description=(
        "Optimize the reviewed article for on-page SEO while preserving exact scientific meaning. "
        "Refine titles, headings, meta description and internal linking."
    ),
    expected_output=(
        "On-page SEO deliverables including:\n"
        "- SEO-optimized title\n"
        "- Meta description\n"
        "- Optimized headings\n"
        "- Internal linking suggestions"
    ),
    agent=seo_onpage_agent
)

applications_task = Task(
    description=(
        "Provide real-world applications related to the chemistry topic. "
        "Connect theory to industry, laboratory practice or everyday phenomena "
        "using only factual and scientifically accurate examples."
    ),
    expected_output=(
        "A list of real-world chemistry applications with brief explanations."
    ),
    agent=applications_agent
)

crew = Crew(
    agents=[strategy_agent, chemistry_expert, seo_researcher, didactic_agent,
            technical_writer, scientific_reviewer, seo_onpage_agent, applications_agent],
    tasks=[strategy_task, chemistry_expert_task, seo_research_task, didactic_task,
           writing_task, review_task, seo_onpage_task, applications_task],
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