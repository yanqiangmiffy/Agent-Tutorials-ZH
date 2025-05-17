import json
import argparse

from agents import ReportStructureAgent, FirstSearchAgent, FirstSummaryAgent, ReflectionAgent, ReflectionSummaryAgent, \
    ReportFormattingAgent
from state import State
from utils import tavily_search, update_state_with_search_results
from language_utils import detect_language

from datetime import datetime
from dotenv import  load_dotenv

load_dotenv()

STATE = State()
# QUERY = "Tell me something interesting about human species"
QUERY = "请帮我调研一下RAG有哪些实用技巧以及经验总结"
NUM_REFLECTIONS = 2
NUM_RESULTS_PER_SEARCH = 3
CAP_SEARCH_LENGTH = 20000


def main(topic: str = QUERY):
    # 检测输入语言
    lang = detect_language(topic)
    print(f"检测到的语言: {lang}")
    
    report_structure_agent = ReportStructureAgent(topic)

    _ = report_structure_agent.mutate_state(STATE)

    first_search_agent = FirstSearchAgent()
    first_summary_agent = FirstSummaryAgent()
    reflection_agent = ReflectionAgent()
    reflection_summary_agent = ReflectionSummaryAgent()
    report_formatting_agent = ReportFormattingAgent()

    print(f"Total Number of Paragraphs: {len(STATE.paragraphs)}")

    idx = 1

    for paragraph in STATE.paragraphs:
        print(f"\nParagraph {idx}: {paragraph.title}")

        idx += 1

    for j in range(len(STATE.paragraphs)):

        print(f"\n\n==============Paragraph: {j + 1}==============\n")
        print(f"=============={STATE.paragraphs[j].title}==============\n")

        ##################

        message = json.dumps(
            {
                "title": STATE.paragraphs[j].title,
                "content": STATE.paragraphs[j].content
            }
        )

        output = first_search_agent.run(message)

        search_results = tavily_search(output["search_query"], max_results=NUM_RESULTS_PER_SEARCH)

        _ = update_state_with_search_results(search_results, j, STATE)

        ##################

        message = {
            "title": STATE.paragraphs[j].title,
            "content": STATE.paragraphs[j].content,
            "search_query": search_results["query"],
            "search_results": [result["raw_content"][0:CAP_SEARCH_LENGTH] for result in search_results["results"] if
                               result["raw_content"]]
        }

        _ = first_summary_agent.mutate_state(message=json.dumps(message), idx_paragraph=j, state=STATE)

        ##################

        for i in range(NUM_REFLECTIONS):
            print(f"Running reflection: {i + 1}")

            message = {"paragraph_latest_state": STATE.paragraphs[j].research.latest_summary,
                       "title": STATE.paragraphs[j].title,
                       "content": STATE.paragraphs[j].content}

            output = reflection_agent.run(message=json.dumps(message))

            search_results = tavily_search(output["search_query"])

            _ = update_state_with_search_results(search_results, j, STATE)

            message = {
                "title": STATE.paragraphs[j].title,
                "content": STATE.paragraphs[j].content,
                "search_query": search_results["query"],
                "search_results": [result["raw_content"][0:20000] for result in search_results["results"] if
                                   result["raw_content"]],
                "paragraph_latest_state": STATE.paragraphs[j].research.latest_summary
            }

            _ = reflection_summary_agent.mutate_state(message=json.dumps(message), idx_paragraph=j, state=STATE)

    report_data = [{"title": paragraph.title, "paragraph_latest_state": paragraph.research.latest_summary} for paragraph
                   in STATE.paragraphs]

    final_report = report_formatting_agent.run(json.dumps(report_data))

    print(final_report)

    with open(f"reports/report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md", "w") as f:
        f.write(final_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default=QUERY)
    args = parser.parse_args()

    main(args.topic)