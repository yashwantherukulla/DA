from gpt_researcher import GPTResearcher
import asyncio
import json

from typing import Tuple, List, Dict, Any, Union

async def get_report(query: str, report_type: str) -> Tuple[str, List[Any], float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    researcher = GPTResearcher(query, report_type, config_path='./config.json')
    print(json.dumps(researcher.cfg.__dict__, indent=2))
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    
    research_context = researcher.get_research_context()
    research_costs = researcher.get_costs()
    research_images = researcher.get_research_images()
    research_sources = researcher.get_research_sources()
    
    return report, research_context, research_costs, research_images, research_sources

def test():
    researcher = GPTResearcher(query, report_type, config_path='./config.json')
    print(json.dumps(researcher.cfg.__dict__, indent=2))

if __name__ == "__main__":
    query = "Should I invest in Nvidia?"
    report_type = "research_report"

    # test()

    report, context, costs, images, sources = asyncio.run(get_report(query, report_type))
    
    print("Report:")
    print(report)
    print("\nResearch Costs:")
    print(costs)
    print("\nResearch Images:")
    print(images)
    print("\nResearch Sources:")
    print(sources)