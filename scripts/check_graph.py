"""Static smoke test: import the agent module and validate the workflow graph.

Does not call any model — just confirms the ADK 2.1 Workflow builds correctly.
Run with:  uv run python scripts/check_graph.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import german_teacher.agent as a

print("IMPORT OK")
print("provider USING_LM_STUDIO =", a.USING_LM_STUDIO)
print("root type:", type(a.root_agent).__name__, "| name:", a.root_agent.name)

g = a.root_agent.graph
print("nodes:", sorted(n.name for n in g.nodes))
print("edges:")
for e in g.edges:
    print("   ", e.from_node.name, "->", e.to_node.name)
print("terminal nodes:", g._terminal_node_names)
print("news feed:", a.NEWS_RSS_URL)
