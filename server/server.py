"""
server.py — MCP server for the PUCIT GPA/CGPA tools.

Local:  python server.py            -> http://127.0.0.1:8000/mcp
Cloud:  host reads PORT from the environment (Render sets it).
"""
import os

from fastmcp import FastMCP

import tools

mcp = FastMCP("cgpa")

# docstring + type hints become the tool description the model reads
for fn in (
    tools.marks_to_grade_points,
    tools.calculate_semester_gpa,
    tools.calculate_new_cgpa,
    tools.required_gpa_for_target,
    tools.get_semester_courses,
    tools.get_remaining_credit_hours,
    tools.save_report,
):
    mcp.tool(fn)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    mcp.run(transport="http", host=host, port=port)
