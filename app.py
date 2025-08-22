import json
import os
import uuid
from typing import List, Optional, Dict, Any
from textwrap import dedent
import warnings

warnings.filterwarnings("ignore")

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from flask import Flask, request, jsonify
from generate_report import generate_report, load_json_file
from dotenv import load_dotenv

from langchain_groq import ChatGroq

# --- 1. SETUP ---
load_dotenv(override=True)
llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY")
)


# --- 2. DEFINE STRUCTURED OUTPUTS (PYDANTIC MODELS) ---
class ClarificationQuestions(BaseModel):
    questions: List[str] = Field(
        description="A list of 2-3 critical questions to refine the user's high-level goal."
    )


class HighLevelPlan(BaseModel):
    plan: Dict[str, str] = Field(
        description="A dictionary where keys are descriptive phase names and values are clear descriptions of the objective for that phase."
    )


class TaskAssignment(BaseModel):
    project_name: str = Field(
        description="A short, descriptive name for the overall project (e.g., 'DSML Placement Excellence Q3')."
    )
    assignee_name: str = Field(
        description="The full name of the single employee best suited for this task."
    )
    task_title: str = Field(
        description="A concise, one-line, action-oriented title for the task."
    )
    detailed_description: str = Field(
        description="A detailed description of the task, its context, and expected deliverables."
    )
    success_metrics: List[str] = Field(
        description="A list of 2-3 specific, measurable metrics to define success."
    )
    deadline: str = Field(
        description="A suggested deadline for the task (e.g., 'End of Week 2')."
    )


class SubtaskBreakdown(BaseModel):
    sub_tasks: List[TaskAssignment] = Field(
        description="A list of new, granular TaskAssignment objects to be delegated to a manager's direct subordinates."
    )


# --- 3. SETUP ROBUST OUTPUT PARSERS ---
clarify_parser = PydanticOutputParser(pydantic_object=ClarificationQuestions)
plan_parser = PydanticOutputParser(pydantic_object=HighLevelPlan)
assign_parser = PydanticOutputParser(pydantic_object=TaskAssignment)
subtask_parser = PydanticOutputParser(pydantic_object=SubtaskBreakdown)


# --- 4. DEFINE THE STATE FOR THE GRAPH ---
class AgentState(BaseModel):
    hierarchy_data: Dict[str, Any]
    original_goal: str
    clarified_goal: Optional[str] = None
    projects: Dict[str, Any] = Field(default_factory=dict)
    task_queue: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)


# --- 5. HELPER FUNCTIONS ---
def load_hierarchy(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        return json.load(f)


def find_employee(name: str, hierarchy_node: Dict) -> Optional[Dict]:
    if hierarchy_node["name"] == name:
        return hierarchy_node
    for subordinate in hierarchy_node.get("subordinates", []):
        found = find_employee(name, subordinate)
        if found:
            return found
    return None


def _assign_task(
    state: AgentState, task_details: Dict, parent_task_id: Optional[str] = None
) -> str:
    project_name = task_details["project_name"]
    assignee_name = task_details["assignee_name"]
    if project_name not in state.projects:
        state.projects[project_name] = {
            "project_name": project_name,
            "overall_goal": state.clarified_goal,
            "status": "In Progress",
            "tasks": [],
        }
    task_id = str(uuid.uuid4())
    full_task_object = {
        "task_id": task_id,
        "parent_task_id": parent_task_id,
        "status": "Assigned",
        **task_details,
    }
    state.projects[project_name]["tasks"].append(full_task_object)
    employee = find_employee(assignee_name, state.hierarchy_data)
    if employee:
        if "action_items" not in employee:
            employee["action_items"] = []
        action_item = {
            "task_id": task_id,
            "project_name": project_name,
            "task_title": task_details["task_title"],
            "deadline": task_details["deadline"],
            "status": "Assigned",
        }
        employee["action_items"].append(action_item)
    else:
        state.logs.append(
            f"⚠️ [ERROR] Could not find employee '{assignee_name}' to assign task."
        )
    return task_id


# --- 6. DEFINE THE NODES OF THE LANGGRAPH AGENT ---
def start_node(state: AgentState) -> AgentState:
    state.logs.append(f"▶️ [START] Goal Received: '{state.original_goal}'")
    return state


def clarify_goal_node(state: AgentState) -> AgentState:
    state.logs.append("🤔 [CLARIFY] Agent is formulating clarifying questions...")
    prompt = dedent(
        f"""
        You are a Senior Strategist at Scaler. A high-level goal has been proposed: "{state.original_goal}".
        Our mission is to ensure learners achieve top-tier career outcomes. Ask questions to make this goal SMART.
        {clarify_parser.get_format_instructions()}
    """
    )
    chain = llm | clarify_parser
    questions_obj = chain.invoke(prompt)
    state.logs.append("❓ [CLARIFY] Agent has questions for the user:")
    for q in questions_obj.questions:
        state.logs.append(f"  - {q}")
    simulated_answers = (
        "We want to increase DSML placements by 20% in the next quarter. "
        "Focus on improving the quality and relevance of our interview preparation content. "
        "We should target high-growth startups and established tech companies."
    )
    state.clarified_goal = f"Original Goal: {state.original_goal}. User Clarifications: {simulated_answers}"
    state.logs.append(
        f"✅ [CLARIFY] Goal clarified: Increase DSML placements by 20% next quarter by improving interview content quality."
    )
    return state


def create_plan_node(state: AgentState) -> AgentState:
    state.logs.append("📝 [PLAN] Creating a high-level, phased strategic plan...")
    prompt = dedent(
        f"""
        You are the VP of Projects at Scaler. Create a strategic roadmap for the initiative: "{state.clarified_goal}".
        Break this down into a 3-5 phase plan representing major workstreams to enhance learner placement success.
        {plan_parser.get_format_instructions()}
    """
    )
    chain = llm | plan_parser
    plan_obj = chain.invoke(prompt)
    state.logs.append("📋 [PLAN] High-level plan created:")
    for i, (phase, desc) in enumerate(plan_obj.plan.items()):
        state.logs.append(f"  Phase {i+1}: {phase} - {desc}")
        state.task_queue.append(
            {
                "task_description": f"{phase}: {desc}",
                "context": state.clarified_goal,
                "level": 0,
            }
        )
    return state


def process_task_node(state: AgentState) -> AgentState:
    if not state.task_queue:
        return state

    task_info = state.task_queue.pop(0)
    task_description = task_info["task_description"]
    task_level = task_info["level"]
    indent = "  " * task_level

    state.logs.append(f"{indent}🔄 [ASSIGN] Processing task: '{task_description}'")

    def get_hierarchy_summary(node, level=0):
        summary = "  " * level + f"- {node['name']} ({node['job_role']})\n"
        summary += "  " * (level + 1) + f"  Description: {node['job_description']}\n"
        for sub in node.get("subordinates", []):
            summary += get_hierarchy_summary(sub, level + 1)
        return summary

    hierarchy_summary = get_hierarchy_summary(state.hierarchy_data)

    # --- FIX IS HERE: MORE FORCEFUL PROMPT ---
    prompt = dedent(
        f"""
        You are the Chief of Staff at Scaler. Your job is to assign a strategic task by generating a single, valid JSON object.

        **Company-Wide Goal:** "{state.clarified_goal}"
        **Task to be Assigned:** "{task_description}"
        **Organizational Chart & Roles:**
        ```
        {hierarchy_summary}
        ```
        **CRITICAL INSTRUCTIONS:**
        1.  **Analyze and Select:** Based on the task and roles, identify the SINGLE most appropriate person to own this task.
        2.  **Generate JSON ONLY:** Your entire response MUST be a single, valid JSON object. Do NOT include any introductory text, explanations, markdown formatting, or any text outside of the JSON structure.
        3.  **Follow the Schema:** The JSON object must strictly conform to the provided schema.

        {assign_parser.get_format_instructions()}
    """
    )

    assign_chain = llm | assign_parser
    assignment_obj = assign_chain.invoke(prompt)
    task_details = assignment_obj.dict()

    parent_task_id = _assign_task(state, task_details)
    state.logs.append(
        f"{indent}✅ [ASSIGN] Task '{task_details['task_title']}' ASSIGNED to -> {task_details['assignee_name']}"
    )

    assignee = find_employee(task_details["assignee_name"], state.hierarchy_data)
    if assignee and assignee.get("subordinates"):
        state.logs.append(
            f"{indent}↪️ [SUB-TASK] {assignee['name']} is a manager. Breaking down their task for the team..."
        )
        subordinates_summary = "\n".join(
            [
                f"- {s['name']} ({s['job_role']}): {s['job_description']}"
                for s in assignee["subordinates"]
            ]
        )
        project_name = task_details["project_name"]

        # --- FIX IS HERE: MORE FORCEFUL PROMPT ---
        subtask_prompt = dedent(
            f"""
            You are the manager: {assignee['name']} at Scaler. Your job is to delegate tasks by generating a single, valid JSON object.
            
            **Overall Project Name:** "{project_name}"
            **Your Assigned Task:**
            - Title: {task_details['task_title']}
            - Description: {task_details['detailed_description']}
            **Your Team:**
            ```
            {subordinates_summary}
            ```
            **CRITICAL INSTRUCTIONS:**
            1.  **Decompose and Assign:** Break down your main task into smaller sub-tasks and assign them to the most appropriate people on your team.
            2.  **Use Project Name:** For EACH sub-task, you MUST include the 'project_name' field set to "{project_name}".
            3.  **Generate JSON ONLY:** Your entire response MUST be a single, valid JSON object containing a 'sub_tasks' list. Do NOT include any introductory text, explanations, or markdown formatting.
            4.  **Follow the Schema:** The JSON object must strictly conform to the provided schema.

            {subtask_parser.get_format_instructions()}
        """
        )

        subtask_chain = llm | subtask_parser
        subtask_obj = subtask_chain.invoke(subtask_prompt)

        for sub_task in subtask_obj.sub_tasks:
            sub_task_dict = sub_task.dict()
            _assign_task(state, sub_task_dict, parent_task_id=parent_task_id)
            state.logs.append(
                f"{indent}  ✅ [SUB-TASK ASSIGNED] '{sub_task_dict['task_title']}' -> {sub_task_dict['assignee_name']}"
            )

    return state


# --- 7. DEFINE CONDITIONAL EDGES & BUILD GRAPH ---
def should_continue(state: AgentState) -> str:
    if state.task_queue:
        return "process_task"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("start", start_node)
workflow.add_node("clarify_goal", clarify_goal_node)
workflow.add_node("create_plan", create_plan_node)
workflow.add_node("process_task", process_task_node)
workflow.set_entry_point("start")
workflow.add_edge("start", "clarify_goal")
workflow.add_edge("clarify_goal", "create_plan")
workflow.add_edge("create_plan", "process_task")
workflow.add_conditional_edges(
    "process_task",
    should_continue,
    {"process_task": "process_task", END: END},
)
workflow_app = workflow.compile()

# Initialize Flask app
flask_app = Flask(__name__)


@flask_app.route("/start_goal", methods=["POST"])
def start_initiative():
    try:
        data = request.json
        if not data or "goal" not in data:
            return jsonify({"error": "Please provide a goal in the request body"}), 400

        # Load hierarchy data
        hierarchy = load_hierarchy("test_data.json")

        goal = """
        The Goal description is as follow.
        """
        for k, v in data.items():
            goal += f"{k}: {v}\n"

        # Create initial state
        initial_state = AgentState(hierarchy_data=hierarchy, original_goal=goal)
        # return jsonify({"message": "Goal received and processing started."}), 200
        # Run the workflow
        final_state_dict = workflow_app.invoke(initial_state)

        # Return the result
        return jsonify(final_state_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flask_app.route("/generate-report", methods=["POST"])
def generate_report_endpoint():
    data = request.get_json()
    requester_name = data.get("requester_name")
    project_name = data.get("project_name")

    if not requester_name or not project_name:
        return (
            jsonify({"error": "Both requester_name and project_name are required"}),
            400,
        )

    try:
        # Load required data
        hierarchy = load_json_file("final_team_assignments.json")
        projects = load_json_file("project_report_data.json")

        # Generate the report
        report = generate_report(requester_name, project_name, hierarchy, projects)

        if report:
            return jsonify(
                {
                    "requester_name": requester_name,
                    "project_name": project_name,
                    "report": report,
                }
            )
        else:
            return jsonify({"error": "Failed to generate report"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 8. RUN THE AGENT AND SAVE THE OUTPUT ---
if __name__ == "__main__":

    flask_app.run(debug=True, port=5000)
