from typing import Literal
from typing_extensions import TypedDict
from random import randint

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

class GameState(TypedDict):
    messages: list[AnyMessage]
    board: list[str]
    winner: str | None

def winning_logic(board):
    combos = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]

    for a,b,c in combos:
        if board[a] != "" and board[a] == board[b] == board[c]:
            return board[a]

    if "" not in board:
        return "Draw"

    return None


def next_player(board):
    return "X" if board.count("X") == board.count("O") else "O"


def get_valid_moves(board):
    return [i for i in range(9) if board[i] == ""]

@tool
def get_valid_moves_tool(board):
    """
    Tool for the model to use to get valid moves for the game
    """
    return get_valid_moves(board)

@tool
def next_player_tool(board):
    """
    Tool to decide what the agent is going to play X or O
    """
    return next_player(board)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    groq_api_key="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"
)

tools = [get_valid_moves_tool, next_player_tool]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

def router_node(state: GameState) -> GameState:
    winner = winning_logic(state["board"])
    if winner:
        state["winner"] = winner
    return state


def router_logic(state: GameState) -> Literal["human_move","model_move","print_board_node"]:
    if state["winner"]:
        return "print_board_node"

    if next_player(state["board"]) == "X":
        return "human_move"

    return "model_move"

def human_move(state: GameState) -> GameState:
    board = state["board"]
    play = next_player(board)
    valid = get_valid_moves(board)

    print(f"Choose position for {play}: {valid}")

    move = -1
    while move not in valid:
        try:
            move = int(input("Enter position: "))
        except:
            move = -1

    board[move] = play
    state["board"] = board

    return state


def llm_call(state: GameState) -> GameState:
    board = state["board"]

    if not state["messages"]:
        system_prompt = f"""
        You are playing TicTacToe like a 12 year old.

        Board index layout:

        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8

        Current board:
        {board}

        First call get_valid_moves_tool.
        Then choose exactly one valid move.
        Return only the move number.
        """

        state["messages"].append(SystemMessage(content=system_prompt))

    print("Running inference")

    response = model_with_tools.invoke(state["messages"])
    state["messages"].append(response)

    return state

def tool_node(state: GameState) -> GameState:
    last = state["messages"][-1]

    for call in last.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        observation = tools_by_name[tool_name].invoke(tool_args)

        state["messages"].append(
            ToolMessage(
                content=str(observation),
                tool_call_id=call["id"],
            )
        )

    return state

def router_llm_node(state: GameState) -> GameState:
    return state


def router_llm_logic(state: GameState) -> Literal["tool_node","print_board_node"]:
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"

    board = state["board"]
    play = next_player(board)

    try:
        move = int(last.content.strip())
    except:
        move = get_valid_moves(board)[0]

    if move not in get_valid_moves(board):
        move = get_valid_moves(board)[0]

    board[move] = play
    state["board"] = board

    state["messages"] = []

    return "print_board_node"

def print_board_node(state: GameState) -> GameState:
    b = state["board"]

    def cell(i):
        return b[i] if b[i] else " "

    print(f" {cell(0)} | {cell(1)} | {cell(2)} ")
    print("---+---+---")
    print(f" {cell(3)} | {cell(4)} | {cell(5)} ")
    print("---+---+---")
    print(f" {cell(6)} | {cell(7)} | {cell(8)} ")

    if state["winner"]:
        print("Winner:", state["winner"])
    else:
        print("Next turn:", next_player(b))

    return state

def after_print_logic(state: GameState) -> Literal["router", END]:
    if state["winner"]:
        return END
    return "router"

builder = StateGraph(GameState)

builder.add_node("router", router_node)
builder.add_node("human_move", human_move)
builder.add_node("model_move", llm_call)
builder.add_node("router_llm", router_llm_node)
builder.add_node("tool_node", tool_node)
builder.add_node("print_board_node", print_board_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    router_logic,
    {
        "human_move": "human_move",
        "model_move": "model_move",
        "print_board_node": "print_board_node",
    },
)

builder.add_edge("human_move", "print_board_node")

builder.add_edge("model_move", "router_llm")

builder.add_conditional_edges(
    "router_llm",
    router_llm_logic,
    {
        "tool_node": "tool_node",
        "print_board_node": "print_board_node",
    },
)

builder.add_conditional_edges(
    "print_board_node",
    after_print_logic,
    {
        "router": "router",
        END: END,
    },
)

builder.add_edge("tool_node", "model_move")

agent = builder.compile()

if __name__ == "__main__":

    initial_state = {
        "board": [""] * 9,
        "winner": None,
        "messages": [],
    }

    print_board_node(initial_state)

    agent.invoke(initial_state)
