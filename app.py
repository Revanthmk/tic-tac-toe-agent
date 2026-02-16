from flask import Flask, request, jsonify
from typing_extensions import TypedDict
from typing import Literal
from random import randint
import os

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

app = Flask(__name__)

# 🔐 Use environment variable instead of hardcoded key
groq_api_key = os.environ.get("GROQ_API_KEY")

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
    return get_valid_moves(board)


@tool
def next_player_tool(board):
    return next_player(board)


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    groq_api_key=groq_api_key
)

tools = [get_valid_moves_tool, next_player_tool]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


def ai_move(board):
    system_prompt = f"""
    You are playing TicTacToe like a 12 year old.

    Board layout:
    0 | 1 | 2
    3 | 4 | 5
    6 | 7 | 8

    Current board:
    {board}

    First call get_valid_moves_tool.
    Then choose exactly one valid move.
    Return only the move number.
    """

    messages = [SystemMessage(content=system_prompt)]
    response = model_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        valid = get_valid_moves(board)
        return valid[0]

    try:
        move = int(response.content.strip())
    except:
        move = get_valid_moves(board)[0]

    if move not in get_valid_moves(board):
        move = get_valid_moves(board)[0]

    return move


@app.route("/")
def home():
    return """
    <h2>TicTacToe Web CLI</h2>
    <p>POST to /move with JSON:</p>
    <pre>
    {
        "board": ["","","","","","","","",""],
        "move": 0
    }
    </pre>
    """


@app.route("/move", methods=["POST"])
def move():
    data = request.json
    board = data["board"]
    human_move = data["move"]

    if human_move in get_valid_moves(board):
        board[human_move] = "X"

    winner = winning_logic(board)

    if not winner:
        ai = ai_move(board)
        board[ai] = "O"
        winner = winning_logic(board)

    return jsonify({
        "board": board,
        "winner": winner
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
