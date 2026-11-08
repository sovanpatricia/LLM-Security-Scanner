# Adapted from langchain-ai/streamlit-agent (commit 634e1cecf23d7ca3a4c5e708944673e057765b2a),
# streamlit_agent/chat_with_sql_db.py lines 1-10, 19, 50-58. Apache-2.0; see LICENSE.upstream.
# Extracted 2026-11-08 for a controlled pilot case; see case.json.
import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3

LOCALDB = "USE_LOCALDB"


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    if db_uri == LOCALDB:
        # Make the DB connection read-only to reduce risk of injection attacks
        # See: https://python.langchain.com/docs/security
        db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    return SQLDatabase.from_uri(database_uri=db_uri)
