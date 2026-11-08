# Adapted from ReversecLabs/damn-vulnerable-llm-agent, Apache-2.0.
# Modified 2026-11-08 for an isolated controlled disclosure case.
# See case.json and LICENSE.upstream for provenance and limitations.
import json
from langchain.agents import Tool

class TransactionDb:
    def get_user(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT userId,username FROM Users WHERE userId = ?", (user_id,)
        )
        rows = cursor.fetchall()

        # Get column names
        columns = [column[0] for column in cursor.description]

        # Convert rows to dictionaries with column names as keys
        users = [dict(zip(columns, row)) for row in rows]

        # Convert to JSON format
        return json.dumps(users, indent=4)

def get_current_user(input : str):
    db = TransactionDb()
    user = db.get_user(1)
    db.close()
    return user

get_current_user_tool = Tool(
    name='GetCurrentUser',
    func= get_current_user,
    description="Returns the current user for querying transactions."
)
