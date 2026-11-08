# Extracted/adapted from ReversecLabs/damn-vulnerable-llm-agent.
# Apache-2.0; see LICENSE.upstream. Modified for a controlled pilot on 2026-11-08.
import json

def get_user_transactions(self, userId):
    cursor = self.conn.cursor()
    cursor.execute(f"SELECT * FROM Transactions WHERE userId = '{str(userId)}'")
    rows = cursor.fetchall()

    # Get column names
    columns = [column[0] for column in cursor.description]

    # Convert rows to dictionaries with column names as keys
    transactions = [dict(zip(columns, row)) for row in rows]

    # Convert to JSON format
    return json.dumps(transactions, indent=4)
