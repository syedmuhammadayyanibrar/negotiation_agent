import psycopg2
import psycopg2.extras
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class NegotiationLogger:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        psycopg2.extras.register_uuid()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS negotiation_logs (
                    id SERIAL PRIMARY KEY,
                    negotiation_id TEXT NOT NULL,
                    round_number INT,
                    sender_agent_id TEXT,
                    recipient_agent_id TEXT,
                    message_type TEXT,
                    message_data JSONB,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_negotiation_id
                    ON negotiation_logs(negotiation_id);

                CREATE TABLE IF NOT EXISTS negotiation_outcomes (
                    id SERIAL PRIMARY KEY,
                    negotiation_id TEXT UNIQUE NOT NULL,
                    outcome TEXT,
                    agreed_allocation JSONB,
                    rounds_taken INT,
                    via_coalition BOOLEAN DEFAULT FALSE,
                    failure_reason TEXT,
                    agent_ids JSONB,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            self.conn.commit()

    def log_message(self, message_data: Dict[str, Any]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO negotiation_logs
                   (negotiation_id, round_number, sender_agent_id,
                    recipient_agent_id, message_type, message_data)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    message_data["negotiation_id"],
                    message_data.get("round_number", 0),
                    message_data["sender_agent_id"],
                    message_data["recipient_agent_id"],
                    message_data["message_type"],
                    json.dumps(message_data),
                ),
            )
            self.conn.commit()

    def log_outcome(self, outcome_data: Dict[str, Any]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO negotiation_outcomes
                   (negotiation_id, outcome, agreed_allocation, rounds_taken,
                    via_coalition, failure_reason, agent_ids)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (negotiation_id) DO UPDATE
                   SET outcome=EXCLUDED.outcome,
                       agreed_allocation=EXCLUDED.agreed_allocation,
                       rounds_taken=EXCLUDED.rounds_taken,
                       via_coalition=EXCLUDED.via_coalition,
                       failure_reason=EXCLUDED.failure_reason""",
                (
                    outcome_data["negotiation_id"],
                    outcome_data.get("outcome"),
                    json.dumps(outcome_data.get("agreed_allocation")),
                    outcome_data.get("rounds_taken"),
                    outcome_data.get("via_coalition", False),
                    outcome_data.get("failure_reason"),
                    json.dumps(outcome_data.get("agent_ids", [])),
                ),
            )
            self.conn.commit()

    def get_transcript(self, negotiation_id: str) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT message_data FROM negotiation_logs "
                "WHERE negotiation_id = %s ORDER BY timestamp",
                (negotiation_id,),
            )
            return [row["message_data"] for row in cur.fetchall()]

    def get_outcome(self, negotiation_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM negotiation_outcomes WHERE negotiation_id = %s",
                (negotiation_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def close(self) -> None:
        self.conn.close()
