import asyncio
from typing import Dict, Set, Optional
from uuid import UUID, uuid4
from protocol.message import BaseMessage, encode_channel


class Channel:
    def __init__(self):
        self.agent_inboxes: Dict[str, asyncio.Queue] = {}
        self.seen_message_ids: Set[UUID] = set()
        self.sequence_counters: Dict[str, int] = {}  # encode_channel(a,b) -> last seq

    def register_agent(self, agent_id: str) -> None:
        if agent_id not in self.agent_inboxes:
            self.agent_inboxes[agent_id] = asyncio.Queue()

    async def send(self, message: BaseMessage) -> None:
        if message.recipient_agent_id not in self.agent_inboxes:
            raise ValueError(f"Agent {message.recipient_agent_id} not registered")

        # Deduplication
        if message.message_id in self.seen_message_ids:
            return
        self.seen_message_ids.add(message.message_id)

        # Sequence enforcement (per directed channel)
        # Sequence enforcement (per directed channel)
        # Sequence enforcement (per directed channel)
        channel_key = encode_channel(message.sender_agent_id, message.recipient_agent_id)
        last_seq = self.sequence_counters.get(channel_key, 0)
        if message.sequence_number != 0 and message.sequence_number <= last_seq:
            return
        self.sequence_counters[channel_key] = message.sequence_number
        await self.agent_inboxes[message.recipient_agent_id].put(message)

    async def receive(self, agent_id: str) -> BaseMessage:
        if agent_id not in self.agent_inboxes:
            raise ValueError(f"Agent {agent_id} not registered")
        return await self.agent_inboxes[agent_id].get()

    def receive_nowait(self, agent_id: str) -> Optional[BaseMessage]:
        """Non-blocking receive — returns None if inbox empty."""
        if agent_id not in self.agent_inboxes:
            raise ValueError(f"Agent {agent_id} not registered")
        try:
            return self.agent_inboxes[agent_id].get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def broadcast(self, message: BaseMessage) -> None:
        """Send message to all agents except sender, each with unique message_id."""
        tasks = []
        for agent_id in self.agent_inboxes:
            if agent_id == message.sender_agent_id:
                continue
            # Build new message copy addressed to this specific agent
            msg_data = message.model_dump()
            msg_data["recipient_agent_id"] = agent_id
            msg_data["message_id"] = uuid4()
            new_msg = BaseMessage.from_dict(msg_data)
            tasks.append(self.send(new_msg))
        await asyncio.gather(*tasks)

    def get_inbox_size(self, agent_id: str) -> int:
        return self.agent_inboxes[agent_id].qsize() if agent_id in self.agent_inboxes else 0
