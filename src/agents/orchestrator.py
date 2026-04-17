# src/agents/orchestrator.py

"""
MultiAgentOrchestrator — 4-agent pipeline.

Luồng 3 stages:
    Stage 1: CriticalAgent (text + image + question) → {Tc, Ic}
    Stage 2: TextAgent  (text+table chunks + Tc) → aT
             ImageAgent (image chunks + Ic)       → aI
    Stage 3: SumAgent   (ghép thô aT + aI → chỉnh sửa mạch lạc)

4 LLM calls/question: Critical + Text + Image + Sum
"""

from openai import OpenAI
from src.retrieval.retriever import RetrievalResult
from src.agents.agents import (
    CriticalAgent, TextAgent, ImageAgent, SumAgent,
    AgentOutput, _format_chunks,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class MultiAgentOrchestrator:
    """Điều phối 4-agent reasoning pipeline."""

    def __init__(
        self,
        api_key: str,
        text_model: str,
        vision_model: str,
    ):
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

        self.critical_agent = CriticalAgent(client, text_model)
        self.text_agent     = TextAgent(client, text_model)
        self.image_agent    = ImageAgent(client, vision_model)
        self.sum_agent      = SumAgent(client, text_model)

    def run(self, question: str, result: RetrievalResult) -> dict:
        """
        Chạy 4-agent pipeline.

        Returns:
            {
                "answer", "sources", "confidence", "reasoning",
                "agent_outputs", "critical_info",
                "model", "has_image", "num_images",
            }
        """
        text_chunks  = result.text_chunks
        image_chunks = result.image_chunks

        print(f"    Text chunks  : {len(text_chunks)}")
        print(f"    Image chunks : {len(image_chunks)}")

        # ── Stage 1: Critical Agent ──
        print(f"    → Critical Agent...", end=" ", flush=True)
        text_context  = _format_chunks(text_chunks)
        image_context = _format_chunks(image_chunks)

        critical_info = self.critical_agent.extract(
            question=question,
            text_context=text_context,
            image_context=image_context,
        )
        print(f"text='{critical_info['text'][:50]}' image='{critical_info['image'][:50]}'")

        # ── Stage 2: Text + Image (guided) ──
        print(f"    → Text Agent...", end=" ", flush=True)
        text_output = self.text_agent.analyze(
            question, text_chunks,
            critical_info=critical_info["text"],
        )
        print(f"confidence={text_output.confidence:.2f}")

        print(f"    → Image Agent...", end=" ", flush=True)
        image_output = self.image_agent.analyze(
            question, image_chunks,
            critical_info=critical_info["image"],
        )
        print(f"confidence={image_output.confidence:.2f}")

        # ── Stage 3: Sum Agent (editor) ──
        print(f"    → Sum Agent...", end=" ", flush=True)
        final = self.sum_agent.synthesize(question, text_output, image_output)
        print(f"confidence={final['confidence']:.2f}")

        # Metadata
        final["critical_info"] = critical_info
        final["model"]         = "multi-agent-4"
        final["has_image"]     = image_output.has_data
        final["num_images"]    = len(image_chunks) if image_output.has_data else 0

        return final
