# src/agents/orchestrator.py

"""
MultiAgentOrchestrator — 3-agent pipeline.

Luồng 2 stages:
    Stage 1: TextAgent  (text+table chunks) → aT
             ImageAgent (image chunks)      → aI
    Stage 2: SumAgent   (ghép thô aT + aI → chỉnh sửa mạch lạc)

3 LLM calls/question: Text + Image + Sum
"""

from openai import OpenAI
from src.retrieval.retriever import RetrievalResult
from src.agents.agents import (
    TextAgent, ImageAgent, SumAgent,
    AgentOutput, _format_chunks,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class MultiAgentOrchestrator:
    """Điều phối 3-agent reasoning pipeline."""

    def __init__(
        self,
        api_key: str,
        text_model: str,
        vision_model: str,
    ):
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

        self.text_agent     = TextAgent(client, text_model)
        self.image_agent    = ImageAgent(client, vision_model)
        self.sum_agent      = SumAgent(client, text_model)

    def run(self, question: str, result: RetrievalResult) -> dict:
        """
        Chạy 3-agent pipeline.

        Returns:
            {
                "answer", "sources", "confidence", "reasoning",
                "agent_outputs",
                "model", "has_image", "num_images",
            }
        """
        text_chunks  = result.text_chunks
        image_chunks = result.image_chunks

        print(f"    Text chunks  : {len(text_chunks)}")
        print(f"    Image chunks : {len(image_chunks)}")

        # ── Stage 1: Text + Image (parallel — saves ~40% latency) ──
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            print(f"    → Text + Image Agents (parallel)...", flush=True)
            text_future = executor.submit(
                self.text_agent.analyze, question, text_chunks,
            )
            image_future = executor.submit(
                self.image_agent.analyze, question, image_chunks,
            )
            text_output = text_future.result()
            image_output = image_future.result()

        print(f"      text={text_output.confidence:.2f}, image={image_output.confidence:.2f}")

        # ── Stage 2: Sum Agent (editor) ──
        print(f"    → Sum Agent...", end=" ", flush=True)
        final = self.sum_agent.synthesize(question, text_output, image_output)
        print(f"confidence={final['confidence']:.2f}")

        # Metadata
        final["model"]         = "multi-agent-3"
        final["has_image"]     = image_output.has_data
        final["num_images"]    = len(image_chunks) if image_output.has_data else 0

        return final
