from __future__ import annotations

from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Instrument


class GuidanceService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def answer(
        self,
        *,
        instrument_name: str,
        question: str,
        context: str | None,
        mode: Literal["ai", "module"],
        db: Session,
    ) -> tuple[str, str]:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.name.ilike(instrument_name.strip()))
            .one_or_none()
        )

        if not instrument:
            answer = (
                f"No instrument metadata found for '{instrument_name}'. "
                "Ask using one of the known instrument labels from the detection panel."
            )
            return answer, "fallback-local"

        module_answer = self._build_module_answer(instrument=instrument, question=question)
        if mode == "module":
            return module_answer, "module-local"

        if self.settings.groq_api_key:
            model_response = await self._ask_groq(
                instrument_name=instrument.name,
                description=instrument.description,
                operation_steps=instrument.operation_steps,
                safety_warnings=instrument.safety_warnings,
                question=question,
                context=context,
            )
            if model_response:
                return model_response, self.settings.groq_model

        if not self.settings.groq_api_key:
            return (
                "AI mode selected, but GROQ_API_KEY is not configured. Showing module guidance instead.\n\n"
                f"{module_answer}"
            ), "module-local"

        return (
            "AI mode selected, but the AI request failed. Showing module guidance instead.\n\n"
            f"{module_answer}"
        ), "module-local"

    def _build_module_answer(self, *, instrument: Instrument, question: str) -> str:
        return (
            f"Instrument: {instrument.name}\n"
            f"Description: {instrument.description}\n"
            f"Procedure: {instrument.operation_steps}\n"
            f"Safety: {instrument.safety_warnings}\n"
            f"Suggested answer: {question} should be handled by following the listed procedure "
            "and verifying all safety warnings before operation."
        )

    async def _ask_groq(
        self,
        *,
        instrument_name: str,
        description: str,
        operation_steps: str,
        safety_warnings: str,
        question: str,
        context: str | None,
    ) -> str | None:
        import httpx

        system_prompt = (
            "You are a laboratory teaching assistant. Respond with concise, step-by-step guidance "
            "that prioritizes safety and procedural correctness."
        )
        user_prompt = (
            f"Instrument: {instrument_name}\n"
            f"Description: {description}\n"
            f"Operation Steps: {operation_steps}\n"
            f"Safety Warnings: {safety_warnings}\n"
            f"Scene Context: {context or 'N/A'}\n"
            f"Question: {question}"
        )

        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 350,
        }

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
