REFUSAL_PHRASE = (
    "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden."
)

SYSTEM_PROMPT = f"""Du bist ein Assistent für deutsche gynäkologische Leitlinien.

Regeln:
- Beantworte NUR anhand des bereitgestellten Kontexts (Chunks). Nutze kein Vorwissen außerhalb des Kontexts.
- Wenn die Antwort NICHT im Kontext steht, antworte exakt:
  "{REFUSAL_PHRASE}"
- Auch wenn der Kontext thematisch ähnlich klingt, aber die konkrete Antwort nicht enthält — verweigere.
- Erfinde nichts. Rate nicht. Antworte auf Deutsch.
- Wenn im Kontext mehrere Altersgruppen oder Screening-Bänder vorkommen (z. B. 20–34 vs. ab 35), beantworte NUR das in der Frage genannte Band. Vermische keine Empfehlungen zwischen Altersgruppen.
- Wenn Untersuchung/Testverfahren und Intervall im Kontext stehen, nenne beides ausdrücklich in der Antwort.
"""


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        page = c.get("page", "?")
        blocks.append(
            f"[{c['id']}] (source={c.get('source')}, Seite={page}, typ={c.get('type')})\n{c['text']}"
        )
    ctx = "\n\n".join(blocks) if blocks else "(kein Kontext)"
    return f"Kontext:\n{ctx}\n\nFrage: {question}\n\nAntwort:"


def is_refusal(text: str) -> bool:
    return REFUSAL_PHRASE.lower() in text.lower()
