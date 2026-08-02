"""Retrieval eval bank: Hit@k / Recall@k vs gold Chroma chunk ids.

Same questions as questions.py, plus relevant_chunk_ids for labeled golds
(Q1–3). Labels are eval-only and never leaked into retrieve().
relevant_chunk_ids use Chroma ids `{element_id}_cN` from ingest chunking.
"""
from __future__ import annotations

DOC = "015-027OLl_Praevention_Zervixkarzinom_2020-03-verlaengert"

QUESTIONS_CHUNK_IDS: list[dict] = [
    {
        "id": 1,
        "answerable": True,
        "gold": True,
        "question": (
            "Welche Untersuchung und in welchem Intervall wird für Frauen im Alter "
            "von 20 bis 34 Jahren im organisierten Screening empfohlen?"
        ),
        "expected_evidence": "Altersband 20–34, zytologiebasiert jährlich",
        "gold_pages": [73, 82],
        "relevant_chunk_ids": [
            f"{DOC}_930_c0",
            f"{DOC}_1017_c0",
        ],
        "evidence_keywords": [
            ["20", "34"],
            ["jaehrlich", "jährlich"],
            ["zytolog"],
        ],
    },
    {
        "id": 2,
        "answerable": True,
        "gold": True,
        "question": (
            "Welches Testverfahren und welches Intervall gelten für Frauen ab 35 Jahren?"
        ),
        "expected_evidence": "Ko-Testung HPV + Zytologie, 3-Jahres-Intervall",
        "gold_pages": [73, 82],
        "relevant_chunk_ids": [
            f"{DOC}_930_c0",
            f"{DOC}_1018_c0",
        ],
        "evidence_keywords": [
            ["ko-test", "kotest", "ko test", "kombinations"],
            ["hpv"],
            ["zytolog"],
            ["3 jahr", "3-jahr", "drei jahr", "3 jahre", "alle 3"],
        ],
    },
    {
        "id": 3,
        "answerable": True,
        "gold": True,
        "question": (
            "Wie ist das empfohlene Vorgehen bei einer Frau ab 35, die HPV-positiv, "
            "aber zytologisch unauffällig ist?"
        ),
        "expected_evidence": "Wiederholung Ko-Testung nach ~12 Monaten",
        "gold_pages": [84],
        "relevant_chunk_ids": [
            f"{DOC}_1040_c0",
        ],
        "evidence_keywords": [
            ["hpv"],
            ["12 monate", "12-monat", "12 monat", "einem jahr", "nach 12"],
            ["wiederhol", "ko-test", "kotest", "ko testung", "kontroll"],
        ],
    },
    {
        "id": 4,
        "answerable": True,
        "gold": False,
        "question": (
            "Ab welchem Alter beginnt das organisierte Zervixkarzinom-Screening in Deutschland? "
        ),
        "expected_evidence": "Ab 25 Jahren beginnt das organisierte Zervixkarzinomscreening in Deutschland.",
        "gold_pages": [76],
        "relevant_chunk_ids": [],
        "evidence_keywords": [
            ["25"],
            ["screening", "frueherkenn", "früherkenn", "organisiert"],
        ],
    },
    {
        "id": 5,
        "answerable": True,
        "gold": False,
        "question": (
            "Welche Rolle spielt die HPV-Selbstabnahme (Selbstentnahme) laut Leitlinie, "
            "und für welche Gruppe?"
        ),
        "expected_evidence": (
            "Laut Leitlinie dient die HPV-Selbstabnahme dazu, die Teilnahme am "
            "Zervixkarzinom-Screening bei Frauen zu erhöhen, die trotz Einladung nicht an der "
            "regulären Früherkennung teilnehmen. Diesen sogenannten Non-Respondern sollte eine "
            "Selbstentnahme angeboten werden."
        ),
        "gold_pages": [129, 130, 127],
        "relevant_chunk_ids": [],
        "evidence_keywords": [
            ["selbstabnahme", "selbstentnahme", "self-sampling", "self sampling", "eigenentnahme"],
            [
                "non-responder",
                "nonresponder",
                "nicht teil",
                "nicht an der regulaeren",
                "nicht an der regulären",
            ],
        ],
    },
    {
        "id": 6,
        "answerable": True,
        "gold": False,
        "question": (
            "Welche HPV-Typen sind für die Mehrzahl der Zervixkarzinome verantwortlich?"
        ),
        "expected_evidence": (
            "Die HPV-Typen 16 und 18 sind für 60 bis 70% aller Zervixkarzinome verantwortlich."
        ),
        "gold_pages": [38],
        "relevant_chunk_ids": [],
        "evidence_keywords": [
            ["16"],
            ["18"],
            ["60", "70"],
        ],
    },
    {
        "id": 7,
        "answerable": True,
        "gold": False,
        "question": (
            "Was ist der Unterschied zwischen einer zytologiebasierten und einer "
            "HPV-basierten Screening-Strategie?"
        ),
        "expected_evidence": (
            "Zytologie: Pap-Abstrich; HPV: molekularbiologischer Test. "
            "HPV höhere Sensitivität, niedrigere Spezifität."
        ),
        "gold_pages": [59, 50],
        "relevant_chunk_ids": [],
        "evidence_keywords": [
            ["zytolog"],
            ["hpv"],
            ["sensitivitaet", "sensitivität", "spezifitaet", "spezifität"],
        ],
    },
    {
        "id": 8,
        "answerable": True,
        "gold": False,
        "question": (
            "Welche Empfehlung gibt die Leitlinie zur Kolposkopie bei auffälligem Befund?"
        ),
        "expected_evidence": (
            "Abklärungskolposkopie bei ausgewählten auffälligen Screeningbefunden."
        ),
        "gold_pages": [100, 101, 105],
        "relevant_chunk_ids": [],
        "evidence_keywords": [
            ["kolposkop"],
            ["abklaer", "abklär", "auffaellig", "auffällig", "screeningbefund"],
        ],
    },
    {
        "id": 9,
        "answerable": True,
        "gold": False,
        "question": (
            "Welche Bedeutung hat die HPV-Impfung im Kontext der Prävention laut Leitlinie?"
        ),
        "expected_evidence": (
            "HPV-Impfung ist Primärprävention; ersetzt Screening nicht."
        ),
        "gold_pages": [42, 43],
        "relevant_chunk_ids": [],
        "evidence_keywords": [
            ["impfung", "impfstoff", "vakzin"],
            ["primaerpraevention", "primärprävention", "primaer", "primär"],
            ["screening", "frueherkenn", "früherkenn", "ersetzt nicht"],
        ],
    },
    # --- Traps: no gold chunks; used for parity with questions.py, not Hit/Recall ---
    {
        "id": 10,
        "answerable": False,
        "gold": False,
        "trap": True,
        "question": (
            "Welche Thrombolyse-Therapie und welches Zeitfenster gelten beim "
            "akuten ischämischen Schlaganfall?"
        ),
        "expected_evidence": (
            "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden."
        ),
        "gold_pages": [],
        "relevant_chunk_ids": [],
        "evidence_keywords": [],
    },
    {
        "id": 11,
        "answerable": False,
        "gold": False,
        "trap": True,
        "question": ("Welches Antibiotikum ist first-line bei einer ambulant erworbenen Pneumonie?"),
        "expected_evidence": (
            "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden."
        ),
        "gold_pages": [],
        "relevant_chunk_ids": [],
        "evidence_keywords": [],
    },
    {
        "id": 12,
        "answerable": False,
        "gold": False,
        "trap": True,
        "question": (
            "Welche medikamentöse Erstlinientherapie wird bei arterieller Hypertonie empfohlen?"
        ),
        "expected_evidence": (
            "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden."
        ),
        "gold_pages": [],
        "relevant_chunk_ids": [],
        "evidence_keywords": [],
    },
]
