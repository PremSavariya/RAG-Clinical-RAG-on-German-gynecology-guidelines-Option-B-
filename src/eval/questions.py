"""Answer-quality / refusal eval bank (Q1–9 answerable, Q10–12 traps).

Flags (eval only — never passed into retrieve/generate):
  answerable / trap — traps must refuse; answerable Qs should answer from context
  gold — core screening questions used in gold averages
  expected_evidence — reference text for judge / semantic similarity
  evidence_keywords — groups; a chunk matches if EVERY group has ≥1 term in the text
"""
from __future__ import annotations

QUESTIONS: list[dict] = [
    {
        "id": 1,
        "answerable": True,
        "gold": True,
        "question": (
            "Welche Untersuchung und in welchem Intervall wird für Frauen im Alter "
            "von 20 bis 34 Jahren im organisierten Screening empfohlen?"
        ),
        "expected_evidence": "Altersband 20–34, zytologiebasiert jährlich",
        "gold_pages": [82],
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
        "gold_pages": [82, 112],
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
            "Selbstentnahme angeboten werden. Sie soll Frauen vorbehalten bleiben, die sich nicht "
            "an der regulären Krebsvorsorge beteiligen."
        ),
        "gold_pages": [129, 130, 127],
        "evidence_keywords": [
            ["selbstabnahme", "selbstentnahme", "self-sampling", "self sampling", "eigenentnahme"],
            ["non-responder", "nonresponder", "nicht teil", "nicht an der regulaeren", "nicht an der regulären"],
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
            "Bei einer zytologiebasierten Screening-Strategie werden Zellen vom Gebärmutterhals "
            "entnommen und mikroskopisch auf auffällige Zellveränderungen untersucht (Pap-Abstrich). "
            "Bei einer HPV-basierten Strategie wird die Probe mit einem molekularbiologischen Test "
            "auf Hochrisiko-HPV-Typen untersucht. Laut Leitlinie besitzt der HPV-Test für CIN2+-Läsionen "
            "eine höhere Sensitivität, aber eine niedrigere Spezifität als die Zytologie. HPV-Tests "
            "sind außerdem üblicherweise besser reproduzierbar."
        ),
        "gold_pages": [59, 50],
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
            "Die Leitlinie empfiehlt eine Abklärungskolposkopie bei ausgewählten auffälligen "
            "Screeningbefunden, insbesondere bei hochgradigen zytologischen Auffälligkeiten, einem "
            "positiven HPV-16/18-Nachweis oder einem geschätzten Risiko von mindestens 10 % für "
            "CIN 3+, AIS beziehungsweise ein Adenokarzinom."
        ),
        "gold_pages": [100, 101, 105],
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
            "Laut Leitlinie ist die HPV-Impfung eine zentrale Maßnahme der Primärprävention. "
            "Sie verhindert Infektionen mit den durch den Impfstoff abgedeckten HPV-Typen und "
            "reduziert dadurch das Risiko für Krebsvorstufen und HPV-assoziierte Krebserkrankungen. "
            "Die Impfung ersetzt jedoch nicht die regelmäßige Teilnahme am Zervixkarzinom-Screening."
        ),
        "gold_pages": [42, 43],
        "evidence_keywords": [
            ["impfung", "impfstoff", "vakzin"],
            ["primaerpraevention", "primärprävention", "primaer", "primär"],
            ["screening", "frueherkenn", "früherkenn", "ersetzt nicht"],
        ],
    },
    # --- Traps: out of guideline scope; correct behavior is refusal ---
    {
        "id": 10,
        "answerable": False,
        "gold": False,
        "trap": True,
        "question": (
            "Welche Thrombolyse-Therapie und welches Zeitfenster gelten beim "
            "akuten ischämischen Schlaganfall?"
        ),
        "expected_evidence": "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden.",
        "gold_pages": [],
        "evidence_keywords": [],
    },
    {
        "id": 11,
        "answerable": False,
        "gold": False,
        "trap": True,
        "question": ("Welches Antibiotikum ist first-line bei einer ambulant erworbenen Pneumonie?"),
        "expected_evidence": "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden.",
        "gold_pages": [],
        "evidence_keywords": [],
    },
    {
        "id": 12,
        "answerable": False,
        "gold": False,
        "trap": True,
        "question": ("Welche medikamentöse Erstlinientherapie wird bei arterieller Hypertonie empfohlen?"),
        "expected_evidence": "Ich konnte in den bereitgestellten Textpassagen keine Antwort auf diese Frage finden.",
        "gold_pages": [],
        "evidence_keywords": [],
    },
]
