"""
ABSA project — LOCKED aspect + sentiment taxonomy.

Single source of truth. Every downstream step (labeling, baseline, transformer,
eval, serving) imports from here so the label space never drifts.

LOCKED 2026-06-03.
Decisions:
  - 9 aspects (no separate `data` aspect; folded: analytics->feature,
    database/backup/storage->software, data-loss/privacy->security).
  - Keyword map allows MULTI-MAP (a tag may seed >1 aspect, e.g.
    Recovery->account+performance, Firewall->security+network).
  - Expanded mid-tail keywords for ~98% tag coverage.
Do not rename/reorder aspects without re-labeling.
"""

# ---------------------------------------------------------------------------
# 9 canonical aspects (multi-label: a ticket may mention several)
# ---------------------------------------------------------------------------
ASPECTS = [
    "billing",
    "account",
    "performance",
    "security",
    "bug",
    "feature",
    "network",
    "hardware",
    "software",
]

# One-line scope definition per aspect (used verbatim in the labeling prompt).
ASPECT_DEFINITIONS = {
    "billing":     "Charges, invoices, payments, refunds, pricing, subscriptions, discounts, cost.",
    "account":     "Login, passwords, authentication, account access/recovery, onboarding, access control.",
    "performance": "Speed, crashes, outages, downtime, latency, disruption, instability, optimization, scalability, availability.",
    "security":    "Vulnerabilities, malware/virus, breaches, data protection/privacy, compliance (HIPAA), encryption, firewall, unauthorized/sensitive data, data loss.",
    "bug":         "Defects / errors / unexpected behavior / conflicts / incompatibility (not a perf outage).",
    "feature":     "Capabilities, integrations, API, documentation, enhancements, customization, compatibility, plugins, workflow/automation, dashboards, analytics.",
    "network":     "Connectivity, VPN, connection drops, server reachability, infrastructure, cloud reachability.",
    "hardware":    "Physical devices, equipment, drivers, firmware, hardware replacement/failure.",
    "software":    "Application install/setup, software update/upgrade, configuration, deployment, database/storage/backup, platform/tooling.",
}

# ---------------------------------------------------------------------------
# Per-aspect sentiment label space
# ---------------------------------------------------------------------------
# For each (ticket, aspect): one of these 3.
#   not_present -> aspect not mentioned in this ticket (implicit default)
#   negative    -> complaint / problem / dissatisfaction about this aspect
#   neutral     -> mentioned as inquiry/request/factual statement, no complaint
# NOTE: a `positive` class was dropped — support tickets are ~never genuine
# praise; the LLM's "positive" labels were ~100% actually-neutral inquiries
# (13/13 sampled wrong). Folded positive -> neutral. See labeled_llm_v1.csv.
SENTIMENT_LABELS = ["not_present", "negative", "neutral"]
POLARITY_LABELS = ["negative", "neutral"]  # when aspect IS present

# ---------------------------------------------------------------------------
# Keyword -> aspect map (lowercase substring match). Drives the WEAK
# `aspect_tags` prior in consolidate.py. NOT ground truth — a seed/prior only.
# A tag may match >1 aspect (multi-map is intentional).
# ---------------------------------------------------------------------------
ASPECT_MAP = {
    "billing": [
        "bill", "payment", "invoice", "refund", "charge", "pricing",
        "subscription", "discount", "cost", "financial", "finance",
    ],
    "account": [
        "account", "login", "recovery", "password", "authentication", "auth",
        "onboarding", "access control", "access management", "access",
        "unauthorized access",
    ],
    "performance": [
        "performance", "crash", "outage", "disruption", "slow", "latency",
        "downtime", "maintenance", "optimization", "scalability",
        "availability", "delay", "critical failure", "service recovery",
        "recovery", "tuning", "instability",
    ],
    "security": [
        "security", "virus", "malware", "breach", "compliance", "encryption",
        "vulnerability", "privacy", "confidential", "firewall", "hipaa",
        "antivirus", "cyber", "unauthorized", "sensitive", "integrity",
        "data protection", "data privacy", "data breach", "data loss",
        "dataloss", "phishing", "protection", "threat", "audit",
    ],
    "bug": [
        "bug", "issue", "error", "troubleshoot", "defect", "conflict",
        "incompatib", "outdated", "glitch", "malfunction",
    ],
    "feature": [
        "feature", "documentation", "integration", "api", "functionality",
        "enhancement", "customization", "compatib", "plugin", "tutorial",
        "workflow", "automation", "sync", "dashboard", "analytics",
        "user experience", "improvement",
    ],
    "network": [
        "network", "connectivity", "connection", "vpn", "server",
        "infrastructure", "cloud", "firewall",
    ],
    "hardware": [
        "hardware", "device", "equipment", "driver", "firmware", "replacement",
    ],
    "software": [
        "software", "application", "installation", "update", "upgrade",
        "configuration", "config", "setup", "deployment", "implementation",
        "database", "backup", "storage", "platform", "docker", "elasticsearch",
        "sap", "erp", "jira", "smartsheet", "website", "version", "patch",
        "monitoring", "cache",
    ],
}

# ---------------------------------------------------------------------------
# Routing target (kept here so serving uses the same constant) — from `queue`.
# ---------------------------------------------------------------------------
QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]


def map_aspects(raw_tags):
    """list[str] -> sorted unique list of canonical aspects (multi-map)."""
    found = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        t = tag.lower()
        for aspect, keywords in ASPECT_MAP.items():
            if any(k in t for k in keywords):
                found.add(aspect)
    return sorted(found)


# sanity
assert len(ASPECTS) == 9
assert all(a in ASPECT_DEFINITIONS for a in ASPECTS)
assert all(a in ASPECT_MAP for a in ASPECTS)
