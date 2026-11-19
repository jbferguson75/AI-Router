# ==================================================
# MODELS
# ==================================================

# Local RAG answer generation
CHAT_MODEL = "qwen2.5:3b"

# Embeddings
EMBED_MODEL = "nomic-embed-text"

# Routing decisions
ROUTER_MODEL = "qwen2.5:3b"

# Answer evaluation
EVALUATOR_MODEL = "qwen2.5:3b"

# Memory extraction
MEMORY_EXTRACTOR_MODEL = "qwen2.5:3b"

#qwen2.5:3b
# Memory scoring
MEMORY_SCORER_MODEL = "qwen2.5:3b"

# Cloud model
CLOUD_MODEL = "gpt-5"

# ==================================================
# OLLAMA SETTINGS
# ==================================================

# Context window
OLLAMA_NUM_CTX = 4096

# Lower temperature = more factual and less creative
OLLAMA_TEMPERATURE = 0.2

# Bound stalled local-model requests and retry transient failures.
OLLAMA_TIMEOUT_SECONDS = 180

OLLAMA_MAX_RETRIES = 2

# Optional
OLLAMA_TOP_P = 0.9

# ==================================================
# CHROMA SETTINGS
# ==================================================

CHROMA_DB_DIR = "./chroma_db"

DOCUMENT_COLLECTION = "documents"

MEMORY_COLLECTION = "memories"

# ==================================================
# DOCUMENT INGESTION
# ==================================================

DOCUMENT_DIR = "./documents"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

# ==================================================
# RETRIEVAL
# ==================================================

# Number of document chunks retrieved
DOCUMENT_RESULTS = 3

# Number of memories retrieved
MEMORY_RESULTS = 5

# Maximum Chroma distance for a remembered fact to be injected. Lower means
# closer match; unrelated saved facts are omitted from the answer prompt.
MEMORY_DISTANCE_THRESHOLD = 0.85

# Number of messages to include in history
CHAT_HISTORY_MESSAGES = 6

# ==================================================
# ROUTER THRESHOLDS
# ==================================================

# Chroma distance thresholds
#
# Lower distance = stronger match
#

# Strong document match
LOCAL_THRESHOLD = 0.85

# Moderate document match
HYBRID_THRESHOLD = 1.20

# ==================================================
# ANSWER EVALUATION - Original=85/60
# ==================================================

# Keep answer local
LOCAL_ACCEPT_THRESHOLD = 65

# Escalate to hybrid if score falls below local threshold
HYBRID_ACCEPT_THRESHOLD = 40

# ==================================================
# MEMORY SCORING - Original=80
# ==================================================

MEMORY_SAVE_THRESHOLD = 60

# ==================================================
# AUTOMATIC BEHAVIOR CONTROLS
# ==================================================

ENABLE_ANSWER_EVALUATION = True

ENABLE_MEMORY_EXTRACTION = True

ENABLE_MEMORY_INJECTION = True

# Automatic hybrid routing requires both switches: hybrid requests use OpenAI.
ENABLE_HYBRID_ESCALATION = True

# Disabling this keeps automatic routing local; explicit /cloud and /hybrid
# overrides remain available to the user.
ENABLE_CLOUD_ESCALATION = True

# ==================================================
# DEBUGGING
# ==================================================

DEBUG_ROUTING = True

DEBUG_RETRIEVAL = True

DEBUG_MEMORY = True

DEBUG_EVALUATION = True

# Logs full prompts, which can include user questions, memories, and documents.
DEBUG_PROMPTS = False

# ==================================================
# UI
# ==================================================

SHOW_SOURCES = True

SHOW_MEMORY_NOTIFICATIONS = True

SHOW_CONFIDENCE_SCORES = True

# ==================================================
# FILE TYPES
# ==================================================

ALLOWED_EXTENSIONS = [
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx"
]
