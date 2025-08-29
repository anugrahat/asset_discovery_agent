# Evaluation Configuration
# Modify these settings as needed

DISEASES = [
    "Alzheimer's disease",
    "pancreatic cancer", 
    "type 2 diabetes",
    "hypertension"
]

EVALUATION_SETTINGS = {
    "k_values": [5, 10, 20],
    "timeout_seconds": 30,
    "max_retries": 3,
    "cache_results": True
}

API_SETTINGS = {
    "rate_limit_delay": 0.1,  # seconds between API calls
    "batch_size": 10,
    "enable_caching": True
}
