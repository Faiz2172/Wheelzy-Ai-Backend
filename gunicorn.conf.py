# Gunicorn configuration for Render deployment

import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"

# Worker processes
workers = int(os.getenv('WORKERS', 1))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120  # Increased timeout for model initialization

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "ai-car-recommendation"

# Preload app for better performance
preload_app = True 