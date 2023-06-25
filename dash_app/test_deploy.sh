export OPENAI_API_KEY='OPENAI_API_KEY'
gunicorn synthesis_parser:server -b 0.0.0.0:8000