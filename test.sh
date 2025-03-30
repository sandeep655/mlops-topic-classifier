#!/bin/bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "The game was exciting!"}'
