# run.sh
#!/bin/bash

echo "🚀 Deploying Service Aggregator..."

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser
python -m playwright install chromium

# Create necessary directories
mkdir -p data
mkdir -p logs

# Initialize database
python -c "from main import MinimalExtractor; MinimalExtractor()"

echo "✅ Deployment complete!"
echo ""
echo "To start the admin interface:"
echo "streamlit run main.py --server.port 8501"
echo ""
echo "To start the API server:"
echo "python -c 'from main import create_simple_api; create_simple_api().run(host=\"0.0.0.0\", port=5000)'"