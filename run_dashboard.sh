#!/bin/bash

# Cross-Lingual QA Dashboard Launcher
# ===================================

echo "🌍 Cross-Lingual QA Dashboard Launcher"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "   Run: ./activate_env.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit and dashboard dependencies..."
    pip install streamlit streamlit-plotly-events streamlit-aggrid streamlit-option-menu
fi

# Check if results exist
if [ ! -d "results" ] || [ -z "$(ls -A results 2>/dev/null)" ]; then
    echo "📊 No results found. Creating sample data for demonstration..."
    python dashboard/create_sample_data.py
fi

# Start the dashboard
echo "🚀 Starting Cross-Lingual QA Dashboard..."
echo "🌐 Dashboard will be available at: http://localhost:8501"
echo "📊 Press Ctrl+C to stop the dashboard"
echo ""

cd dashboard
python run_dashboard.py
