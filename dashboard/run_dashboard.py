#!/usr/bin/env python3
"""
Dashboard Runner Script
======================

Convenience script to run the Streamlit dashboard with proper configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit dashboard."""
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent
    app_file = dashboard_dir / "app.py"
    
    if not app_file.exists():
        print(f"❌ Dashboard app not found at {app_file}")
        sys.exit(1)
    
    print("🚀 Starting Cross-Lingual QA Dashboard...")
    print(f"📁 Dashboard directory: {dashboard_dir}")
    print(f"🌐 Dashboard will be available at: http://localhost:8501")
    print("📊 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], cwd=dashboard_dir)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
