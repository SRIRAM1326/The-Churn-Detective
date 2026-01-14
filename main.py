import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n--- Running: {script_name} ---")
    try:
        # Use sys.executable to ensure we use the same environment
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}:")
        print(e.stderr)
        sys.exit(1)

def main():
    print("========================================")
    print("Starting THE CHURN DETECTIVE AI AGENT")
    print("========================================\n")

    # Step 1: Data Generation
    run_script("Data Generation.py")

    # Step 2: Modeling Pipeline (Prediction + Segmentation)
    run_script("modeling_pipeline.py")

    # Step 3: Retention Strategy Engine
    run_script("retention_engine.py")

    # Step 4: Executive Report Generation
    run_script("report_generator.py")

    print("========================================")
    print("PIPELINE COMPLETE - ALL OUTPUTS READY")
    print("========================================")
    print("Outputs generated:")
    print("- Churn_Detective_Executive_Report.pdf")
    print("- Churn_Detective_Executive_Slide.pptx")
    print("- retention_insights.csv")
    print("- priority_retention_list.csv")

if __name__ == "__main__":
    main()
