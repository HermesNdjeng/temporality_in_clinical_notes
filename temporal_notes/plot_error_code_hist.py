from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from typing import List, Dict, Any

# --- Hardcoded issues from GPT-4o ---
issues_gpt4o = [
    # First JSON (disease: BPCO)
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN",  "disease": "BPCO"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "WARN", "disease": "BPCO"},
    # Second JSON (disease: BPCO)
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN",  "disease": "BPCO"},
    # Third JSON (disease: Asthma)
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "Asthma"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "Asthma"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "WARN", "disease": "Asthma"},
    {"code": "EVENT_TIME_MISSING", "severity": "WARN", "disease": "Asthma"},
]

# --- Hardcoded issues from GPT-4o-mini ---
issues_gpt4o_mini = [
    # doc2 vs doc4
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "BEFORE_CYCLE", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN", "disease": "BPCO"},
    # doc3 vs doc4
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "BEFORE_CYCLE", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN", "disease": "BPCO"},
]

# Add this new dataset for dyspnée comparison (doc1 vs doc4)
issues_dyspnee_comparison = [
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "dyspnée"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "ERROR", "disease": "dyspnée"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN", "disease": "dyspnée"},
]

# Update the hardcoded issues to include the dyspnée comparison
issues_gpt4o_updated = [
    # Original GPT-4o issues
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN",  "disease": "BPCO"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "WARN", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "ERROR", "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN",  "disease": "BPCO"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "Asthma"},
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "Asthma"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "WARN", "disease": "Asthma"},
    {"code": "EVENT_TIME_MISSING", "severity": "WARN", "disease": "Asthma"},
    # New dyspnée comparison results
    {"code": "ANCHOR_MISMATCH", "severity": "ERROR", "disease": "dyspnée"},
    {"code": "BEFORE_CHAIN_MISMATCH", "severity": "ERROR", "disease": "dyspnée"},
    {"code": "ANCHOR_MISMATCH", "severity": "WARN", "disease": "dyspnée"},
]

def plot_code_histogram(issues: List[Dict[str, Any]]) -> None:
    code_counts = Counter(issue["code"] for issue in issues)
    total = sum(code_counts.values())
    codes, values = zip(*sorted(code_counts.items()))
    percentages = [v / total * 100 for v in values]

    plt.figure(figsize=(10, 6))
    bars: BarContainer = plt.bar(codes, values, color="skyblue", edgecolor="black")
    plt.xlabel("Error/Warning Code")
    plt.ylabel("Count")
    plt.title("Distribution of Error/Warning Codes")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    for bar, value, pct in zip(bars, values, percentages): #type: ignore
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value}\n({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=10)
    plt.show()

def plot_disease_repartition(issues: List[Dict[str, Any]]) -> None:
    disease_counts: Counter[str] = Counter(issue.get("disease", "Unknown") for issue in issues)
    labels, values = zip(*sorted(disease_counts.items()))
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    plt.figure(figsize=(7, 5))
    bars: BarContainer = plt.bar(labels, values, color="lightgreen", edgecolor="black")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.title("Repartition of Issues by Disease")
    plt.tight_layout()
    for bar, value, pct in zip(bars, values, percentages): #type: ignore
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value}\n({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=10)
    plt.show()

def plot_code_histogram_comparison(issues1: List[Dict[str, Any]], issues2: List[Dict[str, Any]], label1: str = "GPT-4o", label2: str = "GPT-4o-mini"):
    codes = sorted(set([i["code"] for i in issues1] + [i["code"] for i in issues2]))
    counts1 = Counter(i["code"] for i in issues1)
    counts2 = Counter(i["code"] for i in issues2)
    values1 = [counts1.get(code, 0) for code in codes]
    values2 = [counts2.get(code, 0) for code in codes]

    x = range(len(codes))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar([i - width/2 for i in x], values1, width, label=label1, color="skyblue", edgecolor="black")
    bars2 = plt.bar([i + width/2 for i in x], values2, width, label=label2, color="orange", edgecolor="black")
    plt.xlabel("Error/Warning Code")
    plt.ylabel("Count")
    plt.title("Distribution of Error/Warning Codes by Model")
    plt.xticks(x, codes)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)): #type: ignore
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(), str(values1[i]),
                 ha='center', va='bottom', fontsize=10, color="blue")
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(), str(values2[i]),
                 ha='center', va='bottom', fontsize=10, color="darkorange")
    plt.show()

def plot_disease_repartition_comparison(issues1: List[Dict[str, Any]], issues2: List[Dict[str, Any]], label1: str = "GPT-4o", label2: str = "GPT-4o-mini") -> None:
    diseases = sorted(set([i["disease"] for i in issues1] + [i["disease"] for i in issues2]))
    counts1 = Counter(i["disease"] for i in issues1)
    counts2 = Counter(i["disease"] for i in issues2)
    values1 = [counts1.get(d, 0) for d in diseases]
    values2 = [counts2.get(d, 0) for d in diseases]

    x = range(len(diseases))
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars1 = plt.bar([i - width/2 for i in x], values1, width, label=label1, color="lightgreen", edgecolor="black")
    bars2 = plt.bar([i + width/2 for i in x], values2, width, label=label2, color="salmon", edgecolor="black")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.title("Repartition of Issues by Disease and Model")
    plt.xticks(x, diseases)
    plt.legend()
    plt.tight_layout()
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)): #type: ignore
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(), str(values1[i]),
                 ha='center', va='bottom', fontsize=10, color="green")
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(), str(values2[i]),
                 ha='center', va='bottom', fontsize=10, color="red")
    plt.show()

def plot_comprehensive_comparison():
    """Plot comprehensive comparison including dyspnée results"""

    # Update the main function call
    plot_code_histogram(issues_gpt4o_updated)
    plot_disease_repartition(issues_gpt4o_updated)
    plot_code_histogram_comparison(issues_gpt4o_updated, issues_gpt4o_mini)
    plot_disease_repartition_comparison(issues_gpt4o_updated, issues_gpt4o_mini)

    # Additional plot specifically for dyspnée analysis
    plt.figure(figsize=(8, 5))
    dyspnee_codes = Counter(issue["code"] for issue in issues_dyspnee_comparison)
    codes, values = zip(*sorted(dyspnee_codes.items()))

    bars: BarContainer = plt.bar(codes, values, color="lightcoral", edgecolor="black")
    plt.xlabel("Error/Warning Code")
    plt.ylabel("Count")
    plt.title("Dyspnée Comparison Issues (Doc1 vs Doc4)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    for bar, value in zip(bars, values): #type: ignore
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value),
                 ha='center', va='bottom', fontsize=10)
    plt.show()

if __name__ == "__main__":
    # Use updated issues including dyspnée comparison
    plot_code_histogram(issues_gpt4o_updated)
    plot_disease_repartition(issues_gpt4o_updated)
    plot_code_histogram_comparison(issues_gpt4o_updated, issues_gpt4o_mini)
    plot_disease_repartition_comparison(issues_gpt4o_updated, issues_gpt4o_mini)
    plot_comprehensive_comparison()
