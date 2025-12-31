"""Quick validation test for Stage-2."""
from stage2.trust_validation import run_known_drug_validation

print("=== EGFR Known Drug Validation ===")
results = run_known_drug_validation("EGFR")

for c in results.get("candidates", []):
    name = c.get("compound_name", "Unknown")
    label = c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label", "N/A")
    docking = c.get("stage2", {}).get("docking", {}).get("docking_status", "N/A")
    print(f"  {name}: ADME={label}, Docking={docking}")

print("\n=== BRAF Known Drug Validation ===")
results = run_known_drug_validation("BRAF")

for c in results.get("candidates", []):
    name = c.get("compound_name", "Unknown")
    label = c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label", "N/A")
    docking = c.get("stage2", {}).get("docking", {}).get("docking_status", "N/A")
    print(f"  {name}: ADME={label}, Docking={docking}")

print("\n=== HMG-CoA Reductase Known Drug Validation ===")
results = run_known_drug_validation("HMGCR")

for c in results.get("candidates", []):
    name = c.get("compound_name", "Unknown")
    label = c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label", "N/A")
    docking = c.get("stage2", {}).get("docking", {}).get("docking_status", "N/A")
    print(f"  {name}: ADME={label}, Docking={docking}")

print("\n✅ All known drug validations complete")
