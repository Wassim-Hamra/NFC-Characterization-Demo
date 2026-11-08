import sys
import polars as pl
sys.path.insert(0, '.')

print("=== Import smoke test ===")

# 1. Data generator
from src.data_generator import generate_dataset, save_dataset
df = generate_dataset(n_total=5000)
save_dataset(df, 'data/synthetic_measurements.csv')
print(f"[OK] data_generator: {len(df)} rows, {df['lot_id'].n_unique()} lots")

# 2. Data processor
from src.data_processor import process, validate_upload_schema
cleaned, report = process(df)
print(f"[OK] data_processor: {report.n_input} -> {report.n_output} rows, dupes={report.n_duplicates_removed}")

# 3. Statistics engine
from src.statistics_engine import compute_stats_from_df
import json
with open('data/specifications.json') as f:
    specs = json.load(f)
s = compute_stats_from_df(df, 'load_modulation_mv', lsl=120.0, usl=200.0)
print(f"[OK] statistics_engine: mean={s.mean:.3f}, std={s.std:.3f}, Cp={s.cp}, Cpk={s.cpk}")

# 4. Multimodal analysis
from src.multimodal_analysis import fit_gmm
import numpy as np
lot_d = df.filter(pl.col('lot_id') == 'LOT-D').filter(pl.col('parameter_name') == 'load_modulation_mv')
vals = lot_d['measured_value'].to_numpy()
result = fit_gmm(vals)
print(f"[OK] multimodal_analysis: is_bimodal={result.is_bimodal}, bic_improvement={result.bic_improvement:.1f}")

# 5. Specification validator
from src.specification_validator import compute_parameter_results, compute_lot_parameter_matrix
param_results = compute_parameter_results(df, specs)
print(f"[OK] specification_validator: {len(param_results)} parameters validated")
for r in param_results:
    print(f"     {r.parameter}: pass={r.pass_rate:.3f} status={r.status}")
matrix = compute_lot_parameter_matrix(df, specs)
print("     Lot Matrix summary:")
for row in matrix.filter(pl.col('status') != 'PASS').iter_rows(named=True):
    print(f"       -> {row['lot_id']} | {row['parameter_name']}: pass_rate={row['pass_rate']}% status={row['status']}")

# 6. Insight generator
from src.insight_generator import generate_findings, compute_overall_pass_rate
findings = generate_findings(df, specs)
overall = compute_overall_pass_rate(df, specs)
print(f"[OK] insight_generator: {len(findings)} findings, overall_pass={overall:.4f}")

# 7. RAG engine (no API key needed for BM25)
from src.rag_engine import RAGEngine
rag = RAGEngine()
print(f"[OK] rag_engine: {len(rag.documents)} documents indexed, llm_available={rag.llm_available}")
results = rag.retrieve("bimodal load modulation distribution", k=2)
print(f"     Top result: '{results[0].document.title}' score={results[0].score:.3f}")

print("")
print("=== All modules OK ===")
