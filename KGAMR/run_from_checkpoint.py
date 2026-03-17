# run_from_checkpoint.py   # <– create this new file or paste into a notebook
import os, json, sys
# ensure workspace root is on path so we can import our modules
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_root)
from modules.kg_construction import trial_2_kg

# 1. load the checkpoint
ckpt = '/scratch/data/r24ab0001/kg_checkpoints/extracted_results.jsonl'
results = []
with open(ckpt) as f:
    for idx,line in enumerate(f):
        item = json.loads(line)
        # each `result` is already dict form; convert back to ExtractionResult
        obj = trial_2_kg.ExtractionResult.model_validate(item['result'])
        results.append((item['impression_id'], obj))
        if idx % 100 == 0:
            print(f"  parsed {idx+1} checkpoint lines... (last id {item['impression_id']})")

print(f"loaded {len(results)} reports from checkpoint")

# 2. link to UMLS
print("linking extracted entities to UMLS...")
# decide where to put the UMLS cache; prefer env var then scratch path
cache_db = os.getenv('UMLS_CACHE_DB', '/scratch/data/r24ab0001/umls_cache.db')
linker = trial_2_kg.UMLSLinker(api_key=trial_2_kg.UMLS_API_KEY, db_path=cache_db)
umls_linked = []
for idx,(impid, res) in enumerate(results):
    if idx % 100 == 0:
        print(f"  linking entities for report {idx}/{len(results)} (id {impid})")
    for t in res.triplets:
        subj = linker.get_cui(t.subject.name)
        obj  = linker.get_cui(t.object.name)
        umls_linked.append({
            "subject": { "mention": t.subject.name,
                         "category": t.subject.category,
                         "umls_cui": subj['cui'] if subj else "N/A",
                         "umls_type": subj['type'] if subj else "N/A"},
            "predicate": t.predicate,
            "object": { "mention": t.object.name,
                        "category": t.object.category,
                        "umls_cui": obj['cui'] if obj else "N/A",
                        "umls_type": obj['type'] if obj else "N/A"},
            "assertion": t.assertion,
            "impression_id": impid
        })

print(f"linked {len(umls_linked)} entities to UMLS (including duplicates)")

# 3. run the validators (returns validation_cache & filtered list)
print("applying medical validation...")
all_valid, df_cache, filtered = trial_2_kg.apply_medical_validator_to_triplets(
                                         umls_linked, impression_id_tag="batch")

# 4. generate weighted triplets for the graph
weighted = trial_2_kg.generate_weighted_triplets(filtered)

# 5. save the results somewhere convenient
import pandas as pd
pd.DataFrame(weighted).to_csv('/scratch/data/r24ab0001/kg_outputs/weighted_triplets.csv',
                             index=False)

print("done – weighted triplets written, validation cache populated")