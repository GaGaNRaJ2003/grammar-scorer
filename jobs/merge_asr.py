"""Union forward + reverse transcript files, dedup by (split, filename); print coverage or write merged."""
import json, sys
paths = ["outputs/transcripts_fw_large_v3.jsonl", "outputs/transcripts_fw_large_v3.rev.jsonl"]
seen = {}
for p in paths:
    try:
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:  # half-written last line while a job is running
                continue
            seen.setdefault((r["split"], r["filename"]), l if l.endswith("\n") else l + "\n")
    except FileNotFoundError:
        pass
if sys.argv[1:] == ["write"]:
    open("outputs/transcripts_fw_large_v3.merged.jsonl", "w", encoding="utf-8").writelines(seen.values())
print(len(seen))
