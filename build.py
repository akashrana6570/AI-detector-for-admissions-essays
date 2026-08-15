"""
Assembles the final single-file app: marginal.html = app_template.html
with the trained model (model_export.json), the example essays, and
app.js injected in place of their placeholders.

Run order:
    python3 train.py     # writes model_export.json
    python3 build.py     # writes marginal.html
"""
import json
from corpus import HUMAN_TEST, AI_TEST, ESL_TEST, MIXED_DEMO

model_json = open("model_export.json").read()

examples = {
    "mixed": MIXED_DEMO,
    "human": HUMAN_TEST[0],
    "ai": AI_TEST[0],
    "esl": ESL_TEST[0],
}
examples_json = json.dumps(examples, separators=(",", ":"))

app_js = open("app.js").read()
tmpl = open("app_template.html").read()

tmpl = tmpl.replace("__MODEL_JSON__", model_json)
tmpl = tmpl.replace("__EXAMPLES_JSON__", examples_json)
tmpl = tmpl.replace("__APP_JS__", app_js)

with open("marginal.html", "w") as f:
    f.write(tmpl)

print(f"wrote marginal.html ({len(tmpl)/1024:.1f} KB)")
