import os

# set paths and initial steps
root_path = "C:/Users/Bertram/Dropbox/forecast_wiki_traffic"
file_in = root_path + "/summary/Forecast_Wiki_Traffic_SummaryDetailed.md"
file_out = root_path + "/readme.md"

## convert Jupyter summary notebook to Markdown
os.system("jupyter nbconvert --to markdown Forecast_Wiki_Traffic_SummaryDetailed.ipynb")

# read in the raw Markdown
with open(file_in, "r") as f:
    mdcode = f.readlines()

# remove code snippets from being displayed
mdcode_new = []
put_in_mdcode_new = True
for l in mdcode:
    if l == "```python\n":
        put_in_mdcode_new = False
        continue

    if l == "```\n":
        put_in_mdcode_new = True
        continue

    if put_in_mdcode_new:
        mdcode_new.append(l)

# remove table borders
mdcode_new = [l.replace("table border=\"1\"", "table border=\"0\"") for l in mdcode_new]

# re-point paths to image files generated for Markdown
mdcode_new = [l.replace(
    "Forecast_Wiki_Traffic_SummaryDetailed_files/",
    "summary/Forecast_Wiki_Traffic_SummaryDetailed_files/") for l in mdcode_new]

# save new Markdown file
with open(file_out, "w") as f:
    f.writelines(mdcode_new)
