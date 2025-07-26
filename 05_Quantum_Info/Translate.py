import nbformat
from openai import OpenAI
from tqdm import tqdm

#model="gpt-4o-mini"  # or "gpt-3.5-turbo"


# 🔐 Your OpenAI API key (replace with your actual key)
API_KEY = "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"

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# 📘 Input and Output notebook paths
INPUT_PATH  = "052_Fun_Info_Cuant.ipynb"
OUTPUT_PATH = "052_Fun_Info_Cuant_en.ipynb"

# 📥 Load the notebook
nb = nbformat.read(INPUT_PATH, as_version=4)

# 🔄 Translate only Markdown cells
for cell in tqdm(nb.cells, desc="Translating Markdown cells"):
    if cell.cell_type == "markdown":
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # You can change to "gpt-4o-mini" or "gpt-3.5-turbo" if needed
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {
                        "role": "user",
                        "content": (
                            "Translate the following Markdown text from Spanish to English. "
                            "Keep **all original markdown formatting** (headings, lists, code blocks, links, etc.) intact.\n\n"
                            f"```markdown\n{cell.source}\n```"
                        ),
                    },
                ],
                temperature=0.0,
            )
            translation = response.choices[0].message.content.strip()

            # Remove ```markdown block if it exists in response
            if translation.startswith("```") and translation.endswith("```"):
                translation = "\n".join(translation.split("\n")[1:-1])

            cell.source = translation

        except Exception as e:
            print(f"⚠️ Error translating cell: {e}")
            # Optionally keep original if error occurs
            continue

# 💾 Save the translated notebook
nbformat.write(nb, OUTPUT_PATH)
print(f"✅ Translation complete. Saved to: {OUTPUT_PATH}")
