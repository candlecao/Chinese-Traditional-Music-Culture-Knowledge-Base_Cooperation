import csv
from openai import OpenAI

client = OpenAI(
    api_key="sk-lIjVysUlrOO0Ywpk34FdCa7719C544B4B90e6d316cC68e2f",
    base_url="https://oneapi.xty.app/v1"
)

def callGPT(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to assist me in extracting RDF triples from unstructured text, particularly in the domain of Chinese Music Organology."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

with open("Instrument_context_B.ttl", "r") as f:
    context = f.read()# including ontology, other instruction
with open("example_B.csv", "r") as f:
    example = f.read()# including a non-structural text as corpus, and...
with open("aRightOutcome_B.ttl", "r") as f:
    aRightOutcome = f.read()# an outcome about how one managed to extract RDF triples from that corpus

chunk_size = 40
output_file = "output_outcomeForAnotherTask_3081-(第二轮_补漏).ttl"

def process_chunk(chunk):
    prompt = f"""
    Given the ontology as a context:
    {context}
    
    Here is an example data in CSV format:
    {example}
    --According to the content from the CSV, the first row is heading or attribute. Starting from the second row,
    you will see the name of the instrument, followed by a corresponding specific illustration seperated by a half-width
    comma (This means, as a spreadsheet, it has 2 columns, of which one is "nameOfInstrument", and the other is
    "illustration").
    
    According to the ontology in the context, I can extract RDF info (in format of N-Triples) from the CSV as corpus,
    as shown in the aRightOutcome:
    {aRightOutcome}
    
    Now, here is anotherChunkOfDataToBeExtracted, also in CSV format:
    {chunk}
    --According to the aforementioned context, example, aRightOutcome, please extract RDF data.
    
    In addition to the context as above, there are some other principles you should abide by:
    
    0. The results you feed back should only contain triples. There is no need to include the namespace prefixes.
    
    1. Don't generate duplicated triples. 
    
    2. As to objective properties as dcterms:isPartOf, skos:relatedMatch, because there are no assertions for their 
    domains or ranges, it's inadequate to infer the types of some entities using them.
    
    3. As to some illustration such as:
      3.0 In terms of “的一种”, it possibly indicates a ctm:instrument_broaderTerm relationship; sometimes you may see 
      the relationship from an entry itself, such as "扁八角高音二胡", then you can extract:
      <扁八角高音二胡> ctm:instrument_broaderTerm <二胡> .; another example is as "鼜，古代军中警戒之鼓", then you may 
      extract <鼜> ctm:instrument_broaderTerm <鼓> . In summary, you need to learn to understand and identify the 
      subclass-parentClass relationship from the text and use ctm:instrument_broaderTerm to represent this relationship. 
      3.1 In terms of [<entityAsEntry>, 即“<another entity>”] or [<entityAsEntry>, 同“<another entity>”], a triple can be
       generated as follow: <entityAsEntry> ctm:instrumentAlternateName <another entity> .
      3.2 In terms of 参见“<another entity>” corresponding to an <entity>, a triple can be generated as follow:
      <entity> skos:relatedMatch <another entity> .
      3.3 To summarize additionally, if in an illustration, there is some other instrument entity that is mentioned 
      but not extracted as other specific semantic relations, you may use skos:relatedMatch to link it to the instrument
       entry which the illustration is associated with.
    
    These are only for your reference.
    """
    outcome = callGPT(prompt).strip()
    return outcome

with open("anotherDataToBeExtracted.csv", 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Read the header row
    chunk = []
    for i, row in enumerate(reader, start=1):
        chunk.append(row)
        if i % chunk_size == 0:
            chunk_data = "\n".join([",".join(headers)] + [",".join(r) for r in chunk])
            outcome = process_chunk(chunk_data)
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(outcome + "\n")
            chunk = []
    # Process any remaining rows
    if chunk:
        chunk_data = "\n".join([",".join(headers)] + [",".join(r) for r in chunk])
        outcome = process_chunk(chunk_data)
        with open(output_file, "a", encoding="utf-8") as out_f:
            out_f.write(outcome + "\n")

print("Content written to", output_file)