You are indexing a care home's own policy or procedure so a care assistant chatbot can find it later. Read the document and return JSON only: {"title": short title, "summary": one or two plain sentences saying what it covers (no advice of your own), "tags": 3 to 6 short lowercase search terms people would use, "kind": one of ${kinds}, "emergency": only if it is an emergency procedure, one of ${emergency_ids} or null, "text": the document's main points as a list of short plain sentences, in its own words}.

DOCUMENT (${file_name}):
${document}
