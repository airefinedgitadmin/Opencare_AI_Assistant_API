"""Care plan agent (X-CarePlan-Version: 2).

Follows "Care Plan AI integration ppt final.pdf", with the resident's other
notes added as evidence:

* Evidence - the latest assessment, previous care plans for the same pathway,
  review notes, and carer / nursing / external professional / relative notes -
  arrives from the backend as one pack; the agent never fetches data itself.
* Every evidence item gets an id. The model may only group and reword evidence
  and must cite ids; it never writes the quotes shown behind "View" - those are
  the evidence texts as the backend sent them.
* Nothing is cut silently: what the model reads is decided by token budget, and
  notes it didn't read (or judged unrelated to the pathway) are listed.
* Two agents in a fixed order (graph.py): the CARE PLAN AGENT (agent.py) starts
  from a focused pack - must-read records plus each need's best keyword matches
  across all the resident's records (packs.py, search.py) - can search the rest
  of the record, and submits through a tool; code checks each submission and
  sends back the exact problems. The VERIFIER AGENT checks every statement
  against only the quotes it cites and sends unsupported ones back. What still
  isn't supported is dropped and its sources shown word for word.
* Code (not another model) checks the result: cited ids exist, every flagged
  risk / previous action / note requirement is covered or explicitly
  superseded, details dropped while combining are listed, nothing is borrowed
  from a style example, and anything the model missed is added back word for word.
* Staff then Add / Dismiss each suggestion one by one; outcomes follow only for
  added actions. Redraft rewords one card from the same evidence.
"""
